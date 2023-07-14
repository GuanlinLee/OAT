import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import Tensor
import resnet
from tqdm import tqdm
import dataloader
from torchvision import transforms
from augment import CIFAR10Policy



class RandomHorizontalFlip:
	"""Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
	.. note::
		This transform acts out of place by default, i.e., it does not mutate the input tensor.
	Args:
		p (float): probability of an image being flipped.
		inplace(bool,optional): Bool to make this operation in-place.
	"""

	def __init__(self, p=0.5, inplace=False):
		self.p = p
		self.inplace = inplace

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
		Returns:
			Tensor: Randomly flipped Tensor.
		"""
		if not self.inplace:
			tensor = tensor.clone()

		flipped = torch.rand(tensor.size(0)) < self.p
		tensor[flipped] = torch.flip(tensor[flipped], [3])
		return tensor


class RandomCrop:
	"""Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
	Args:
		size (int): Desired output size of the crop.
		padding (int, optional): Optional padding on each border of the image.
			Default is None, i.e no padding.
		device (torch.device,optional): The device of tensors to which the transform will be applied.
	"""

	def __init__(self, size, padding=None, device='cpu'):
		self.size = size
		self.padding = padding
		self.device = device

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
		Returns:
			Tensor: Randomly cropped Tensor.
		"""
		if self.padding is not None:
			padded = torch.zeros((tensor.size(0), tensor.size(1), tensor.size(2) + self.padding * 2,
								  tensor.size(3) + self.padding * 2), dtype=tensor.dtype, device=self.device)
			padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = tensor
		else:
			padded = tensor

		h, w = padded.size(2), padded.size(3)
		th, tw = self.size, self.size
		if w == tw and h == th:
			i, j = 0, 0
		else:
			i = torch.randint(0, h - th + 1, (tensor.size(0),), device=self.device)
			j = torch.randint(0, w - tw + 1, (tensor.size(0),), device=self.device)

		rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:, None]
		columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:, None]
		padded = padded.permute(1, 0, 2, 3)
		padded = padded[:, torch.arange(tensor.size(0))[:, None, None], rows[:, torch.arange(th)[:, None]],
				 columns[:, None]]
		return padded.permute(1, 0, 2, 3)


def build_oracle(args):
	encoder = resnet.resnet18(args.dataset)
	classifier = torch.nn.Linear(encoder.fc.in_features, args.num_classes)
	proj_head = torch.nn.Sequential(torch.nn.Linear(encoder.fc.in_features, 256),
									torch.nn.BatchNorm1d(256),
									torch.nn.ReLU(),
									torch.nn.Linear(256, 128))
	pred_head = torch.nn.Sequential(torch.nn.Linear(128, 256),
									torch.nn.BatchNorm1d(256),
									torch.nn.ReLU(),
									torch.nn.Linear(256, 128))
	encoder.fc = torch.nn.Identity()
	encoder.cuda()
	classifier.cuda()
	proj_head.cuda()
	pred_head.cuda()
	return encoder, classifier, proj_head, pred_head


def build_dataloader(args):
	batch_size = args.batch_size
	transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									])
	strong_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
										   transforms.RandomHorizontalFlip(),
										   CIFAR10Policy(),
										   transforms.ToTensor()])
	transform_test = transforms.Compose([transforms.Resize((32, 32)),
										 transforms.ToTensor(),
										 ])
	#Create Noisy Dataset
	train_data_oracle = dataloader.cifar_dataset(dataset=args.dataset, root_dir='./data',
												 transform=KCropsTransform(strong_transform, 2),
												 dataset_mode='train', noise_ratio=args.nr, noise_mode=args.noise_type)
	eval_data_oracle = dataloader.cifar_dataset(dataset=args.dataset, root_dir='./data', transform=transform,
												dataset_mode='train', noise_ratio=args.nr, noise_mode=args.noise_type)
	all_data_oracle = dataloader.cifar_dataset(dataset=args.dataset, root_dir='./data',
											   transform=MixTransform(strong_transform=strong_transform,
																	  weak_transform=transform, K=1),
											   dataset_mode='train', noise_ratio=args.nr, noise_mode=args.noise_type)
	trainset, testset = dataloader.get_dataset(args, transform, transform_test)
	train_data_oracle.cifar_data = eval_data_oracle.cifar_data = all_data_oracle.cifar_data = trainset.data
	train_data_oracle.cifar_label = eval_data_oracle.cifar_label = all_data_oracle.cifar_label = trainset.targets
	train_data_oracle.clean_label = eval_data_oracle.clean_label = all_data_oracle.clean_label = trainset.clean_labels

	#Create LT Dataset
	if args.dataset == 'cifar10':
		train_lt = dataloader.IMBALANCECIFAR10(imbalance_ratio=args.imb, random_seed=args.seed)
	elif args.dataset == 'cifar100':
		train_lt = dataloader.IMBALANCECIFAR100(imbalance_ratio=args.imb, random_seed=args.seed)
	train_lt.targets = trainset.targets
	train_lt.clean_labels = trainset.clean_labels
	if args.imb < 1.0:
		train_lt.gen_imbalanced_data(train_lt.img_num_list)
	trainset.data = train_lt.data
	trainset.targets = np.asarray(train_lt.targets)
	trainset.clean_labels = np.asarray(train_lt.clean_labels)
	if args.imb < 1.0:
		train_lt.upsample_new_dataset()
	train_data_oracle.cifar_data = eval_data_oracle.cifar_data = all_data_oracle.cifar_data = train_lt.data
	train_data_oracle.cifar_label = eval_data_oracle.cifar_label = all_data_oracle.cifar_label = np.asarray(train_lt.targets)
	train_data_oracle.clean_label = eval_data_oracle.clean_label = all_data_oracle.clean_label = np.asarray(train_lt.clean_labels)

	targets = np.asarray(trainset.targets)
	clean_targets = torch.from_numpy(trainset.clean_labels).cuda()
	noisy_label = torch.tensor(eval_data_oracle.cifar_label).cuda()
	data_size = len(trainset.data)

	ori_data_idx = train_lt.ori_data_idx if args.imb < 1.0 else None
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=4)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											 shuffle=False, num_workers=4)
	eval_loader = torch.utils.data.DataLoader(eval_data_oracle, batch_size=batch_size, shuffle=False, num_workers=4,
											  pin_memory=True)
	all_loader = torch.utils.data.DataLoader(all_data_oracle, batch_size=batch_size, num_workers=4, shuffle=True,
											 pin_memory=True, drop_last=True)
	return targets, clean_targets, noisy_label, data_size, ori_data_idx, trainloader, testloader, eval_loader, all_loader, train_data_oracle


class KCropsTransform:
	"""Take K random crops of one image as the query and key."""

	def __init__(self, base_transform, K=2):
		self.base_transform = base_transform
		self.K = K

	def __call__(self, x):
		res = [self.base_transform(x) for i in range(self.K)]
		return res


class MixTransform:
	def __init__(self, strong_transform, weak_transform, K=2):
		self.strong_transform = strong_transform
		self.weak_transform = weak_transform
		self.K = K

	def __call__(self, x):
		res = [self.weak_transform(x) for i in range(self.K)] + [self.strong_transform(x) for i in range(self.K)]
		return res


def D(p, z, version='simplified'):  # negative cosine similarity
	if version == 'original':
		z = z.detach()  # stop gradient
		p = F.normalize(p, dim=1)  # l2-normalize
		z = F.normalize(z, dim=1)  # l2-normalize
		return -(p * z).sum(dim=1).mean()

	elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
		return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
	else:
		raise Exception


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
	# compute cos similarity between each feature vector and feature bank ---> [B, N]

	sim_matrix = torch.mm(feature, feature_bank)#cos similarity
	# [B, K]

	sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)  # top k cos similarity
	sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
	sim_weight = torch.ones_like(sim_weight)
	sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

	# counts for each class
	one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
	# [B*K, C]
	one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
	# weighted score ---> [B, C]
	pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

	pred_labels = pred_scores.argmax(dim=-1)
	return pred_scores, pred_labels


def knn(cur_feature, feature, label, num_classes, knn_k=200, chunks=10):
	# distributed fast KNN and sample selection with three different modes
	num = len(cur_feature)
	num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(
			feature.device) + 1e-10
	pi = num_class / num_class.sum()
	split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
	score = torch.tensor([]).to(feature.device)
	pred = torch.tensor([], dtype=torch.long).to(feature.device)
	feature = F.normalize(feature, dim=1)
	with torch.no_grad():
		for i in range(chunks):
			torch.cuda.empty_cache()
			part_feature = cur_feature[split[i]: split[i + 1]]

			part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)

			score = torch.cat([score, part_score], dim=0)
			pred = torch.cat([pred, part_pred], dim=0)

		# balanced vote
		score = score / pi

		score = score/score.sum(1, keepdim=True)

	return score


def evaluate(dataloader, encoder, classifier, args, noisy_label, ori_data_idx):
	encoder.eval()
	classifier.eval()
	feature_bank = []
	prediction = []
	################################### feature extraction ###################################
	with torch.no_grad():
		# generate feature bank
		for (data, target, _, index) in tqdm(dataloader, desc='Feature Extracting',
							ncols=0):
			data = data.cuda()
			feature = encoder(data)
			feature_bank.append(feature)
			res = classifier(feature)
			prediction.append(res)
		feature_bank = F.normalize(torch.cat(feature_bank, dim=0), dim=1)

		################################### sample relabelling ###################################
		prediction_cls = torch.softmax(torch.cat(prediction, dim=0), dim=1)
		his_score, his_label = prediction_cls.max(1)
		conf_id = torch.where(his_score > args.theta_r)[0]
		modified_label = torch.clone(noisy_label).detach()
		modified_label[conf_id] = his_label[conf_id]

		################################### sample selection ###################################

		prediction_knn = knn(feature_bank, feature_bank, modified_label, args.num_classes, args.k,
									  10)  # temperature in weighted KNN

		vote_y = torch.gather(prediction_knn, 1, modified_label.view(-1, 1)).squeeze()
		vote_max = prediction_knn.max(dim=1)[0]
		right_score = vote_y / vote_max
		clean_id = torch.where(right_score >= 1.0)[0]
		noisy_id = torch.where(right_score < 1.0)[0]
		if ori_data_idx is not None:
			clean_id_ori = list(set(clean_id.cpu().numpy())&set(ori_data_idx))
		else:
			clean_id_ori = clean_id.clone()
	return clean_id, clean_id_ori, noisy_id, modified_label


def train(labeled_trainloader, modified_label, all_trainloader, encoder, classifier, proj_head, pred_head, optimizer, args, model):
	encoder.train()
	classifier.train()
	proj_head.train()
	pred_head.train()
	model.eval()
	labeled_train_iter = iter(labeled_trainloader)
	all_bar = tqdm(all_trainloader, desc='Oracle Training',
							ncols=0)
	for batch_idx, ([inputs_u1, inputs_u2], _, _, fidx) in enumerate(all_bar):
		try:
			[inputs_x1, inputs_x2], labels_x, _, index = labeled_train_iter.next()
		except:
			labeled_train_iter = iter(labeled_trainloader)
			[inputs_x1, inputs_x2], labels_x, _, index = labeled_train_iter.next()
		# cross-entropy training with mixup
		batch_size = inputs_x1.size(0)

		inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
		labels_x = modified_label[index]
		targets_x = torch.zeros(batch_size, args.num_classes, device=inputs_x1.device).scatter_(1, labels_x.view(-1, 1), 1)
		l = np.random.beta(4, 4)
		l = max(l, 1 - l)
		all_inputs_x = torch.cat([inputs_x1, inputs_x2], dim=0)
		all_targets_x = torch.cat([targets_x, targets_x], dim=0)
		idx = torch.randperm(all_inputs_x.size()[0])
		input_a, input_b = all_inputs_x, all_inputs_x[idx]
		target_a, target_b = all_targets_x, all_targets_x[idx]
		mixed_input = l * input_a + (1 - l) * input_b
		mixed_target = l * target_a + (1 - l) * target_b

		logits1 = classifier(encoder(mixed_input))
		Lce = - torch.mean(torch.sum(F.log_softmax(logits1, dim=1) * mixed_target, dim=1))
		if args.it == 1:
			logits2 = model(mixed_input).detach()
			Lce -= F.mse_loss(torch.softmax(logits1, dim=1), torch.softmax(logits2, dim=1))

		# optional feature-consistency
		inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()

		feats_u1 = encoder(inputs_u1)
		feats_u2 = encoder(inputs_u2)
		f, h = proj_head, pred_head

		z1, z2 = f(feats_u1), f(feats_u2)
		p1, p2 = h(z1), h(z2)

		Lfc = D(p2, z1)

		loss = Lce + Lfc
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

