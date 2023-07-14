import copy
import pickle
import torchvision as tv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision

def get_dataset(args, transform_train, transform_test):
	# prepare datasets
	#################################### Train set #############################################
	if (args.dataset == 'cifar10'):
		cifar_train = Cifar10Train(args, train=True, transform=transform_train, target_transform=transform_test,
								   download=True)
	else:
		cifar_train = Cifar100Train(args, train=True, transform=transform_train, target_transform=transform_test,
									download=True)
	#################################### Noise corruption ######################################

	if args.noise_type == "sym":
		cifar_train.random_in_noise()

	elif args.noise_type == "asy":
		cifar_train.real_in_noise()

	else:
		print('No noise')
	cifar_train.labelsNoisyOriginal = cifar_train.targets.copy()

	#################################### Test set #############################################
	if (args.dataset == 'cifar10'):
		testset = tv.datasets.CIFAR10(root='./data', train=False, download=True,
									  transform=transform_test)
	else:
		testset = tv.datasets.CIFAR100(root='./data', train=False, download=True,
									   transform=transform_test)
	###########################################################################################

	return cifar_train, testset


class Cifar10Train(tv.datasets.CIFAR10):
	def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
		super(Cifar10Train, self).__init__('./data', train=train, transform=transform,
										   target_transform=target_transform, download=download)
		self.root = './data'
		self.transform = transform
		self.target_transform = target_transform
		self.train = train  # Training set or validation set

		self.args = args
		if sample_indexes is not None:
			self.data = self.data[sample_indexes]
			self.targets = list(np.asarray(self.targets)[sample_indexes])

		self.num_classes = self.args.num_classes
		self.in_index = []
		self.out_index = []
		self.noisy_indexes = []
		self.clean_indexes = []
		self.clean_labels = []
		self.noisy_labels = []
		self.out_data = []
		self.out_labels = []
		self.soft_labels = []
		self.labelsNoisyOriginal = []
		self._num = []
		self._count = 1
		self.prediction = []
		self.confusion_matrix_in = np.array([])
		self.confusion_matrix_out = np.array([])
		self.labeled_idx = []
		self.unlabeled_idx = []

		# From in ou split function:
		self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
		self._num = int(len(self.targets) * self.args.nr)

	################# Random in-distribution noise #########################
	def random_in_noise(self):
		# to be more equal, every category can be processed separately
		np.random.seed(self.args.seed)
		idxes = np.random.permutation(len(self.targets))
		clean_labels = np.copy(self.targets)
		noisy_indexes = idxes[0:self._num]
		clean_indexes = idxes[self._num:]
		for i in range(len(idxes)):
			if i < self._num:
				self.soft_labels[idxes[i]][
					self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
				# targets[idxes[i]] -> another category
				label_sym = np.random.randint(self.num_classes, dtype=np.int32)
				# while(label_sym==self.targets[idxes[i]]):#To exclude the original label
				# label_sym = np.random.randint(self.num_classes, dtype=np.int32)
				self.targets[idxes[i]] = label_sym
			self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

		self.targets = np.asarray(self.targets, dtype=np.long)
		self.noisy_labels = np.copy(self.targets)
		self.noisy_indexes = noisy_indexes
		self.clean_labels = clean_labels
		self.clean_indexes = clean_indexes
		self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(
			self.args.num_classes)) \
								   * (self.args.nr / (self.num_classes - 1)) + \
								   np.identity(self.args.num_classes) * (1 - self.args.nr)
		print('clean_num', sum(self.noisy_labels == self.clean_labels))

	##########################################################################

	################# Real in-distribution noise #########################

	def real_in_noise(self):
		# to be more equal, every category can be processed separately
		np.random.seed(self.args.seed)

		##### Create te confusion matrix #####

		self.confusion_matrix_in = np.identity(self.args.num_classes)

		# truck -> automobile
		self.confusion_matrix_in[9, 9] = 1 - self.args.nr
		self.confusion_matrix_in[9, 1] = self.args.nr

		# bird -> airplane
		self.confusion_matrix_in[2, 2] = 1 - self.args.nr
		self.confusion_matrix_in[2, 0] = self.args.nr

		# cat -> dog
		self.confusion_matrix_in[3, 3] = 1 - self.args.nr
		self.confusion_matrix_in[3, 5] = self.args.nr

		# dog -> cat
		self.confusion_matrix_in[5, 5] = 1 - self.args.nr
		self.confusion_matrix_in[5, 3] = self.args.nr

		# deer -> horse
		self.confusion_matrix_in[4, 4] = 1 - self.args.nr
		self.confusion_matrix_in[4, 7] = self.args.nr

		idxes = np.random.permutation(len(self.targets))
		clean_labels = np.copy(self.targets)

		for i in range(len(idxes)):
			self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
			current_label = self.targets[idxes[i]]
			if self._num > 0:
				# current_label = self.targets[idxes[i]]
				conf_vec = self.confusion_matrix_in[current_label, :]
				label_sym = np.random.choice(np.arange(0, self.num_classes), p=conf_vec.transpose())
				self.targets[idxes[i]] = label_sym
			else:
				label_sym = current_label

			self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

			if label_sym == current_label:
				self.clean_indexes.append(idxes[i])
			else:
				self.noisy_indexes.append(idxes[i])

		self.targets = np.asarray(self.targets, dtype=np.long)
		self.clean_indexes = np.asarray(self.clean_indexes, dtype=np.long)
		self.noisy_indexes = np.asarray(self.noisy_indexes, dtype=np.long)
		self.noisy_labels = self.targets
		self.clean_labels = clean_labels
		print('clean_num', sum(self.noisy_labels == self.clean_labels))

	def __getitem__(self, index):
		if self.train:
			img, labels = self.data[index], self.targets[index]

			img = Image.fromarray(img)
			if self.transform is not None:
				img = self.transform(img)

			return img, labels, index

		else:
			img, labels = self.data[index], self.targets[index]
			# doing this so that it is consistent with all other datasets.
			img = Image.fromarray(img)
			if self.transform is not None:
				img = self.transform(img)

			return img, labels


class Cifar100Train(tv.datasets.CIFAR100):
	def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
		super(Cifar100Train, self).__init__('./data', train=train, transform=transform,
											target_transform=target_transform, download=download)
		self.root = './data'
		self.transform = transform
		self.target_transform = target_transform
		self.train = train  # Training set or validation set

		self.args = args
		if sample_indexes is not None:
			self.data = self.data[sample_indexes]
			self.targets = list(np.asarray(self.targets)[sample_indexes])

		self.num_classes = self.args.num_classes
		self.in_index = []
		self.out_index = []
		self.noisy_indexes = []
		self.clean_indexes = []
		self.clean_labels = []
		self.noisy_labels = []
		self.out_data = []
		self.out_labels = []
		self.soft_labels = []
		self.labelsNoisyOriginal = []
		self._num = []
		self._count = 1
		self.prediction = []
		self.confusion_matrix_in = np.array([])
		self.confusion_matrix_out = np.array([])
		self.labeled_idx = []
		self.unlabeled_idx = []

		# From in ou split function:
		self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
		self._num = int(len(self.targets) * self.args.nr)

	def random_in_noise(self):

		# to be more equal, every category can be processed separately
		np.random.seed(self.args.seed)
		idxes = np.random.permutation(len(self.targets))
		clean_labels = np.copy(self.targets)
		noisy_indexes = idxes[0:self._num]
		clean_indexes = idxes[self._num:]
		for i in range(len(idxes)):
			if i < self._num:
				self.soft_labels[idxes[i]][
					self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
				# train_labels[idxes[i]] -> another category
				label_sym = np.random.randint(self.num_classes, dtype=np.int32)
				# while(label_sym==self.targets[idxes[i]]):#To exclude the original label
				# label_sym = np.random.randint(self.num_classes, dtype=np.int32)
				self.targets[idxes[i]] = label_sym
			self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

		self.targets = np.asarray(self.targets, dtype=np.long)
		self.noisy_labels = np.copy(self.targets)
		self.noisy_indexes = noisy_indexes
		self.clean_labels = clean_labels
		self.clean_indexes = clean_indexes
		self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(
			self.args.num_classes)) \
								   * (self.args.nr / (self.num_classes - 1)) + \
								   np.identity(self.args.num_classes) * (1 - self.args.nr)
		print('clean_num', sum(self.noisy_labels == self.clean_labels))

	##########################################################################

	################# Asymmetric noise #########################

	def real_in_noise(self):
		# to be more equal, every category can be processed separately
		np.random.seed(self.args.seed)

		##### Create te confusion matrix #####

		self.confusion_matrix_in = np.identity(self.args.num_classes) * (1 - self.args.nr)

		idxes = np.random.permutation(len(self.targets))
		clean_labels = np.copy(self.targets)

		with open(self.root + '/cifar-100-python/train', 'rb') as f:
			entry = pickle.load(f, encoding='latin1')

		coarse_targets = np.asarray(entry['coarse_labels'])

		targets = np.array(self.targets)
		num_subclasses = self.args.num_classes // 20

		for i in range(20):
			# embed()
			subclass_targets = np.unique(targets[coarse_targets == i])
			clean = subclass_targets
			noisy = np.concatenate([clean[1:], clean[:1]])
			for j in range(num_subclasses):
				self.confusion_matrix_in[clean[j], noisy[j]] = self.args.nr

		for t in range(len(idxes)):
			self.soft_labels[idxes[t]][self.targets[idxes[t]]] = 0  ## Remove soft-label created during label mapping
			current_label = self.targets[idxes[t]]
			conf_vec = self.confusion_matrix_in[current_label, :]
			label_sym = np.random.choice(np.arange(0, self.num_classes), p=conf_vec.transpose())
			self.targets[idxes[t]] = label_sym
			self.soft_labels[idxes[t]][self.targets[idxes[t]]] = 1

			if label_sym == current_label:
				self.clean_indexes.append(idxes[t])
			else:
				self.noisy_indexes.append(idxes[t])

		self.targets = np.asarray(self.targets, dtype=np.long)
		self.clean_indexes = np.asarray(self.clean_indexes, dtype=np.long)
		self.noisy_indexes = np.asarray(self.noisy_indexes, dtype=np.long)
		self.noisy_labels = self.targets
		self.clean_labels = clean_labels
		print('clean_num', sum(self.noisy_labels == self.clean_labels))

	def __getitem__(self, index):
		if self.train:
			img, labels = self.data[index], self.targets[index]

			img = Image.fromarray(img)
			if self.transform is not None:
				img = self.transform(img)

			return img, labels, index

		else:
			img, labels = self.data[index], self.targets[index]
			# doing this so that it is consistent with all other datasets.
			img = Image.fromarray(img)
			if self.transform is not None:
				img = self.transform(img)

			return img, labels


def unpickle(file):
	import _pickle as cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo, encoding='latin1')
	return dict

class cifar_dataset(Dataset):
	def __init__(self, dataset, root_dir, transform, noise_mode='sym',
				 dataset_mode='train', noise_ratio=0.5):

		self.r = noise_ratio  # total noise ratio
		self.transform = transform
		self.mode = dataset_mode
		self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
		self.open_noise = None
		self.closed_noise = None

		if self.mode == 'test':
			if dataset == 'cifar10':
				cifar_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
				self.cifar_data = cifar_dic['data']
				self.cifar_data = self.cifar_data.reshape((10000, 3, 32, 32))
				self.cifar_data = self.cifar_data.transpose((0, 2, 3, 1))
				self.cifar_label = cifar_dic['labels']
			elif dataset == 'cifar100':
				cifar_dic = unpickle('%s/cifar-100-python/test' % root_dir)
				self.cifar_data = cifar_dic['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
				self.cifar_label = cifar_dic['fine_labels']

		elif self.mode == 'train':
			if dataset == 'cifar10':
				cifar_data = []
				cifar_label = []
				for n in range(1, 6):
					dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
					data_dic = unpickle(dpath)
					cifar_data.append(data_dic['data'])
					cifar_label = cifar_label + data_dic['labels']
				self.cifar_data = np.concatenate(cifar_data).reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

			elif dataset == 'cifar100':
				cifar_dic = unpickle('%s/cifar-100-python/train' % root_dir)
				cifar_label = cifar_dic['fine_labels']
				self.cifar_data = cifar_dic['data'].reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
			self.clean_label = cifar_label

			# inject noise
			noise_labels = []  # all labels (some noisy, some clean)
			idx = list(range(50000))  # indices of cifar dataset
			random.shuffle(idx)
			num_total_noise = int(self.r * 50000)  # total amount of noise
			self.closed_noise = idx[:num_total_noise]  # closed set noise indices
			# populate noise_labels
			for i in range(50000):
				if i in self.closed_noise:
					if noise_mode == 'sym':
						if dataset == 'cifar10':
							noiselabel = random.randint(0, 9)
						elif dataset == 'cifar100':
							noiselabel = random.randint(0, 99)
					elif noise_mode == 'asy':
						noiselabel = self.transition[cifar_label[i]]
					noise_labels.append(noiselabel)
				else:
					noise_labels.append(cifar_label[i])
			self.cifar_label = noise_labels
			# self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None

	def update_labels(self, new_label):
		self.cifar_label = new_label.cpu()

	def __getitem__(self, index):
		# print(index)
		if self.mode == 'train':
			img = self.cifar_data[index]
			img = Image.fromarray(img)
			img = self.transform(img)
			target = self.cifar_label[index]
			clean_target = self.clean_label[index]
			return img, target, clean_target, index
		else:
			img = self.cifar_data[index]
			img = Image.fromarray(img)
			img = self.transform(img)
			target = self.cifar_label[index]
			return img, target, index

	def __len__(self):
		return len(self.cifar_data)

	def get_noise(self):
		return (self.open_noise, self.closed_noise)

	def __repr__(self):
		return f'dataset_mode: {self.mode}, dataset number: {len(self)} \n'

	def get_data(self, idx):
		img = self.cifar_data[idx]
		img = Image.fromarray(img)
		img = self.transform(img)
		return img

random_seed = 0
train_sampler_type = 'default'
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
	cls_num = 10

	def __init__(self, imbalance_ratio=0.02, root='./data', train=True, imb_type='exp',
				 transform=None, target_transform=None, download=True, random_seed=0):
		mode = "train" if train else "evaluation"
		super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
		self.train = train
		rand_number = random_seed
		if self.train:
			np.random.seed(rand_number)
			random.seed(rand_number)
			imb_factor = imbalance_ratio
			self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
			#self.gen_imbalanced_data(self.img_num_list)
			self.transform = transform
		else:
			self.transform = transform
		if train_sampler_type == "weighted sampler" and self.train:
			self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
			self.class_dict = self._get_class_dict()

		self.clean_labels = np.copy(self.targets)

	def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
		img_max = len(self.data) / cls_num
		img_num_per_cls = []
		if imb_type == 'exp':
			for cls_idx in range(cls_num):
				num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
				img_num_per_cls.append(int(num))
		elif imb_type == 'step':
			for cls_idx in range(cls_num // 2):
				img_num_per_cls.append(int(img_max))
			for cls_idx in range(cls_num // 2):
				img_num_per_cls.append(int(img_max * imb_factor))
		else:
			img_num_per_cls.extend([int(img_max)] * cls_num)
		return img_num_per_cls

	def sample_class_index_by_weight(self):
		rand_number, now_sum = random.random() * self.sum_weight, 0
		for i in range(self.cls_num):
			now_sum += self.class_weight[i]
			if rand_number <= now_sum:
				return i

	def reset_epoch(self, cur_epoch):
		self.epoch = cur_epoch

	def _get_class_dict(self):
		class_dict = dict()
		for i, anno in enumerate(self.get_annotations()):
			cat_id = anno["category_id"]
			if not cat_id in class_dict:
				class_dict[cat_id] = []
			class_dict[cat_id].append(i)
		return class_dict

	def get_weight(self, annotations, num_classes):
		num_list = [0] * num_classes
		cat_list = []
		for anno in annotations:
			category_id = anno["category_id"]
			num_list[category_id] += 1
			cat_list.append(category_id)
		max_num = max(num_list)
		class_weight = [max_num / i for i in num_list]
		sum_weight = sum(class_weight)
		return class_weight, sum_weight

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)


		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target, index

	def get_num_classes(self):
		return self.cls_num

	def reset_epoch(self, epoch):
		self.epoch = epoch

	def get_annotations(self):
		annos = []
		for target in self.targets:
			annos.append({'category_id': int(target)})
		return annos

	def get_cls_num_list(self):
		cls_num_list = []
		for i in range(self.cls_num):
			cls_num_list.append(self.num_per_cls_dict[i])
		return cls_num_list

	def gen_imbalanced_data(self, img_num_per_cls):
		new_data = []
		new_targets = []
		new_clean_targets = []
		targets_np = np.array(self.targets, dtype=np.int64)
		clean_targets_np = np.array(self.clean_labels, dtype=np.int64)
		self.classes = np.unique(targets_np)
		# np.random.shuffle(classes)
		self.num_per_cls_dict = dict()
		self.num_per_cls_list = []
		idsx = []
		for the_class, the_img_num in zip(self.classes, img_num_per_cls):
			# print(the_img_num, end=' ')
			idx = np.where(targets_np == the_class)[0]
			np.random.shuffle(idx)
			selec_idx = idx[:the_img_num]
			#print(selec_idx.shape, the_img_num)
			new_data.append(self.data[selec_idx, ...])
			self.num_per_cls_dict[the_class] = self.data[selec_idx, ...].shape[0]
			self.num_per_cls_list.append(self.data[selec_idx, ...].shape[0])
			new_targets.extend([the_class, ] * self.data[selec_idx, ...].shape[0])
			new_clean_targets.extend(clean_targets_np[selec_idx].tolist())
			idsx.append(selec_idx)
		# print()

		new_data = np.vstack(new_data)
		self.data = new_data
		self.targets = new_targets
		self.clean_labels = new_clean_targets
		#print(self.data.shape)
		#print(len(self.targets))
		#print(len(self.clean_labels))
		#print(self.get_cls_num_list())
		#print(sum(self.get_cls_num_list()))
		#exit(0)
	def upsample_new_dataset(self):
		new_data = []
		new_targets = []
		new_clean_targets = []
		targets_np = np.array(self.targets, dtype=np.int64)
		clean_targets_np = np.array(self.clean_labels, dtype=np.int64)
		upper_bound = max(self.num_per_cls_list)
		self.num_per_cls_list_after_upsample = [0] * self.classes
		self.ori_data_idx = []
		for class_idx in self.classes:
			idx = np.where(targets_np == class_idx)[0].tolist()
			len_idx = len(idx)
			if len_idx == upper_bound:
				new_data.append(self.data[idx, ...])
				new_targets.extend([class_idx, ] * self.data[idx, ...].shape[0])
				new_clean_targets.extend(clean_targets_np[idx].tolist())
				self.num_per_cls_list_after_upsample[class_idx] = self.data[idx, ...].shape[0]
			else:
				selec_idx = copy.deepcopy(idx)
				scale = upper_bound // len(idx) + 1
				idx = idx * scale
				np.random.shuffle(idx)
				selec_idx += idx[:(upper_bound-len(selec_idx))]
				new_data.append(self.data[selec_idx, ...])
				new_targets.extend([class_idx, ] * self.data[selec_idx, ...].shape[0])
				new_clean_targets.extend(clean_targets_np[selec_idx].tolist())
				self.num_per_cls_list_after_upsample[class_idx] = self.data[selec_idx, ...].shape[0]
			self.ori_data_idx.extend([(i + class_idx * upper_bound) for i in range(len_idx)])
		new_data = np.vstack(new_data)
		self.data = new_data
		self.targets = new_targets
		self.clean_labels = new_clean_targets

class IMBALANCECIFAR100(IMBALANCECIFAR10):
	"""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
	This is a subclass of the `CIFAR10` Dataset.
	"""
	base_folder = 'cifar-100-python'
	url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
	filename = "cifar-100-python.tar.gz"
	tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
	train_list = [
		['train', '16019d7e3df5f24257cddd939b257f8d'],
	]

	test_list = [
		['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
	]
	meta = {
		'filename': 'meta',
		'key': 'fine_label_names',
		'md5': '7973b15100ade9c7d40fb424638fde48',
	}
	cls_num = 100
