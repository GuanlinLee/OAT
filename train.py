import torch
import torch.utils.data
import resnet
from torch import nn
import torchattacks
from tqdm import tqdm
import logging
import random
import os
import numpy as np
import argparse
import utils
import loss_functions
from torch.optim.lr_scheduler import MultiStepLR
import time
from datetime import timedelta
from logging import getLogger
from torch.utils.data import Subset
import sampler

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
					help='model architecture')
parser.add_argument('--dataset', default='cifar10', type=str,
					help='which dataset used to train')
parser.add_argument('--num_classes', default=10, type=int, metavar='N',
					help='number of classes')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='wd')
parser.add_argument('--save', default='OAT.pkl', type=str,
					help='model save name')
parser.add_argument('--seed', type=int,
					default=0, help='random seed')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--eps', type=float, default=8./255., help='perturbation bound')
parser.add_argument('--ns', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--ss', type=float, default=2./255., help='step size')
parser.add_argument('--beta', type=float, default=6.0)


parser.add_argument('--exp', default='OAT', type=str,
					help='exp name')
parser.add_argument('--method', default='OAT', type=str,
					help='AT method to use')

##Noise Setting
parser.add_argument('--nr', type=float, default=0.0,
					help='noisy ratio for dataset')
parser.add_argument('--noise_type', type=str, default='sym',
					help='type of label noise')

##Long Tail Setting
parser.add_argument('--imb', type=float, default=0.01,
					help='imbalance ratio for dataset')


##Oracle Setting
parser.add_argument('--theta_r', default=0.8, type=float, help='threshold for relabelling samples (default: 0.8)')
parser.add_argument('--k', default=200, type=int, help='neighbors for knn sample selection (default: 100)')

parser.add_argument('--it', default=1, type=int, help='if interaction between oracle and model')
parser.add_argument('--bal', default=1, type=int, help='if using class weights to balance Knn')

args = parser.parse_args()

if args.dataset == 'cifar10':
	args.num_classes = 10
else:
	args.num_classes = 100

class LogFormatter:
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime("%x %X"),
			timedelta(seconds=elapsed_seconds),
		)
		message = record.getMessage()
		message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
		return "%s - %s" % (prefix, message) if message else ""
def create_logger(filepath, rank):
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	if filepath is not None:
		if rank > 0:
			filepath = "%s-%i" % (filepath, rank)
		file_handler = logging.FileHandler(filepath, "a")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	if filepath is not None:
		logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)


#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
setup_seed(args.seed)
NCOLs = 0

logger = getLogger()
if not os.path.exists(args.dataset+'/'+ args.arch +'/'+args.exp):
	os.makedirs(args.dataset+'/'+ args.arch +'/'+args.exp)
logger = create_logger(
	os.path.join(args.dataset+'/'+ args.arch +'/'+args.exp + '/', args.exp + ".log"), rank=0
)
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)
args.save = args.dataset+'/'+ args.arch +'/'+args.exp + '/' +  args.save


wd=args.wd
learning_rate=args.lr
epochs=args.epochs
batch_size=args.batch_size


torch.backends.cudnn.benchmark = True

targets, clean_targets, noisy_label, data_size, ori_data_idx, trainloader, testloader, eval_loader, all_loader, train_data_oracle = utils.build_dataloader(args)



n = resnet.resnet18(args.dataset).cuda()
f_d = 512


encoder, classifier, proj_head, pred_head = utils.build_oracle(args)
optimizer = torch.optim.SGD(n.parameters(),momentum=args.momentum,
							lr=learning_rate,weight_decay=wd)
optimizer_surrgate = torch.optim.SGD([{'params': encoder.parameters()}, {'params': classifier.parameters()}, {'params': proj_head.parameters()}, {'params': pred_head.parameters()}],
                    		lr=learning_rate, momentum=args.momentum,weight_decay=wd)


milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

if args.method == 'OAT':
	oat = loss_functions.OAT(targets, num_samples=data_size,
					   num_classes=args.num_classes,
					   step_size=args.ss,
					   epsilon=args.eps,
					   perturb_steps=args.ns,
					   norm='linf')


train_clean_acc_nl = []
train_adv_acc_nl = []
train_clean_acc_cl = []
train_adv_acc_cl = []
test_clean_acc = []
test_adv_acc = []
label_pred_acc = []
sorted_pred = [np.zeros((args.num_classes,)),] * args.epochs
num_class_list_total = []
best_eval_acc = 0.0
weights_gt = [torch.sum(torch.as_tensor(clean_targets, dtype=torch.int) == i).item() for i in
		   range(args.num_classes)]
num_class_list_total.append(weights_gt)


for epoch in range(epochs):

	clean_id, clean_id_ori, noisy_id, modified_label = utils.evaluate(eval_loader, encoder, classifier, args, noisy_label, ori_data_idx)

	# balanced_sampler
	clean_subset = Subset(train_data_oracle, clean_id.cpu())

	sampler_oracle = sampler.ClassBalancedSampler(labels=modified_label[clean_id], num_classes=args.num_classes)

	labeled_loader = torch.utils.data.DataLoader(clean_subset, batch_size=batch_size,sampler=sampler_oracle,
											 	num_workers=4, drop_last=True)

	utils.train(labeled_loader, modified_label, all_loader, encoder, classifier, proj_head, pred_head,
				optimizer_surrgate, args, n)

	encoder.eval()
	classifier.eval()
	pred_label = torch.zeros((data_size), dtype=torch.long).cuda()
	for x, y, idx in trainloader:
		pred = classifier(encoder(x.cuda())).detach()
		pred_label[idx] = torch.argmax(pred, dim=1)
	weights = [torch.sum(torch.as_tensor(pred_label, dtype=torch.int) == i).item() for i in
			   range(args.num_classes)]
	num_class_list_total.append(weights)
	print(weights_gt, weights)

	loadertrain = tqdm(trainloader, desc='{} E{:03d}'.format('train', epoch),
							ncols=NCOLs)
	epoch_loss = 0.0
	total=0.0
	clean_acc_nl = 0.0
	adv_acc_nl = 0.0

	clean_acc_cl = 0.0
	adv_acc_cl = 0.0

	label_pred_acc_ = 0.0

	for x_train, y_train, idx in loadertrain:
		n.eval()
		x_train, y_train = x_train.cuda(), y_train.cuda()
		y_pre = n(x_train)
		#adversarial training in a semi-supervised fashion
		logits_adv, loss, x_adv, pred_sort = oat(x_train, y_train, idx, n, optimizer, encoder, classifier, proj_head, pred_head, args, weights)
		loss.backward()
		optimizer.step()

		epoch_loss += loss.data.item()
		_, predicted = torch.max(y_pre.data, 1)
		_, predictedadv = torch.max(logits_adv.data, 1)
		total += y_train.size(0)
		sorted_pred[epoch] += pred_sort
		clean_acc_nl += predicted.eq(y_train.data).cuda().sum()
		adv_acc_nl += predictedadv.eq(y_train.data).cuda().sum()
		clean_acc_cl += predicted.eq(clean_targets[idx].data).cuda().sum()
		adv_acc_cl += predictedadv.eq(clean_targets[idx].data).cuda().sum()
		fmt = '{:.4f}'.format
		loadertrain.set_postfix(loss=fmt(epoch_loss / total * batch_size),
								acc_clean_nl=fmt(clean_acc_nl.item() / total * 100),
								acc_adv_nl=fmt(adv_acc_nl.item() / total * 100),
								acc_clean_cl=fmt(clean_acc_cl.item() / total * 100),
								acc_adv_cl=fmt(adv_acc_cl.item() / total * 100))

	#update clean set prediction
	soft_labels = oat.soft_labels

	label_pred = torch.argmax(soft_labels, dim=1)
	label_pred_acc_ = label_pred.eq(clean_targets.data).sum()
	sorted_pred[epoch] /= data_size
	label_pred_acc.append(label_pred_acc_.item() / data_size * 100)
	train_clean_acc_nl.append(clean_acc_nl.item() / total * 100)
	train_adv_acc_nl.append(adv_acc_nl.item() / total * 100)
	train_clean_acc_cl.append(clean_acc_cl.item() / total * 100)
	train_adv_acc_cl.append(adv_acc_cl.item() / total * 100)
	scheduler.step()

	if (epoch) % 1 == 0:
		Loss_test = nn.CrossEntropyLoss().cuda()
		test_loss_cl = 0.0
		test_loss_adv = 0.0
		correct_cl = 0.0
		correct_adv = 0.0
		total = 0.0
		n.eval()
		pgd_eval = torchattacks.PGD(n, eps=8.0/255.0, steps=20)
		loadertest = tqdm(testloader, desc='{} E{:03d}'.format('test', epoch),
							ncols=NCOLs)
		with torch.enable_grad():
			for x_test, y_test in loadertest:
				x_test, y_test = x_test.cuda(), y_test.cuda()
				x_adv = pgd_eval(x_test, y_test)
				n.eval()
				y_pre = n(x_test)
				y_adv = n(x_adv)
				loss_cl = Loss_test(y_pre, y_test)
				loss_adv = Loss_test(y_adv, y_test)
				test_loss_cl += loss_cl.data.item()
				test_loss_adv += loss_adv.data.item()
				_, predicted = torch.max(y_pre.data, 1)
				_, predicted_adv = torch.max(y_adv.data, 1)
				total += y_test.size(0)
				correct_cl += predicted.eq(y_test.data).cuda().sum()
				correct_adv += predicted_adv.eq(y_test.data).cuda().sum()
				fmt = '{:.4f}'.format
				loadertest.set_postfix(loss_cl=fmt(loss_cl.data.item()),
									   loss_adv=fmt(loss_adv.data.item()),
									   acc_cl=fmt(correct_cl.item() / total * 100),
									   acc_adv=fmt(correct_adv.item() / total * 100),
									   label_pred_acc=label_pred_acc_.item() / data_size * 100)
			test_clean_acc.append(correct_cl.item() / total * 100)
			test_adv_acc.append(correct_adv.item() / total * 100)
		if correct_adv.item() / total * 100 > best_eval_acc:
			best_eval_acc = correct_adv.item() / total * 100
			checkpoint = {
					'state_dict': n.state_dict(),
					'epoch': epoch
				}
			torch.save(checkpoint, args.save+ 'best.pkl')
checkpoint = {
			'state_dict': n.state_dict(),
			'epoch': epoch
		}

torch.save(checkpoint, args.save + 'last.pkl')

np.save(args.save+'_train_acc_clean_nl.npy', train_clean_acc_nl)
np.save(args.save+'_train_acc_adv_nl.npy', train_adv_acc_nl)
np.save(args.save+'_train_acc_clean_cl.npy', train_clean_acc_cl)
np.save(args.save+'_train_acc_adv_cl.npy', train_adv_acc_cl)
np.save(args.save+'_test_acc_clean.npy', test_clean_acc)
np.save(args.save+'_test_acc_adv.npy', test_adv_acc)
np.save(args.save+'_label_pred_acc.npy', label_pred_acc)
np.save(args.save+'_sorted_pred.npy', sorted_pred)
torch.save(num_class_list_total, args.save+'_num_class.pt')