from torch.autograd import Variable
import torch
import torch.nn.functional as F
import utils

class OAT():
	def __init__(self, labels, num_samples=50000, num_classes=10, step_size=0.003, epsilon=0.031,
				 perturb_steps=10, norm='linf'):
		# initialize soft labels to onehot vectors
		print('number samples: ', num_samples, 'num_classes: ', num_classes)
		self.soft_labels = torch.zeros(num_samples, num_classes, dtype=torch.float).cuda(non_blocking=True)
		self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
		self.step_size = step_size
		self.epsilon = epsilon
		self.perturb_steps = perturb_steps
		self.norm = norm
		self.label = labels
		self.num_classes = num_classes

	def __call__(self, x_natural, y, index, model, optimizer, encoder, classifier, proj_head, pred_head, args, num_class_list=None):
		model.eval()
		encoder.eval()
		classifier.eval()
		pred_head.eval()
		proj_head.eval()
		spc = torch.tensor(num_class_list).type_as(x_natural).view(1, -1) + 1.0
		spc = spc.log()
		with torch.no_grad():
			y_surrgate = torch.softmax(classifier(encoder(x_natural)).detach(), dim=1)
			if args.it == 1:
				z1 = proj_head(encoder(x_natural)).detach()
		self.soft_labels[index] = y_surrgate
		# generate adversarial example
		if self.norm == 'linf':
			x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-self.epsilon,
																					  self.epsilon).cuda()
		else:
			x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
		for _ in range(self.perturb_steps):
			x_adv.requires_grad_()
			with torch.enable_grad():
				logits_adv = model(x_adv)
				if args.bal == 1:
					logits_adv += spc
				loss = F.cross_entropy(logits_adv, torch.argmax(self.soft_labels[index], dim=1).long())
			grad = torch.autograd.grad(loss, [x_adv])[0]
			if self.norm == 'linf':
				x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
				x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
			elif self.norm == 'l2':
				g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
				scaled_grad = grad.detach() / (g_norm.detach() + 1e-10)
				x_adv = x_natural + (x_adv.detach() + self.step_size * scaled_grad - x_natural).view(x_natural.size(0),
																									 -1).renorm(p=2,
																												dim=0,
																												maxnorm=self.epsilon).view_as(
					x_natural)
			x_adv = torch.clamp(x_adv, 0.0, 1.0)

		# compute loss
		model.train()
		optimizer.zero_grad()
		x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

		# calculate robust loss
		f_adv, logits = model(x_adv, True)

		if args.bal == 1:
			loss = torch.sum(-F.log_softmax(logits+spc, dim=1) * self.soft_labels[index], dim=1).mean()
		else:
			loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1).mean()
		if args.it == 1:
			loss += utils.D(pred_head(proj_head(f_adv)), z1)
		return logits, loss, x_adv, torch.sort(y_surrgate, dim=1, stable=True)[0].sum(0).cpu().numpy()


