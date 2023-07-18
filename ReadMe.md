# Omnipotent Adversarial Training for Unknown Label-noisy and Imbalanced Datasets [[pdf](https://arxiv.org/abs/2307.08596)]

Introduction: This paper introduced a new practical challenge in adversarial training, i.e., how to train a robust model on a label-noisy and imbalanced dataset. Our proposed OAT can effectively address such a problem. Notice: This paper is not related to using adversarial training to solve label noise and long-tail problems. It is actually a new research area.

## Requirements
1. pytorch == 1.12.0
2. torchvision
3. numpy
4. tqdm
5. PIL

## Adversarial Training

``python train.py --arch resnet
--dataset [cifar10, cifar100] --imb [imbalanced ratio] --nr [noise ratio]
--noise_type [sym, asy] --save [the name you want to save your model]
--exp [experiment name]``




