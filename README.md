## Knowledge Distillations with Pytorch

This repository intergrated various Knowledge Distillation methods. This implementation is based on these repositories:

- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)
- [https://github.com/szagoruyko/attention-transfer](https://github.com/szagoruyko/attention-transfer)
- [https://github.com/lenscloth/RKD](https://github.com/lenscloth/RKD)
- [https://github.com/clovaai/overhaul-distillation](https://github.com/clovaai/overhaul-distillation)
- [https://github.com/HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)

## Distillation Methods in this Repository

- [KD - Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)
- [FN - FitNets: Hints for Thin Deep Nets](https://arxiv.org/pdf/1412.6550.pdf)
- [NST - Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](https://arxiv.org/pdf/1707.01219.pdf)
- [AT - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/pdf/1612.03928.pdf)
- [RKD - Relational Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf)
- [SP - Similarity-Preserving Knowledge Distillation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf)
- [OD - A Comprehensive Overhaul of Feature Distillation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf)

## Dataset
- CIFAR10, CIFAR100

## Model
- ResNet, WideResNet

## Start Distillation
### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy

### Parser Variable description
- type : dataset type (cifar10, cifar100)
- model : network type (resnet, wideresnet)
- depth : depth for resnet and wideresnet (teacher or baseline), sdepth : same for student network
- wfactor : wide factor for wideresnet (teacher or baseline), swfactor : student network
- tn : index number of the multiple trainings (teacher or baseline), stn : same for student network
- distype : type of distillation method (KD, FN, NST, AT, RKD, SP, OD)

### Baseline Training for teacher network or baseline student 
- ex) dataset : cifar100, model: resnet32, index of the number of trainings: 1
```
python3 ./train.py \
--type cifar100 \
--model resnet \
--depth 32 \
--tn 1 \
```
- ex) dataset : cifar100, model: wideresnet16_4, index of the number of trainings: 1
```
python3 ./train.py \
--type cifar100 \
--model wideresnet \
--depth 16 \
--wfactor 4 \
--tn 1 \
```
### Start Distillation
- Hyperparamters for each distillation method are fixed to same values on each original paper
- ex) dataset : cifar100, teacher network : wideresnet16_4, teacher index : 1,  student network : resnet32, student index : 1, index of the number of distillations: 1
```
python3 ./distill.py \
--type cifar100 \
--teacher wideresnet \
--student resnet \
--depth 16 \
--wfactor 4 \
--tn 1 \
--sdepth 32 \
--stn 1 \
--distype KD
```
