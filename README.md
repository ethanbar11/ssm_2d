
# A 2-Dimensional State Space Layer for Spatial Inductive Bias

This repository is a Pytorch implementation of the paper.
![Alt text](main_fig.png?raw=true "")

## Installation
Verify you have the PyTorch installed (it is working with 1 or 2).
Then run:

```bash
pip install --r requirments.txt
```


# Experiments Recreation

## Supported datasets

- CIFAR-10
- CIFAR-100
- CIFAR-100-224px - CIFAR-100 images enhanced to 224x224, so it would be compatible with some of the architectures.
- Celebs-A
- ImageNet-100 
- Tiny-ImageNet  
- ImageNet-1k



## Vit Backbones

### baselines:
```bash
python main.py --model vit --dataset CIFAR100
python main.py --model vit --dataset T-IMNET
python main.py --model swin --dataset CIFAR100 --embed_dim 96
python main.py --model swin --dataset T-IMNET --embed_dim 96
python main.py --model mega --dataset CIFAR100 --ema {choice}
```
### Our Runs

- none (default,no Q&K aggregation mechanism)
- ema (1d ema)
- ssm_2d (ssm_2d)


```bash
python main.py --model vit --dataset CIFAR100 --no_pos_embedding --use_mix_ffn --ema ssm_2d --normalize --n_ssm=2 --ndim 16 --directions_amount 2 --seed 0
python main.py --model swin --dataset CIFAR100 --ema {choice} --use_mega_gating --embed_dim 96
python main.py --model mega --dataset CIFAR100 --ema ssm_2d --n_ssm 8 --ndim 16
```

## ConvNext

ConvNext for small datasets

To create original results from: https://juliusruseckas.github.io/ml/convnext-cifar10.html
Notice original results are with batch-size = 128

```bash
python main.py --model convnext-small --dataset CIFAR10 --lr 1e-3 --batch_size 128 --weight-decay 1e-1
python main.py --model convnext-small --dataset T-IMNET --lr 1e-3 --batch_size 128 --weight-decay 1e-1
```

#### SSM Real:

```bash
python main.py --model convnext-small --dataset CIFAR100 --lr 1e-3 --batch_size 128 --weight-decay 1e-1 --ema ssm_2d --ssm_kernel_size 9 --n_ssm 2 --directions_amount 4 --ndim 16
python main.py --model convnext-small --dataset T-IMNET --lr 1e-3 --batch_size 128 --weight-decay 1e-1 --ema ssm_2d --ssm_kernel_size 13 --n_ssm 2 --directions_amount 2 --ndim 16
```

#### SSM Complex:

```bash
python main.py --model convnext-small --dataset CIFAR100 --lr 1e-3 --batch_size 128 --weight-decay 1e-1 --ema ssm_2d --ssm_kernel_size 9 --n_ssm 2 --directions_amount 2 --ndim 16 --complex_ssm
python main.py --model convnext-small --dataset T-IMNET --lr 1e-3 --batch_size 128 --weight-decay 1e-1 --ema ssm_2d --ssm_kernel_size 7 --n_ssm 2 --directions_amount 2 --ndim 16 --complex_ssm
```

