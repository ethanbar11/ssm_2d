# Pytorch Implementation of [2-D SSM: A General Spatial Layer for Visual Transformers](https://arxiv.org/pdf/2306.06635.pdf)

![Alt text](main_fig.png?raw=true "")

## Repository Structure

There are 2 directories in this repository, each one containing a different implementation base of 2-D SSM. 
### With Patches
The code based on the paper "Vision Transformer for Small-Size Datasets" [1]  that implements the Swin and ViT versions of the 2-D SSM and able to reproduce the experiments on CIFAR-100 and Tiny Imagenet.

### Without Patches 
Based on the Mega [2] repository and is able to reproduce the original CIFAR-10 grayscale isotropic experiment.









<a id="1">[1] Vision Transformer for Small-Size Datasets, Seung Hoon Lee, Seunghyun Lee, Byung Cheol Song, https://arxiv.org/abs/2112.13492</a>
<a id="2">[2] Moving Average Equipped Gated Attention by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, Luke Zettlemoyer https://arxiv.org/abs/2209.10655</a>