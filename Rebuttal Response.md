# To ALL

We thank the reviewers for the extremely helpful feedback, which greatly
improves our manuscript.

**Potential Impact:** Our work has three main goals: (a) incorporating a
two-dimensional positional bias into vision transformers, (b) designing
SSM-based global convolutions for capturing long-range multi-dimensional
dependencies, and (c) developing a spatial layer with several unique
properties and strong image-specific inductive bias.

For the first goal, our method has the potential to be a
backbone-agnostic general-purpose add-on for a wide range of vision
transformers, not just ViT and MEGA.

\(i\) Our method is now shown to boost **Swin transformers**: we
integrate our layer into the Swin transformer in a straightforward
manner and show it can boost its performance with a negligible amount of
additional parameters and without any hyperparameter tuning. We obtained
this improvement by using our original 2D-SSM layer, without any need to
use the MEGA components (such as the gating mechanism, Laplace
attention, or the K & Q manipulation ). For more details, see below.

\(ii\) Attention and SSMs-based convolution are considered complementary
components \[1-5\], and this is a well-known practice in NLP \[1-4\],
Speech \[1,3,5\], FMRI \[1\], and more. For example, the SSM-based H3
\[1\] outperforms GPT-Neo-2.7B (as well as other transformers of the
same size) with only 2 attention layers, and in all the referenced works
hybrid models outperform transformers.

\(iii\) Our method can be easily integrated into a wide range of ViT
variants since it can operate on any sequence of patches with a
plug-and-play approach. For example, we can insert our layer before each
block of the following ViT variants \[16,17\], in the corresponding
lines:
\[[l1](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py#L277),[l2](https://github.com/google-research/maxvit/blob/main/maxvit/models/maxvit.py#L930)\]

The second goal involves designing SSM-based global convolutions. This
type of convolutions are effective in capturing long-range dependencies
and achieving SOTA results in many tasks, such as NLP \[1,2,4,5,12\],
speech \[5,13\], RL \[10,11\], audio generation \[14\], graphs \[15\],
fMRI \[1\], time series, and many more. Inspired by S4 \[12\], a second
line of research involves the design of regularized global convolution
\[6-8\], which is a promising domain. For example, the very recent Hyena
hierarchy \[8\] has almost closed the performance gap with Transformers
on several NLP tasks given the same amount of FLOP budget.

In the context of these two promising areas, we believe that our 2D-SSM
layer has a unique contribution since it models multi-dimensional
sequences directly (in contrast to S4ND, which separately models the
behavior in each dimension). Therefore, we think that our
parametrization, relaxation for spatial bias, stability mechanisms,
kernelization, and proofs (sections 3.2, 3.1, 4.1 in our paper) are a
valuable contribution.

Regarding the third goal, please refer to the first point in the novelty
section in this thread.

**Additional empirical analysis:**

::: {#tab:results_vit}
  **Name**          **\# of Parameters**   **Training Time**   **CIFAR100**   **T-ImageNet**
  ----------------- ---------------------- ------------------- -------------- ----------------
  Vit \[9\]         2.71M (1x)             1x                  76.36          57.07
  Vit (Ours runs)   2.71M (1x)             1x                  73.08          56.53
  Mega Ablate       2.75M (1.015x)         1.1x                74.82          56.43
  Mega EMA          2.98M (1.1x)           1.28x               72.27          54.49
  Mega 2-D SSM      2.80M (1.03x)          2.36x               **76.02**      **57.95**
:::

[]{#tab:results_vit label="tab:results_vit"}

::: {#tab:results_swin}
  **Name**           **\# of Parameters**   **Training Time**   **CIFAR100**   **T-ImageNet**
  ------------------ ---------------------- ------------------- -------------- ----------------
  Swin \[9\]         7.15M (1x)             1x                  76.87          60.87
  Swin (Ours runs)   7.15M (1x)             1x                  77.98          61.29
  Swin w. EMA        7.52M (1.05x)          1.39x               77.01          60.13
  Swin w. SSM        7.25M (1.01x)          2.16x               **80.12**      **65.42**

  : For both tables: The number of parameters and the training time are
  based on the model applied to CIFAR100 and are similar to those of
  Tiny ImageNet. For the baselines, we used the same hyper-parameters as
  \[9\].
:::

::: {#tab:results}
  **Name**            **Nu. of Parameters**   **Imagenet**      
  ------------------- ----------------------- -------------- -- --
  Mega-Tiny Ablate    5.91M                   66.97             
  Mega-Tiny EMA       6.21M                   69.73             
  Mega-Tiny 2-D SSM   5.96M                   70.11             

  : Ablating The Gating Mechanism
:::

Following the reviews, we conducted additional experiments: (i) applied
our method to the Swin transformer backbone, (ii) analyzed the
importance of each component of our model and how they affect the
performance (iii) tested the 2-D SSM layer incorporated in Mega & Swin
on two more datasets (CIFAR100 and Tiny ImageNet).

SSM with Swin

:   To examine the sole effect of the 2D-SSM layer, we incorporated it
    into Swin without the gating mechanism or other components of MEGA.
    We applied it to the entire patch map X prior to partitioning it
    into windows. The following equation illustrates the process:
    $$\hat{X} = SSM_{2D}(X)$$ Attention is performed, as usual, on the
    transformed patch map $\hat{x}$. In contrast to the original MEGA,
    where the SSM layer was applied only to Q and K, we choose to
    incorporate the 2-D SSM in the most straightforward manner.

Ablate Mega without EMA/SSM 2-D

:   For ablation purposes, we ran MEGA without EMA or SSM. Thus, the
    ablated model is ViT with only 1 head, the MEGA gating mechanism and
    relative positional encoding.

**Novelty:**

Our work presents a new spatial layer (2D-SSM) that has several unique
properties such as: (i) A strong inductive bias towards
**two-dimensional neighborhood and locality**, which stems from the
multi-dimensional recurrent rule (see Eq. 14). As far as we know, this
is a novel concept that does not appear in other computer vision layers,
(ii) The new layer can capture **unrestricted controllable context**.
The SSM parameters of the layer can be focused on short or long, and
horizontal, vertical, or diagonal dependencies (see the kernels in Fig.
2 and Fig. 4 (middle) of the submitted manuscript). (iii) The layer has
parameter efficiency and can express kernels of any length via 8 scalars
($A_1,A_2,A_3,A_4,B_1,B_2,C_1,C_2$). Finally, (iv) The layer is
well-grounded by control theory.

**Novelty in comparison to MEGA:** **Novelty in incorporating
image-specific inductive bias into ViT:**

While the bulk of our contributions is independent of MEGA, which is
made clearer with the Swin transformer experiments, an important
contribution is extending the MEGA concept to multidimensional data.

We respectfully disagree with the reviewers that extending the EMA or
SSM layer to multi-dimensional data is trivial. First, there are many
ways to define multi-dimensional EMA and multi-dimensional SSMs.
Applying those recurrent models in deep learning is a challenging task
that arise several problems such as numerical stability issues,
infeasible computation, restricted expressiveness, limited capacity, and
more.

In this paper, we propose several technical contributions to deal with
those problems, including (i) using diagonal parametrization to solve
the computation problems (while diagonal parameterization is not new, in
our case it is not trivial, see 3.2.1). (ii), handling numerical
stability via stable parameterization and stable intra-kernel
normalization (3.2.2), (iii). relaxation of the kernel to achieve
spatial bias (3.2.2), and (iv.) computation of the complex efficiently
by the Alg. in appendix A (see \"Creating the Kernel and Forward
Pass\").

**Computation:** Following the review, we provide a complexity analysis
of our approach, where we compare the number of parameters,
computational complexity during inference and training, and wall-clock
measurements against the baseline methods. Our approach aims to be
parameter-efficient, and Table X demonstrates that we achieve improved
results with fewer parameters than Mega.

During inference, both Mega and 2D-SSM can pre-compute the convolution
kernels, resulting in an additional computation cost of only the
convolution of the layer input and the signal. On Mega, the input image
is treated as a sequence of length L and convoluted with a kernel of
length L, which has a time complexity of O($L \log L$) operations. On
2D-SSM, a multidimensional sequence of dimensions $L_1 \times L2$ is
convoluted with a corresponding kernel of the same size, taking
O($L_1  L2$ log($L_1  L_2$)) operations, which can be simplified to
O($L \log L$) since $L_1  L_2 = L$. Therefore, both Mega and 2D-SSM have
the same time complexity. Experiments on modern GPUs demonstrate that
2D-SSM is slightly faster than Mega

The additional computation overhead relative to the regular ViT is
minimal, as the quadratic complexity dominates the overall complexity.
In terms of training time, see the limitation section in this response.

\(5\) Additional requested sections

#### Limitations

Despite the promising results presented in this paper, there are several
limitations that should be considered. First, the current implementation
of our proposed layer has relatively slow training times, as shown by
the wall-clock measurements presented in the experimental section in
this response. This slow training time may be even more pronounced when
applying our method to a longer two-dimensional sequence of patches,
which could limit its applicability to tasks that require handling
multi-dimensional long-range dependencies.

One possible approach to mitigating this challenge is to use
multi-dimensional parallel scanners, which could potentially reduce the
training time of our layer. The main idea is to extend the work of S5,
which leverages 1-D parallel scanners to apply SSM on 1-D sequences to
multi-dimensional parallel scanners and multi-dmsional multi-dimensional
sequences.

Another limitation of this work is that while our proposed layer shows
promising results on several datasets and vision transformer backbones,
due to limited resources, we were not able to test the proposed layer on
additional backbones.

For reproducibility, the code for our experiments has been published in
an anonymized repository. It can be found at
[Link](https://anonymous.4open.science/r/ssm_2d_submission-E3F2).\
\
**References:**\
\
$[1]$ Dao, Tri, et al. \"Hungry Hungry Hippos: Towards Language Modeling
with State Space Models.\" arXiv preprint arXiv:2212.14052 (2022).\
\
$[2]$ Mehta, Harsh, et al. \"Long range language modeling via gated
state spaces.\" arXiv preprint arXiv:2206.13947 (2022).\
\
$[3]$ Ma, Xuezhe, et al. \"Mega: moving average equipped gated
attention.\" arXiv preprint arXiv:2209.10655 (2022).\
\
$[4]$ Zuo, Simiao, et al. \"Efficient Long Sequence Modeling via State
Space Augmented Transformer.\" arXiv preprint arXiv:2212.08136 (2022).\
\
$[5]$ Saon, George, Ankit Gupta, and Xiaodong Cui. \"Diagonal State
Space Augmented Transformers for Speech Recognition.\" arXiv preprint
arXiv:2302.14120 (2023).\
\
$[6]$ Li, Yuhong, et al. \"What Makes Convolutional Models Great on Long
Sequence Modeling?.\" arXiv preprint arXiv:2210.09298 (2022).\
\
$[7]$ Fu, Daniel Y., et al. \"Simple Hardware-Efficient Long
Convolutions for Sequence Modeling.\" arXiv preprint arXiv:2302.06646
(2023).\
\
$[8]$ Poli, Michael, et al. \"Hyena Hierarchy: Towards Larger
Convolutional Language Models.\" arXiv preprint arXiv:2302.10866
(2023).\
\
$[9]$ Lee, Seung Hoon, Seunghyun Lee, and Byung Cheol Song. \"Vision
transformer for small-size datasets.\" arXiv preprint arXiv:2112.13492
(2021).\
\
$[10]$ Shmuel Bar David, Itamar Zimerman, Eliya Nachmani, Lior Wolf.
\"Decision S4: Efficient Sequence-Based RL via State Spaces Layers.\"
https://openreview.net/pdf?id=kqHkCVS7wbj (2023).\
\
$[11]$ Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob
Foerster, Satinder Singh, Feryal Behbahani. \"Structured State Space
Models for In-Context Reinforcement Learning.\"
https://arxiv.org/pdf/2303.03982.pdf (2023).\
\
$[12]$ Junxiong Wang, Jing Nathan Yan, Albert Gu, Alexander M. Rush.
\"Pretraining Without Attention.\"https://arxiv.org/abs/2212.10544
(2022).\
\
$[13]$ Albert Gu, Karan Goel, Christopher Ré. \"Efficiently Modeling
Long Sequences with Structured State
Spaces.\"https://arxiv.org/abs/2111.00396 (2022).\
\
$[14]$ Karan Goel, Albert Gu, Chris Donahue, Christopher Ré. \"It's Raw!
Audio Generation with State-Space Models.\"
https://arxiv.org/abs/2202.09729 (2022).\
\
$[15]$ Ahmed El Gazzar, Rajat Mani Thomas, Guido Van Wingen. Improving
the Diagnosis of Psychiatric Disorders with Self-Supervised Graph State
Space Models. https://arxiv.org/abs/2206.03331\
\
$[16]$ Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei,
Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo. Swin
Transformer V2: Scaling Up Capacity and Resolution.
https://arxiv.org/abs/2111.09883

$[17]$ Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman
Milanfar, Alan Bovik, Yinxiao Li. IMaxViT: Multi-Axis Vision
Transformer. https://arxiv.org/abs/2204.01697

# To R1(96ST)

We thank the reviewer for the valuable feedback which helped us to
improve the manuscript.\
\
Q1. "The proposed method is largely based on the previous MEGA model.
Although some new designs are proposed, the similar overall architecture
and settings indeed limit the contribution of this paper... limited
overall contribution of the proposed method"\
\
We address this concern on the Novelty section in the main response.
Furthermore, the Swin transformer experiments don't use any components
of the MEGA method, and the additional boost in performance comes from
our 2-D SSM layer.

Q2. \"... Since MEGA is not a popular model in vision tasks, the impact
of this paper may also be limited if the method can only be used in
MEGA..."\
\
Firstly, MEGA is very new (Accepted to ICLR 2023) and thus didn't have
time to become prominent as a model for vision tasks. Secondly, we
address this concern by enhancing the Swin Transformer with the 2-D SSM
layer and demonstrating significant improvement over the baseline.

Q3. "The experimental studies are insufficient and relatively weak.
..... weak and insufficient experimental study"\
\
We thank the reviewer for bringing this to our attention, we address
this concern on the Additional empirical analysis section in the main
response. In short, we thus showed how incorporating 2-D SSM into Swin
Transformer shows promising results, and tested Mega 2D-SSM and Swin 2-D
SSM against more baselines and ablations.

Q4. \"Will the method be effective on a wider range of vision
Transformers?"\
\
We reference the reviewer to the results on Swin Transformer 2-D SSM.
Furthermore, we anticipate incorporating the method with most of the ViT
backbones will be quite straightforward, for more details see a.iii in
Potential Impact.

Q5. \"In the ImageNet experiment, the method is only compared to MEGA
with the ViT/DeiT-T architecture and the result is lower than DeiT-T\"\
\
The main reason that DeiT-T outperforms our method is that it uses a
unique type of student-teacher knowledge distillation procedure. The
technique is based on a custom distillation token and leveraging
ConvNets for the teacher model. Those techniques are orthogonal to our
method, and therefore we consider DeiT-T as irrelevant baselines.
Furthermore, we want to emphasize that we use the default set of
hyperparameters from the MEGA repository, which are optimized for larger
models, and are sub-optimal by definition.

Q6. \"Besides, in the MEGA paper, only the result of the MEGA-B model is
reported. I think it would be better to directly compare with this
result to clearly show the effectiveness of the method."\
\
Unfortunately, due to computational constraints, we weren't able to run
the full Mega-Base model with 2-D SSM.

Q7. \"State-of-the-art vision Transformers are usually based on a
hierarchical design and incorporate local convolutional layers to
improve performance. It seems the proposed method is not useful for
these models. Therefore, it is not clear why we need the proposed
method.\
\
We argue that our method is orthogonal to those concerns, and prove it
by showing how we can easily incorporate our method into Swin
Transformer and other ViT variants in Potential impact section (ii).

Q8. \"computational complexity compared to the baseline MEGA and ViT
models?"\
\
Reported above in section Complexity of the main response. We would also
like to emphasize that with a small number of additional parameters and
negligible additional time at inference, our method drastically improves
over the baseline. For example, on Tiny ImageNet, by increasing the
number of parameters by 0.1%, our method boost the accuracy of the Swin
Transformer model from 60.87 to 65.42.

Q9. Typos, Grammar and Writing issues:\
\
Thank you for this comment. We would send our paper to another round of
professional proofreading, and we will make every effort to ensure that
the paper is accurate and error-free.

# To R2 (4xiL)

We thank the reviewer for the valuable feedback which helped us to
improve the manuscript.\
\
Q1. \"The proposed formulation combines previous work and is
conceptually not very novel."\
\
We address this concern on the Novelty section in the main response.

Q2. \"Not all parts in the model are well motivated. For example, what
is the benefit of the additionally introduces gates?"\
\
The gating mechanism isn't part of our method. We just inserted our 2-D
SSM layer into the MEGA backbone since it's a natural baseline. Please
note that in the Swin transformer experiments, we don't use any
components of the MEGA method, and the additional boost in performance
comes from our 2-D SSM layer.

Q3. "The results improve over the baseline MEGA but there is not
comparison to previously introduced spatial inductive biases, such as
the one from Swin Transformer v2:\..."\
\
As we argue in (a.iii) in the Potential Impact section of the main
response, our method is complementary and orthogonal to those methods.
For example, we can integrate our method into the Swin Transformer v2 by
adding the following line of code. We agree with the reviewer that this
is a valuable future research direction.

Q4. "The comparison to MEGA in table 3 indicates that the proposed
method improves over MEGA, however, it does not report the number of
model parameters, so it is not exactly clear where the improvement comes
from. .."\
\
Thanks. When compared to the original Mega with EMA, 2-D SSM always uses
fewer parameters (due to parameter sharing, and efficiently models
two-dimensional sequences) as shown on the results in Tab. For example,
on Tiny ImageNet, by increasing the number of parameters by 0.1%, our
method boost the accuracy of the Swin Transformer model from 60.87 to
65.42.

Q5. "Could you please ablate the complex model? Would the real part
alone perform well?\
\
We appreciate these insightful questions! We performed the following
ablations on the CIFAR100 & Tiny ImageNet datasets:

-   Mega without SSM/EMA: We ablated the performance of the model by
    removing the SSM/EMA layer before the gating mechanism. The results
    showed significant improvement when using the SSM mechanism over
    both of the baselines. For instance, on Imagenet, accuracy drops
    from 70.11 to 66.97 when removing the 2-D SSM components.

    Also, we integrate our method into the Swin transformer in the
    following 3 manners:

-   Simply apply the 2-D SSM layer, without the gating mechanism,
    Laplace attention, $Q \& K$ manipulation, or any mega components on
    the input of each Swin transformer block.

-   Apply our 2-D SSM layer only on the $Q \& K$ before each Swin
    transformer block.

-   Replaced the Swin attention blocks with the Mega layer, with the EMA
    or SSM as sub-layers.

    Surprisingly, we observe that the first method, which doesn't use
    any of the MEGA components and is the most straight forward to
    integrate achieves much better results than the other alternatives
    on Cifar-100 and Tiny-Imagenet. The last option doesn't match the
    original performance of the Swin transformer, and the second option
    slightly improves the Swin baseline. Please note that for all of
    those experiments, the 2D-SSM outperforms the EMA.

Q6. "What is the increase in parameters/training time over MEGA?\
\
Please refer to the new tables in the main response. It can be seen that
the additional number of parameters is negligible, and the model always
has fewer parameters than the original MEGA. For example, on Imagenet,\
\
Q8. "There is no dedicated limitations section\"\
\
Thanks, it has been added to the main response.
