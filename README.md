# 🔥Representation Matching Information Bottleneck
The official implementation code for [RMIB: Representation Matching Information Bottleneck for Matching Text Representations](https://openreview.net/pdf?id=hsHIxrnrMx) (ICML2024). This repo is build based on [RE2](https://github.com/alibaba-edu/simple-effective-text-matching-pytorch). To ensure the reproducibility of the experiment, we set the seed of all experiments to 32. We also upload training logs for BERT and SBERT to the path `logs/`.

## 🚗Installation and Running

* Experimental env
  * GPU: GeForce RTX 2080Ti
  * CUDA Version: 11.0
  * Python Version: 3.7.13 

* Running script
  * Git clone repo </br>
    `git clone https://github.com/chenxingphh/rmib`
    
  * Install related packages </br>
   `pip install -r requirement.txt`

  * Download [GloVe word vectors](https://nlp.stanford.edu/projects/glove/) (glove.840B.300d) to `resources/`
  
  * Runing script (Using different configs to run different models and datasets) </br>
    `python train.py configs/sbert_sick.json5`

## 🚀 Brief Introduction to RMIB

### Background

Asymmetrical text matching refers to two input texts from different domains. For example, the question and the candidate answers can be viewed as being sampled from two different distributions. For addressing the challenge of matching texts across domains, we introduce the Representation Matching Information Bottleneck (RMIB) framework. 

### Methodology

#### Text Domain Alignment Based on Prior Distribution 

Recent studies have shown that domain matching of text representation will help improve the generalization ability of text matching.  We narrow the distribution between text representations by explicitly aligning text representations with a prior distribution in text matching. 

$$
\theta ^{*} =\arg \min_{\theta}E_{p(x_1,x_2,y)}[-\log p_{\theta}(y\mid x_1,x_2)] \ s.t.\ \begin{cases}
KL(p_{\theta}(z_1\mid x_1,x_2)\parallel p(\mathcal Z))=0  \\
KL(p_{\theta}(z_2\mid x_1,x_2)\parallel p(\mathcal Z))=0
\end{cases}
$$

We prove that domain matching in text matching is equivalent to optimizing the information bottleneck in text matching, which indicates that domain alignment of input texts can make the learned text representation forget the input redundant information as much as possible. 

#### RMIB

Since the interaction between text representations plays an important role in asymmetrical domains text matching, IB does not restrict the interaction between text representations. Therefore, we propose the adequacy of interaction and the incompleteness of a single text representation on the basis of IB and obtain the representation matching information bottleneck (RMIB).

**Sufficient:** The representations $Z_1$ and $Z_2$ should contain as much information related to the target label as possible. This constraint is consistent with the Sufficient of IB.
$$I(Y;Z_1,Z_2)=I(X;Y)$$

**Interaction:** The interaction between text representations should be sufficient, which means there should be enough mutual information between the two text representations.
$$\max I(Z_1; Z_2)$$

**Inadequacy:** The final correct result cannot be obtained only by using a single text representation in text matching.
$$\min I(Y; Z_1) + I(Y; Z_2)$$

The optimization objective of RMIB is:

$$
Z_{1}^{\*}, Z_{2}^{\*} =  \arg \min_{Z_1,Z_2}  I(X_1,X_2;Z_1)+ I(X_1,X_2;Z_2) \ s.t. \ \max I(Z_1;Z_2)+I(Z_1,Z_2;Y)-(I(Y;Z_1)+I(Y;Z_2)) 
$$

We then prove the optimization objective of RMIB can also be expressed as:

$$
Z_{1}^{\*}, Z_{2}^{\*} =  \arg \min_{Z_1,Z_2}  I(X_1,X_2;Z_1)+ I(X_1,X_2;Z_2) \ s.t. \ \max I(Z_1;Z_2\mid Y)
$$

***

⭐If you are interest in RMIB, please consider to cite this paper:
```
@inproceedings{
    pan2024rmib,
    title={RMIB: Representation Matching Information Bottleneck for Matching Text Representations},
    author={Haihui Pan, Zhifang Liao, Wenrui Xie, Kun Han},
    booktitle={International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/pdf?id=hsHIxrnrMx}
}
```
