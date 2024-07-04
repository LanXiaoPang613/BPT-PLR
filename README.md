# BPT-PLR: A Balanced Partitioning and Training Framework with Pseudo-Label Relaxed Contrastive Loss for Noisy Label Learning

<h5 align="center">

*Qian Zhang, Ge Jin, Yi Zhu, Hongjian Wei, Qiu Chen*

[[Entropy]](https://doi.org/xxxx)
[[License: MIT License]](https://github.com/LanXiaoPang613/BPT-PLR/blob/main/LICENSE)

</h5>

The PyTorch implementation code of the paper, [BPT-PLR: A Balanced Partitioning and Training Framework with Pseudo-Label Relaxed Contrastive Loss for Noisy Label Learning](https://doi.org/xxxx).

**Abstract**
While collecting training data, even with manual verification through experts from crowdsourcing platforms, eliminating incorrect annotations (noisy labels) completely is difficult and expensive. In dealing with datasets that contain noisy labels, over-parameterized deep neural networks (DNN) tend to overfit, leading to poor generalization and classification performance. As a result, noisy label learning (NLL) has received significant attention in recent years. Existing research shows that although DNN eventually fits all training data, they first prioritize fitting clean samples, then gradually overfit to noisy samples. Mainstream methods utilize this characteristic to divide training data but face two issues: class imbalance in the segmented data subsets, and the optimi-zation conflict between unsupervised contrastive representation learning and supervised learning. To address these issues, we propose a Balanced Partitioning and Training framework with Pseudo-Label Relaxed contrastive loss called **BPT-PLR**, which includes two crucial processes: a balanced partitioning process with a two-dimensional Gaussian mixture model (GMM) and a semi-supervised oversampling training process with a pseudo-label relaxed contrastive loss (PLR). The former utilizes both semantic feature information and model prediction results to identify noisy labels, introducing a balancing strategy to maintain class balance in the divided subsets as much as possible. The latter adopts the latest PLR to replace unsupervised contrastive loss, re-ducing optimization conflicts between semi-supervised and unsupervised contrastive losses to improve performance. We validate the effectiveness of BPT-PLR on four benchmark datasets in the NLL field: CIFAR-10/100, Animal-10N, and Clothing1M. Extensive experiments comparing with state-of-the-art methods demonstrate that BPT-PLR can achieve optimal or near-optimal perfor-mance. Source code is released at: https://github.com/LanXiaoPang613/BPT-PLR.

![BPT-PLR Framework](./framework.tif)

[//]: # (<img src="./framework.tif" alt="BPT-PLR Framework" style="margin-left: 10px; margin-right: 50px;"/>)

## Installation

```shell
# Please install PyTorch using the official installation instructions (https://pytorch.org/get-started/locally/).
pip install -r requirements.txt
```

## Training

To train on the CIFAR dataset(https://www.cs.toronto.edu/~kriz/cifar.html), run the following command:

```shell
python Train_cifar_bpt_plr.py --r 0.4 --noise_mode 'asym' --lambda_u 30 --data_path './data/cifar10/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
python Train_cifar_bpt_plr.py --r 0.5 --noise_mode 'sym' --lambda_u 30 --data_path './data/cifar10/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
```

To train on the Animal-10N dataset(https://dm.kaist.ac.kr/datasets/animal-10n/), run the following command:

```shell
python Train_animal_bpt_plr.py --num_epochs 200 --lambda_u 0 --data_path './data/Animal-10N' --dataset 'animal10N' --num_class 10
```


## Citation

If you have any questions, do not hesitate to contact zhangqian@jsou.edu.cn

Also, if you find our work useful please consider citing our work:

```bibtex
xxxx
```

## Acknowledgement

* [DivideMix](https://github.com/LiJunnan1992/DivideMix): The algorithm that our framework is based on.
* [UNICON](https://github.com/nazmul-karim170/UNICON-Noisy-Label): Inspiration for the balanced partitioning process.
* [PLReMix](https://github.com/lxysl/PLReMix): Inspiration for the PLR loss.
* [LongReMix](https://github.com/filipe-research/LongReMix): Inspiration for the oversampling strategy.
