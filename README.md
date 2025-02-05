# DENSE
🍀The code of DENSE: Dynamic Ensemble Learning for Continual Test-Time Adaptation🍀

![image](DENSE/pic/Frame%20work.png)

Abstract: Continual Test-Time Adaptation (CTTA) aims to adapt a source domain model to handle target domain data with varying distributions. While existing studies have primarily focused on employing a single model to capture universal information across domains, such approaches are insufficient to address the challenges posed by dynamic domain shifts. In this study, we propose a novel dynamic ensemble framework to overcome this limitation. Our study involves learning a set of models coupled with a gating network, enabling the generation of adapted networks with distinct weight configurations tailored to different distributions. This strategy mitigates the limitations inherent in single-model learning and facilitates domain-specific personalized adaptation. Furthermore, we incorporate high-confidence pseudo-labels for model updates to ensure the reliability of the adaptation process. To validate the effectiveness of our method, we conducted comprehensive comparisons with state-of-the-art CTTA approaches, demonstrating superior performance across three benchmark corruption datasets and one artistic style transformation dataset. Our proposed method is publicly available at https://anonymous.4open.science/r/DENSE-BFF3.

Index Terms: test-time adaptation, ensemble learning, continual test-time adaptation

On the following tasks 🌅
+ CIFAR10 -> CIFAR10-C (Standard/Gradual)
+ CIFAR100 -> CIFAR100-C (Standard)
+ ImageNet -> ImageNet-C (Standard)
+ ImageNet -> ImageNet-R (Standard)

Compare this with the following methods 🌈
+ [CoTTA](https://arxiv.org/abs/2203.13591)
+ [ETA](https://arxiv.org/abs/2204.02610)
+ [CPL](https://arxiv.org/abs/2207.09640)
+ [RMT](https://arxiv.org/abs/2211.13081)
+ [SAR](https://arxiv.org/abs/2302.12400)
+ [Continual-MAE](https://arxiv.org/abs/2312.12480)

## Install ##
```git clone https://github.com/Cadezzyr/DENSE.git```

## Test on ImageNet -> ImageNet-C tasks (Standard) ##
```
cd imagenet
python test_imagenet_net.py
```

