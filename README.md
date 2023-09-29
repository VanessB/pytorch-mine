# PyTorch-MINE
Mutual information neural estimation implemented in PyTorch.

[Russian/Русский](./README_ru.md)

## Description
This library provides you with PyTorch-based classes and functions for mutual information neural estimation.

This method has been proposed by `Belghazi et al` in
```
@InProceedings{pmlr-v80-belghazi18a,
  title = 	 {Mutual Information Neural Estimation},
  author =       {Belghazi, Mohamed Ishmael and Baratin, Aristide and Rajeshwar, Sai and Ozair, Sherjil and Bengio, Yoshua and Courville, Aaron and Hjelm, Devon},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {531--540},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--15 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v80/belghazi18a/belghazi18a.pdf},
  url = 	 {https://proceedings.mlr.press/v80/belghazi18a.html},
  abstract = 	 {We argue that the estimation of mutual information between high dimensional continuous random variables can be achieved by gradient descent over neural networks. We present a Mutual Information Neural Estimator (MINE) that is linearly scalable in dimensionality as well as in sample size, trainable through back-prop, and strongly consistent. We present a handful of applications on which MINE can be used to minimize or maximize mutual information. We apply MINE to improve adversarially trained generative models. We also use MINE to implement the Information Bottleneck, applying it to supervised classification; our results demonstrate substantial improvement in flexibility and performance in these settings.}
}
```

## Documentation
In development.

## Getting started
### Examples
Examples of the application can be found in `source/examples`

## Planned features
### Documentation
- [ ] Documentation file.
- [ ] Wiki.

### Library features
- [x] MINE losses.