# UGrid：一种高效且严谨的线性偏微分方程神经多重网格求解器

本仓库是ICML'2024论文的官方实现

Xi Han, Fei Hou, Hong Qin, 
"UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs",
发表于《第41届国际机器学习会议论文集》,
pp. 17354--17373, 2024年7月。

论文可通过以下链接获取：
- [arXiv](https://arxiv.org/abs/2408.04846)
- [PMLR](https://proceedings.mlr.press/v235/han24a.html)
- [openreview](https://openreview.net/forum?id=vFATIZXlCm)

## 数据生成

要生成数据集，请运行以下命令：

```bash
bash ./script/generate.sh
```

请修改 `generate.sh` 以生成所需大小的 `train`、`evaluate` 和 `test` 数据集。

## 训练

要训练论文中的模型，请运行以下命令：

```bash
bash ./script/train.sh
```

## 评估和测试

要复现UGrid的测试结果，请运行以下命令：

```bash
bash ./script/test.sh
```

要与 [AMGCL](https://github.com/ddemidov/amgcl) 和 [NVIDIA AmgX](https://developer.nvidia.com/amgx) 进行比较，
请首先编译AMGCL和AmgX的Python绑定（参见 `./comparasion/cpmg/`），
然后运行以下命令：

```bash
bash ./script/compare.sh
```

要与 [(Hsieh et al., 2019)](https://openreview.net/forum?id=rklaWn0qK7) 进行比较，
请参考[他们的官方仓库](https://github.com/ermongroup/Neural-PDE-Solver)。

## 预训练模型

预训练模型包含在 `var/checkpoint/22/` 目录中。

## 如何引用

```
@inproceedings{han24-icml-ugrid,
  author={Han, Xi and Hou, Fei and Qin, Hong},
  title={{UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs}},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  volume={235},
  number={},
  pages={17354--17373},
  month={July},
  year={2024},
  url={https://arxiv.org/abs/2408.04846}
}
```
