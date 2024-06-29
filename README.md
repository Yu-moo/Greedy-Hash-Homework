# Greedy-Hash-Homework
论文复现，机器学习大作业，原论文GreedyHash(NIPS2018)
paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)
code [GreedyHash](https://github.com/ssppp/GreedyHash)

不包含数据集图片文件，
除cifar外，训练集与测试集的选择与DPCHash_Baselines中txt文件一致（dataloader中shuffle=True读取顺序不一定一致）
cifar在DPCHash_Baselines中没有提供具体训练集和测试集的选择，选取与论文中相同的设置。

CIFAR:https://www.cs.toronto.edu/~kriz/cifar.html
MSCOCO:https://cocodataset.org/
NUS-WIDES: https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q
flickr25k: https://www.kaggle.com/datasets/e593768f204b802f95db5af3f7258e64ad2fe696d2e6d09258eb03509292ece0?resource=download
