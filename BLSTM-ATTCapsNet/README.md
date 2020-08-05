## 基于注意力胶囊网络的模型关系抽取实验

- 该模型来自于论文：[Multi-labeled Relation Extraction with Attentive Capsule Network](https://arxiv.org/pdf/1811.04354.pdf "链接")，作者：Xinsong Zhang, Pengshuai Li, Weijia jia and Hai Zhao.

- 本实验只使用了词向量、相对实体的位置向量和词性嵌入向量作为网络的输入；

- 模型：见图: ![./pics/model.png]()

- 实验设置：数据集SemEval-2010 Task 8；词向量采用维基百科最新语料训练，维度300d，skip-gram模型；实验结果F1值大小为84.35%（原文F1值为84.5%）

- 训练损失见图：![./pics/loss_plot.png]()
