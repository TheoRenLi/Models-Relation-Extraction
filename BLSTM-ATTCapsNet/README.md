## 基于注意力胶囊网络的模型关系抽取实验

- 该模型来自于论文：[Multi-labeled Relation Extraction with Attentive Capsule Network](https://arxiv.org/pdf/1811.04354.pdf "链接")，作者：Xinsong Zhang, Pengshuai Li, Weijia jia and Hai Zhao.
- 本实验只使用了词向量、相对实体的位置向量和词性嵌入向量作为网络的输入；
- 模型：见图: ![./pics/model.png]()
- 实验设置：数据集SemEval-2010 Task 8；词向量采用维基百科最新语料训练，维度300d，skip-gram模型；实验结果F1值大小为84.35%（原文F1值为84.5%）
- 数据预处理：预处理文件在data_processing下
  - format2json.py（先执行，生成json文件）
  - gen_data.py （后执行，需要json文件）
  - 备注：自己更改路径，以及添加其他所需代码
- 训练损失见图：![./pics/loss_plot.png]()
