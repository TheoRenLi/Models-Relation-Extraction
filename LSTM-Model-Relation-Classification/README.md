## LSTM模型关系抽取实验

- 该模型来自于论文：[Bidirectional Long Short-Term Memory Networks for relation classification](https://www.aclweb.org/anthology/Y15-1009.pdf "链接")，作者：Shu Zhang, Dequan Zheng, Xinchen Hu and Ming Yang. 

- 本实验未构造论文中提到的“relative dependency features”和“Dep features”，只使用了词向量、相对实体的位置向量和词性嵌入向量；

- 模型：LSTM

- 两个特征向量：

  - 词级特征：其中x\_{e1}是来自词嵌入层的实体e1的向量，F\_{e1}是来自LSTM编码之后的实体e1位置的向量
    $$
    [x_{e1}, F_{e1}, x_{e2}, {F_e2}]
    $$

  - 句子级特征：其中m1和m2见图: ![sen-level.png]()
    $$
    m = concatenate[m1:m2]
    $$

- 词向量采用维基百科最新语料训练，维度300d，skip-gram模型；数据集SemEval-2010 Task 8；实验结果F1值大小为84.3%（原文F1值：84.3%）
- 下面是训练损失: ![loss_plot.png]()
