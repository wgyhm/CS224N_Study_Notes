### Lesson1

[python学习](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb#scrollTo=M-E4MUeVL9iC)
[Matplotlib](https://matplotlib.org/stable/gallery/index.html)
[奇异值分解SVD](https://zhuanlan.zhihu.com/p/29846048)

注意到如果连不上imdb_dataset,可以设置系统变量，但无论是 HTTP_PROXY 还是 HTTPS_PROXY = 'http://xxx:xxx'，不是 https

### Lesson2

做一点简单的辨析。

* Task1 中的 co-occurence embedding + SVD，虽然直观，但是矩阵很大，处理大语料不够高效，无法自然使用线性模型
* word2vec 的概率函数（Skip-gram）：通过概率目标函数让模型自动调整向量，使其保持“语义相似词在向量空间也相近”。高效，可在大语料上训练；自然得到低维稠密向量

* 1.3 求 loss 为什么要平方？
  * 首先，算损失函数的本质上是在让模型预测和“真实统计量”尽量接近
  * 平方的意义是防止正负相互抵消。
  * 之后取对数，防止某些 $X_{i,j}$ 差距过大，减小其差距。
* 全局交叉熵损失要做归一化，所以改成了代价更小的最小二乘目标。
* 算出损失以后，模型会根据根据 J 对参数求梯度，再修改参数。
* Intrinsic evaluation 是在中间子任务上快速评估 embedding 质量；Extrinsic evaluation 是在真实下游任务上评估整体效果，通常更慢、更贵。
* 3.1 把很多 NLP 下游任务写成分类问题，并先从线性分类器出发。课上讲的就是归类为 location/no-location
* 3.2 的 retraining words，就是在下游任务上继续微调预训练词向量；数据大时可能有帮助，数据小时可能有害。
* 3.3 里 y 是 one-hot 标签，表示在不在那个类里，所以每个分量是 0/1；C 表示总类别数；θ 是所有可训练参数，不是超参数。
* 3.4 不是只讨论 window 大小，而是在说要把“中心词 + 上下文词向量”组成。把这些词向量拼成的一个大输入向量。后面反向传播时，对这个大向量求梯度，再把梯度分发回每个位置对应的词向量。
3.5 才正式说明：线性分类器不够时，需要非线性分类器，比如神经网络。
