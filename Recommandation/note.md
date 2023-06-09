# 推荐系统概述

## 1. 推荐系统的基本概念

* 长尾：长尾理论是指，大多数的产品销售量集中在少数的产品上，而大多数的产品销售量很少。
* 推荐方式
  * 社会化推荐：例如朋友咨询
  * 基于内容推荐：：例如分析用户曾看过的电影，然后推荐相似的电影
  * 基于协同过滤推荐：例如排行榜，用户之间的相似度

## 2. 推荐系统的评测

### 2.1 推荐系统的实验方法

* 离线实验
  * 优点：不需要实际系统控制权；不需要用户参与；速度快，可以测试大量算法
  * 缺点：试验指标与商业指标不一致；试验结果不一定能够泛化到实际系统
* 用户调查
  * 优点：可以直接反映用户的真实反馈；相比在线实验风险更低
  * 缺点：代价较大；用户在测试环境下的行为可能与实际不符
* 在线实验
  * 优点：可以反映实际在线系统的效果；
  * 缺点：周期较长

### 2.2 推荐系统的评测指标

* 用户满意度：调查问卷
* 预测准确度：
  * 评分预测：u 表示用户，i 表示物品，$r_{ui}$ 表示实际评分，$\hat{r_{ui}}$ 表示预测评分，T 表示测试集
    * 均方更误差（RMSE）：$$\sqrt{\frac{1}{|T|}\sum_{u, i\in T}(r_{ui}-\hat{r_{ui}})^2}$$
    * 平方绝对值误差（MAE）：$$\frac{1}{|T|}\sum_{u, i\in T}|r_{ui}-\hat{r_{ui}}|$$
    * RMSE 对于异常值更加敏感，对预测不准的惩罚更大
  * topN 推荐: U 表示用户集合，$R(u)$ 表示用户 u 的推荐结果，$T(u)$ 表示用户 u 的测试集
    * 召回率：$$\frac{\sum_{u\in U} | R(u) \cap T(u)|}{\sum_{u\in U}|T(u)|}$$
    * 准确率：$$\frac{\sum_{u\in U} | R(u) \cap T(u)|}{\sum_{u\in U}|R(u)|}$$
* 覆盖率：$\frac{\sum_{i\in I}|R(i)|}{|I|}$，其中 I 表示物品集合，$R(i)$ 表示物品 i 被推荐的次数
  * 信息熵：p(i) 是物品 i 的流行度除以所有物品的流行度之和
    $$H=-\sum_{i\in I}p(i)\log_2p(i)$$
  * 基尼系数：$i_j$ 是物品按照流行度从大到小排列的第 j 个物品
    $$G=\frac{1}{n-1}\sum_{j=1}^n(2j-n-1)p(i_j)$$
    > ![基尼系数](img/%E5%9F%BA%E5%B0%BC%E7%B3%BB%E6%95%B0.png)
    > 黑色曲线表示最不热门的 x% 物品的总流行度占系统的比例 y%
    > 基尼系数形象的定义为 $S_A /  (S_A + S_B)$
    > 系统流行度分配越不均匀，基尼系数越大
    * 马太效应：马太效应是指，热门的物品会更加热门，冷门的物品会更加冷门。若G1 是用户初始行为的基尼系数，G2 是基于用户初始行为计算出的基尼系数，若 G2 > G1 表示推荐算法具有马太效应。
* 多样性：s(i, j) 定义了物品i和物品j的相似度，$R(u)$ 表示用户 u 的推荐结果
  $$Diversity=1-\frac{\sum_{i, j\in R(u)}s(i, j)}{|R(u)|(|R(u)|-1)}$$
* 新颖性：用户没听说过
* 惊喜度：和用户历史上喜欢的物品不相似，但用户觉得满意
* 信任度
  * 增加推荐系统透明度：例如解释推荐理由、利用好友信息做推荐
* 实时性
* 健壮性：抗击作弊能力

### 2.3 推荐系统的评测维度

* 用户维度：人口统计学信息、活跃度以及是否新用户等
* 物品信息：物品的流行度、物品的类别、物品的属性等
* 时间维度：季节、白天还是晚上、工作日还是周末等


# 利用用户行为数据

## 用户行为分析

### 长尾分布

每个物品出现的频率和它在热门排行榜中的排名的常数次幂成反比

$$f(x) = \alpha x^k$$

## 实验设计和算法评测

### 数据集

无上下文信息的隐性反馈数据集

### 实验设计

将用户行为数据按照均匀分布随机分成 M 份，其中一份作为测试集，其余 M-1 份作为训练集。

为了保证评测指标并不是过拟合的结果，需要进行 M 次实验，并且每次都使用不同的测试集。

### 评测指标

* 召回率：$$\frac{\sum_{u\in U} | R(u) \cap T(u)|}{\sum_{u\in U}|T(u)|}$$
* 准确率：$$\frac{\sum_{u\in U} | R(u) \cap T(u)|}{\sum_{u\in U}|R(u)|}$$
* 覆盖率：$$\frac{\sum_{i\in I}|R(i)|}{|I|}$$
* 新颖度：$$\frac{\sum log(1 + p_i)}{n}$$，其中 $p_i$ 是物品 i 的流行度，n 是推荐列表的长度

## 基于领域的算法

### 基于用户的协同过滤算法

* 找到和目标用户兴趣相似的用户集合
  * Jaccard 公式：$$w_{uv} = \frac{|A \cap B|}{|A \cup B|}$$
  * 余弦相似度：$$w_{uv} = \frac{|A \cap B|}{\sqrt{|A| |B|}}$$
  * 通过 $\frac{1}{\log(1+|N(i)|)}$ 降低热门物品的影响：$$w_{uv} = \frac{\sum_{i \in N(u) \cap N(v)}\frac{1}{\log(1 + |N(i)|)}}{\sqrt{|N(u)| |N(v)|}}$$
* 找到这个集合中的用户喜欢的，且目标用户没有听说过物品推荐给目标用户
  * 如下公式度量了用户 u 对物品 i 感兴趣程度：$$p_{ui} = \sum_{v\in S(u, K) \cap N(i)}w_{uv}r_{vi}$$ ，其中 $S(u, K)$ 是和用户 u 兴趣相似的前 K 个用户，$N(i)$ 是和物品 i 共现过的用户集合，$r_{vi}$ 是用户 v 对物品 i 的兴趣度

### 基于物品的协同过滤算法

* 计算物品间的相似度
  * 余弦相似度：$$w_{ij} = \frac{|N(i) \cap N(j)|}{\sqrt{|N(i)| |N(j)|}}$$
  * 通过 $\frac{1}{\log(1+|N(u)|)}$ 降低活跃用户的影响：$$w_{ij} = \frac{\sum_{u \in N(i) \cap N(j)}\frac{1}{\log(1 + |N(u)|)}}{\sqrt{|N(i)| |N(j)|}}$$
  * 一般来说热门的类其类内物品相似度一般比较大，若用户同时喜欢热门的类和不热门的类的物品，推荐系统更有可能推荐热门类的物品。通过相似度的归一化可以提高推荐的覆盖率和多样性：$$w_{ij}' = \frac{w_{ij}}{\max_j{w_{ij}}}$$
* 根据物品的相似度和用户的历史行为给用户生成推荐列表
  * 如下公式度量了用户 u 对物品 i 感兴趣程度：$$p_{ui} = \sum_{j\in N(u) \cap S(i, K)}w_{ij}r_{uj}$$，其中 $N(u)$ 是用户 u 历史上交互过的物品集合，$S(i, K)$ 是和物品 i 最相似的前 K 个物品，$r_{uj}$ 是用户 u 对物品 j 的兴趣度

### UserCF 和 ItemCF 的比较

|          | UserCF                                                         | ItemCF                                                         |
| :------- | :------------------------------------------------------------- | :------------------------------------------------------------- |
| 性能     | 使用于用户较少的场合，如果用户很多，计算用户相似度矩阵代价很大 | 使用于物品较少的场合，如果物品很多，计算物品相似度矩阵代价很大 |
| 领域     | 时效性较强，用户个性化兴趣不太明显                             | 长尾物品丰富，用户个性化需求强烈的领域                         |
| 实时性   | 用户有新行为，不一定造成推荐结构立即变化                       | 用户有新行为，一定为导致结果实时变化                           |
| 推荐理由 | 很难提供用户信服的推荐理由                                     | 利用用户的历史行为给用户做推荐理由                             |

## 隐语义模型

* 如下公式计算用户 u 对物品 i 的兴趣：$$Preference(u, i) = r_ui = p_u^T q_i = \sum_{k=1}^K p_{uk}q_{ki}$$，其中 $p_{uk}$ 度量了用户 u 和第 k 个隐类的关系，$q_{ki}$ 度量了物品 i 和第 k 个隐类的关系
* 从用户没有过行为的物品中采样出一些物品作为负样本
* 损失函数：$$C = \sum_{u,i \in K} (r_{ui} - \sum_{k=1}^{K}p_{u,k}^T q_{i, k})^2 + \lambda ||p_u||^2 + \lambda ||q_i||^2$$，其中 $r_{ui}$ 是用户 u 对物品 i 的兴趣，$K$ 是用户 u 和物品 i 的交互集合，$\lambda$ 是正则化参数，$lambda ||p_u||^2$ + $lambda ||q_i||^2$ 是防止过拟合的正则化项
* 求出损失函数偏导数：$$\frac{\partial C}{\partial p_{uk}} = -2 q_{ik} + 2 \lambda p_{uk}$$，$$\frac{\partial C}{\partial q_{ik}} = -2 p_{uk} + 2 \lambda q_{ik}$$
* 使用随机梯度下降法优化损失函数，给定学习速率 $\alpha$，迭代公式如下：$$p_{uk}^{(t+1)} = p_{uk}^{(t)} - \alpha \frac{\partial C}{\partial p_{uk}} = p_{uk}^{(t)} +  \alpha (q_{ik} - \lambda p_{uk}^{(t)})$$，$$q_{ik}^{(t+1)} = q_{ik}^{(t)} - \alpha \frac{\partial C}{\partial q_{ik}} = q_{ik}^{(t)} +  \alpha (p_{uk} - \lambda q_{ik}^{(t)})$$

### LFM 和基于领域的方法的比较

* 空间复杂度：假设有 M 个用户和 N 个物品，F个隐类
  * 用户相关表：$O(M*M)$
  * 物品相关表：$O(N*N)$
  * LFM：$O(F*(M+N))$，在 M 和 N 都很大的情况下，LFM 的空间复杂度要远远小于基于领域的方法
* 时间复杂度：假设有 M 个用户和 N 个物品和 K 条用户对物品的行为记录，F 个隐类且迭代 S 次
  * UserCF：$O(N*(K/N)^2)$
  * ItemCF: $O(M*(K/M)^2)$
  * LFM: $O(K*F*S)$，一般情况由于 LFM 需要多次迭代，所以时间复杂度要略大于基于领域的方法
* 在线实时推荐：LFM 生成用户推荐表太慢，所以不适合在线实时推荐，当用户由了新行为，推荐列表不会发生变化

## 基于图的模型

* 用二分图表示用户行为数据：G(V, E)，其中 V 是用户和物品的集合，E 是用户对物品的行为集合
* 相关性高的节点一般由如下特征：
  * 两个顶点之间由很多路径相连
  * 连接两个顶点之间的路径长度较短
  * 连接两个顶点之间的路径不会经过出度较大的顶点
* 基于随机游走的 PersonalRank 算法
  * $$PR(v) = \begin{cases} \alpha \sum_{v' \in in(v)} \frac{PR(v')}{out(v')} & v \neq v_u \\ (1-\alpha) + \alpha \sum_{v' \in in(v)} \frac{PR(v')}{out(v')} & v = v_u \end{cases}$$，其中 $v_u$ 是用户 u，$in(v)$ 是顶点 v 的入度集合，$out(v)$ 是顶点 v 的出度集合，$\alpha$ 是随机游走的概率
  * 在多次迭代后，每个顶点的访问概率收敛到一个稳定值，这个值就是该顶点的 PageRank 值
* 可将 PersonalRank 转换为矩阵形式：$$M(v, v') = \frac{1}{out(v)}$$，其中 $M(v, v')$ 表示从顶点 v 到顶点 v' 的转移概率，$out(v)$ 是顶点 v 的出度
  * 迭代公式可转化为：$$PR(v) = \alpha M^T PR + (1-\alpha)PR_0$$
  * 解得：$$PR = (I - \alpha M^T)^{-1} (1-\alpha)PR_0$$

# 冷启动问题

* 用户冷启动：用户刚注册，没有任何行为
* 物品冷启动：物品刚上架，没有任何行为
* 系统冷启动：系统刚上线

## 解决方案

* 提供非个性化的推荐：热门推荐、新品推荐、类目推荐、搜索推荐
* 利用用户注册时提供的年龄、性别等数据做粗粒度的个性化推荐
* 利用用户的社交网络账号登录，获取用户的好友列表，然后推荐好友喜欢的物品
* 要求用户在登录时对一些物品进行反馈，然后根据反馈的物品进行推荐
* 对于新加入的物品，可以利用内容信息，将其与已有的物品进行相似度计算，然后推给用户