@[toc]
>参考文章[《如何用简单易懂的例子解释条件随机场（CRF）模型？它和HMM有什么区别？》](https://www.zhihu.com/question/35866596/answer/236886066?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_content=group3_Answer&utm_campaign=shareopn)
>[《条件随机场CRF之从公式到代码》](https://zhuanlan.zhihu.com/p/178731739?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn)
>[《CRF条件随机场的原理、例子、公式推导和应用》](https://zhuanlan.zhihu.com/p/148813079?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn)

## 一、概率图模型
### 1.1 概览
在统计概率图（probability graph models）中，参考宗成庆老师的书，是这样的体系结构：
![在这里插入图片描述](https://img-blog.csdnimg.cn/c8b569f3f6d04c488431dce83ed52f8c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
在概率图模型中，数据(样本)由图$G=(V,E)$建模表示：
- V：节点v的集合。v∈V表示随机变量$Y_v$。
- E：边e的集合。e∈E表示随机变量之间的概率依赖关系。
- P(Y):由图表示的联合概率分布

有向图和无向图区别在于如何求概率分布P(Y)。
### 1.2 有向图 
有向图模型，这么求联合概率： 
$$P(x_{1}...x_{n})=\prod_{i=0}P(x_{i}|\pi (x_{i}))$$
对于下图求概率有：
![在这里插入图片描述](https://img-blog.csdnimg.cn/7256a9d4eb6e4d7ea9647c6d47746801.png)
$$P(x_{1}...x_{n})=P(x_{1})\cdot P(x_{2}| x_{1})\cdot P(x_{3}| x_{2})\cdot P(x_{4})|P(x_{2})P(x_{5}|x_{3},x_{4})$$

### 1.3 无向图
基本概念：
1. 团：节点子集，子集中任何两个节点均有边相连
2. 最大团：不能再加入节点使其更大的团
3. 因子分解：联合概率分布P(Y)表示为`所有最大团上随机变量函数的乘积的形式`为因子分解。

所以有：联合概率分布为最大团势函数的乘积。
<font color='red'>$$P(Y)=\frac{1}{Z}\prod_{C}\psi _{C}(Y_{C})=\frac{1}{Z}\prod_{C}exp^{-E(Y_{C})}$$
- C：无向图的最大团
- $Y_{C}$:最大团C上的节点(随机变量）
- Z：规范化因子。$Z=\sum_{Y}\prod_{C}\psi _{C}(Y_{C})$,使得输出P(Y)具有概率意义。
- $\psi _{C}(Y_{C})$:<font color='red'>严格正的势函数，通常为指数函数$\psi _{C}(Y_{C})=exp^{-E(Y_{C})}$。

马尔科夫性，保证概率图为概率无向图：
- 成对马尔科夫性：u和v没有边相连，O为其它所有节点，$Y_u和Y_v$互相独立。
- 局部马尔科夫性：对于任意节点v、相连节点集合W和无边相连集合O，给定$Y_W$情况下，$Y_v和Y_O$互相独立：
- 全局马尔科夫性：节点A和B被节点集合C分割，$Y_A和Y_B$互相独立：

总之就是没有边相连的节点概率互相独立。

对于一个无向图，举例如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/f77a15a1b61a4829868472dce880feb5.png)
$$P(Y)=\frac{1}{Z}\prod_{C}\psi( _{X_{1},X_{3},X_{4}})\psi( _{X_{2},X_{3},X_{4}})$$
### 1.4 生成式模型和判别式模型
### 1.4.1生成式模型和判别式模型区别
有监督学习中，训练数据包括输入X和标签Y。所以模型求的是X和Y的概率分布。根据概率论的知识可以知道，对应的概率分布（以概率密度函数指代概率分布）有两种：
- 联合概率分布：$P_{\theta }(X,Y)$，表示数据和标签同时出现的概率，对应于生成式模型。
- 条件概率分布：P_{\theta }(Y|X)，表示给定数据条件下，对应标签的概率，对应于判别式模型。

进一步理解：
- 生成式模型：除了能够根据输入数据 X 来预测对应的标签 Y ,还能根据训练得到的模型产生服从训练数据集分布的数据( X ，Y），相当于生成一组新的数据，所以称之为生成式模型。
- 判别式模型：仅仅根据X由条件概率$P_{\theta }(Y|X)$来预测标签Y。牺牲了生成数据的能力，但是比生成式模型的预测准确率高。
#### 1.4.2 为啥判别式模型预测效果更好
原因如下：由全概率公式和信息熵公式可以得到：
$$P(X,Y)=\int P(Y|X)P(X)dX$$
即计算全概率公式$P(X,Y)$时引入了输入数据的概率分布$P(X)$，而这个并不是我们关心的。我们只关心给定X情况下Y的分布，这就相对削弱了模型的预测能力。
另外从信息熵的角度进行定量分析。
1. X的信息熵定义为：
$$H(X)=-\int P(X)logP(X)dX$$
2. 两个离散随机变量 X  和 Y  的联合熵 (Joint Entropy) 表示两事件同时发生系统的不确定度:
$$H(X,Y)=-\int P(X,Y)logP(X,Y)dXdY$$
3. 条件熵 (Conditional Entropy) H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性：
$$H(Y|X)=-\int P(Y|X)logP(Y|X)dX$$

可以推导出来$H(Y|X)=H(X,Y)-H(X)$.一般H(X)>0（所有离散分布和很多连续分布满足这个条件），可以知道<font color='deeppink'>条件分布的信息熵小于联合分布，即判别模型比生成式模型含有更多的信息，所以同条件下比生成式模型效果更好。</font >
## 二、隐式马尔科夫模型HMM
### 2.1 HMM定义
- 隐马尔可夫模型是关于时序的概率模型
- 描述由一个`隐藏的马尔可夫链`随机生成`不可观测的状态随机序列`(state sequence)，再由各个状态生成一个观测而产生`观测随机序列`(observation sequence )的过程,序列的每一个位置又可以看作是一个时刻。


设Q是所有可能状态的集合，V是所有可能观测的集合：
$$Q=(q_{1},q_{2},...q_{N})和V=(v_{1},v_{2},...v_{M})$$



对于长度为T的状态序列I和观测序列O有：
$$i=(i_{1},i_{2},...i_{T})和O=(o_{1},o_{2},...o_{T})$$
其中:
- 状态转移概率矩阵$A=(a_{ij})_{N\times N}\qquad i,j\epsilon (1,N)$。<font color='red'>$a_{ij}$表示t时刻状态$q_i$转移到t+1时刻$q_j$的概率
- 观测概率（发射概率）矩阵$B=[b_{j}(k)]_{N\times M} \quad j\epsilon (1,N)k\epsilon (1,M)$。<font color='red'>$b_{j}(k)$表示t时刻状态$q_i$生成观测$v_k$的概率。
- 初始状态概率向量$\pi =(\pi _{i})=P(i_{1}=q_{i})\quad i\epsilon (1,N)$。表示初始时刻处于状态$q_i$的概率。
![在这里插入图片描述](https://img-blog.csdnimg.cn/82df5f741379490f83e123408e52d70d.png)
- 隐状态节点$i_t$在A的指导下生成下一个隐状态节点$i_{t+1}$，并且$i_t$在B的指导下生成观测节点$o_t$ , 并且我只能观测到序列O。
- 根据概率图分类，可以看到HMM属于有向图，并且是生成式模型，直接对联合概率分布建模:
![在这里插入图片描述](https://img-blog.csdnimg.cn/a468a47db0c34f2db5239e2c4bbf0d91.png)
>只是我们都去这么来表示HMM是个生成式模型,实际不这么计算。
### 2.2 HMM三要素和两个基本假设
1. HMM由`初始状态概率向量π、状态转移概率矩阵A、 观测概率矩阵B`三元素构成。
所以HMM模型$\lambda$可以写成：$\lambda =(A,B,\pi)$。`三者共同决定了隐藏的马尔可夫链生成不可观测的状态序列`。而状态序列和矩阵B综合产生观测序列。
2. HMM模型基本假设
	- 齐次马尔科夫性假设：隐马尔可夫链<font color='red'>任意时刻t的状态只依赖前一时刻t-1的状态，即$P(i_{t}|i_{i-1})$。
	- 观测独立性假设：<font color='red'>任意时刻的观测只依赖当前时刻的状态，即$P(o_{t}|i_{i})$。

### 2.3 HMM三个基本问题
1. 概率计算：给定模型$\lambda =(A,B,\pi)$和观测序列O，计算观测序列O出现的概率$P(O|\lambda)$。
2. 学习问题：已知观测序列O，用最大似然估计的方法计算模型$\lambda =(A,B,\pi)$的参数。（该模型下观测序列O的概率最大）
3. 预测（解码）问题：已知模型$\lambda =(A,B,\pi)$和观测序列，求最有可能的对应状态序列。

- ==HMM可以用于序列标记，观测序列O为tokens，状态序列I为其对应的标记。此时问题是给定序列O预测对应序列I。==
- 问题2对应模型建立过程，问题3 对应解码过程（crf.decode）
### 2.4 HMM基本解法
#### 2.4.1 极大似然估计（根据I和O求λ）
一般做NLP的序列标注等任务，在训练阶段肯定是有隐状态序列的，即根据观测序列O和状态序列I求模型$\lambda =(A,B,\pi)$的参数，是一个有监督学习。
1. 根据状态序列求状态转移矩阵A：
$$\mathbf{a_{ij}=\frac{A_{ij}}{\sum_{j=1}^{N}A_{ij}}}$$
2. 根据状态序列I和观测序列O求观测概率矩阵B：
$$\mathbf{b_{j}(k)=\frac{B_{jk}}{\sum_{k=1}^{M}B_{jk}}}$$
3. 直接估计π
#### 2.4.2 前向后向算法（没有I）
只有观测序列O，没有状态序列I，无监督过程。计算就是一个就EM的过程。
![](https://img-blog.csdnimg.cn/1d4febff1b6043bda7cc7d466111afe4.png)
#### 2.4.3 序列标注（解码）过程
- 学习完了HMM的分布参数，也就确定了一个HMM模型。序列标注问题也就是“预测过程”(解码过程)。对应了序列建模问题3。
- 学习后已知了 联合概率P(I,O),现在要求出条件概率P(I|O)：
$$I_{max}=\underset{all I}{argmax}\frac{P(I,O)}{P(O)}$$
- <font color='red'>用Viterbi算法解码，在给定的观测序列下找出一条概率最大的隐状态序列。
- Viterbi计算有向无环图的一条最大路径，用DP思想减少重复的计算。如图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/8c629c14f21748149969da86a07b677d.png)
## 三、最大熵马尔科夫MEMM模型
### 3.1 MEMM原理和区别
MEMM是判别式模型，直接对条件概率建模：
![在这里插入图片描述](https://img-blog.csdnimg.cn/9f9ee1ef4db54e94826a3f87bf85770e.png)
MEMM需要注意：
1.  HMM是$o_t$只依赖当前时刻的隐藏状态$i_t$，<font color='red'>HEMM是当前时刻隐状态$i_t$依赖观测节点$o_t$和上一时刻状态$i_{t-1}$。

2. `判别式模型是用函数直接判别，学习边界，MEMM即通过特征函数来界定`。HMM是生成式模型，参数即为各种概率分布元参数，数据量足够可以用最大似然估计。但同样，MEMM也有极大似然估计方法、梯度下降、牛顿迭代发、拟牛顿下降、BFGS、L-BFGS等等
3. 需要注意，之所以图的箭头这么画，是由MEMM的公式决定的，而公式是creator定义出来的。
4. 序列标注解码时，一样用维特比算法求概率最大的隐状态序列。

- HMM中，观测节点$o_t$只依赖当前时刻的隐藏状态$i_t$。
- <font color='red'>更多的实际场景下，观测序列是需要很多的特征来刻画的。比如说，我在做NER时，我的标注$i_t$不仅跟当前状态 $o_t$相关，而且还跟前后标注 $i_{j}(j≠i)$相关，比如字母大小写、词性等等。
- <font color='red'>MEMM模型:允许“定义特征”，直接学习条件概率，即：
![在这里插入图片描述](https://img-blog.csdnimg.cn/87be161d470f4af4ba6e3a6a69fbce8c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
- $Z(o,i{}')$：归一化系数
- $f(o,i)$：特征函数，需要自定义，其个数可任意制定
- λ：特征函数系数，需要训练得到
![在这里插入图片描述](https://img-blog.csdnimg.cn/09d86caafb094b6f8a63771e7e77e4e5.png)
### 3.2 标注偏置
![在这里插入图片描述](https://img-blog.csdnimg.cn/a42fb328efa94ae4b8898d3169acd9db.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
用Viterbi算法解码MEMM，状态1倾向于转换到状态2，同时状态2倾向于保留在状态2。 过程细节：

```python
P(1-> 1-> 1-> 1)= 0.4 x 0.45 x 0.5 = 0.09 ，
P(2->2->2->2)= 0.2 X 0.3 X 0.3 = 0.018，
P(1->2->1->2)= 0.6 X 0.2 X 0.5 = 0.06，
P(1->1->2->2)= 0.4 X 0.55 X 0.3 = 0.066 
```

但是得到的最优的状态转换路径是1->1->1->1，
为什么呢？因为状态2可以转换的状态比状态1要多，从而使转移概率降低,即<font color='red'>MEMM倾向于选择拥有更少转移的状态。原因如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/d8f64347c3df42a58170e7768592096d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
## 四、条件随机场CRF
### 4.1 CRF定义
1. <font color='red'>条件随机场：给定随机变量X条件下，输出随机变量Y的条件概率模型，其中Y构成无向图G=(V,E)表示的马尔科夫随机场。</font >
对任意节点v，条件随机场满足：
$$P(Y_{v}|X,Y_{w},w\neq v)=P(Y_{v}|X,Y_{w},w\sim v)$$
w≠ v表示v之外的所有结点，w~v表示与v有边相连的所有结点。即$P(Y_{v}$之和与v有边连接的结点有关。
2. `线性链条件随机场，最大团是相邻两个结点的集合`。满足马尔科夫性(隐状态只和前后时刻状态有关）：
$$P(Y_{i}|X,Y_{1},Y_{2}...Y_{n})=P(Y_{i}|X,Y_{i+1},Y_{i-1})$$
3. 线性链CRF是判别模型，学习方法是利用训练数据的（正则化）极大似然估计得到条件概率模型P(Y|X)。可用于序列标注问题。此时条件概率P(Y|X)中：
	- Y为输出变量，即标记序列（状态序列）
	- X为输入变量，即需要标注的状态序列。

4. 预测时，对于给定输入序列x，求出条件概率最大的输出序列y。
![在这里插入图片描述](https://img-blog.csdnimg.cn/8720f5b7ee7040339fd7e12f6cfee126.png)
### 4.2 线性链CRF的计算
概率无向图的联合概率分布可以在因子分解下表示为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/984eb14e02934c4da21d22985b7d3255.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
1. <font color='red'>下标i表示我当前所在的节点（token）位置。
2. <font color='red'>下标k表示我这是第几个特征函数</font>，并且每个特征函数都附属一个权重$\lambda_{k}$ 。即每个团里面，我将为$token_i$构造M个特征，每个特征执行一定的限定作用，然后建模时我再为每个特征函数加权求和。 
3. Z(O)是用来归一化的，形成概率值。
4. $P(I|O)$表示了在给定的一条观测序列 O的条件下，我用CRF所求出来的隐状态序列$I=(i_{1},i_{2},...i_{T})$的概率。而至于观测序列 O，它可以是一整个训练语料的所有的观测序列；也可以是在推断阶段的一句sample。比如序列标注进行预测，最终选的是最大概率的那条（by viterbi）。
5. 对于CRF，可以为他定义两款特征函数：转移特征&状态特征。 我们将建模总公式展开：

![在这里插入图片描述](https://img-blog.csdnimg.cn/bc28595886384e76ab860cf1eb9175d9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
 <font color='red'>转移特征针对的是前后token之间的限定。
 
为了简单起见，将转移特征和状态特征及其权值用统一符号表示。条件随机场简化公式如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/a407494d8818498dbde165b83a141b1a.png)
再进一步理解的话，我们需要把特征函数部分抠出来：
![在这里插入图片描述](https://img-blog.csdnimg.cn/69ed4fa41ce64a27b87f8ecd717cf56d.png)
我们为$token_i$打分，满足条件的就有所贡献。最后将所得的分数进行log线性表示，求和后归一化，即可得到概率值。
具体应用求解参考[《如何用简单易懂的例子解释条件随机场（CRF）模型？它和HMM有什么区别？》](https://www.zhihu.com/question/35866596/answer/236886066?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_content=group3_Answer&utm_campaign=shareopn)。
### 4.3 从公式到代码的理解
实际计算时，采用概率的对数形式，即logP(Y)。使用最大似然估计来计算分布的参数，即我们的目标就是最大化ogP(Y)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/12330b71b141418f9fb7cc3ec9039943.png)
即$-logP(Y)=logZ(x)-score$。
对应到代码中，forward_score 就是$logZ(x)$，gold_score就是特征函数部分的score。
```python
def neg_log_likelihood(self, sentence, tags):
    feats = self._get_lstm_features(sentence)
    forward_score = self._forward_alg(feats)
    gold_score = self._score_sentence(feats, tags)
    return forward_score - gold_score
```
- 因为模型建立的初衷就是要考虑到$i_{k-1}$对$i_{k}$的影响和X对观测序列的影响.所以我们将图分解成若干个$(i_{k-1},i_{k},X)$。
- 其中$i_{k}$表示观测变量的状态值，比如在BIO标注中状态取值范围是{B,I,O,START,STOP}，则k最大取5，$i_{k}$有5个状态值可取。
![在这里插入图片描述](https://img-blog.csdnimg.cn/6081457d466243bbbbf3e97d99cd0e9b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
只关注其中的某一个团$C_i$,特征函数部分的gold_score表示给定序列X下，表现出的$(i_{k-1}，i_{k})$的费归一化概率，与两个东西有关：
1. 给定序列X下出现$i_{k}$的概率，以$h(i_{k},X)$表示。这个概率使用lstm、cnn建模X对$i_{k})$映射就可以得到，对应结点上的状态特征。
2. 给定序列X下由$i_{k-1}$转移到$i_{k}$的概率，由$g(i_{k-1},i_{k};X)$表示，对应边上的转移特征。在CRF中，观测变量只受临近节点的影响。
3. 考虑到深度学习模型已经能比较充分捕捉各个$i_{k}$与X 的联系，所以假设 $i_{k-1}$ 转移到$i_{k}$的概率与X无关，所以有：$g(i_{k-1},i_{k};X)=g(i_{k-1},i_{k})$

考虑以上几点，可以得到：
$$gold-score=\sum_{c}\sum_{k}\lambda _{k}f_{k}(c,y,x)=\sum_{c}\sum_{k}(g(i_{k-1},i_{k})+h(i_{k},X))$$

剩下计算过程参考：[《条件随机场CRF之从公式到代码》](https://zhuanlan.zhihu.com/p/178731739?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b2612debc2c94e7fbdcaf918a69bc0a7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)


## 五、 HMM vs. MEMM vs. CRF
### 5.1 HMM vs MEMM
HMM模型中存在两个假设：一是输出观察值之间严格独立，二是状态的转移过程中`当前状态只与前一状态有关`。但实际上序列标注问题不仅和单个词相关，而且和观察序列的长度，单词的上下文，等等相关。`MEMM解决了HMM输出独立性假设的问题`。因为HMM只限定在了观测与状态之间的依赖，而`MEMM引入自定义特征函数，不仅可以表达观测之间的依赖，还可表示当前观测与前后多个状态之间的复杂依赖`。
### 5.2 MEMM vs CRF
CRF不仅解决了HMM输出独立性假设的问题，还解决了MEMM的标注偏置问题，`MEMM容易陷入局部最优是因为只在局部做归一化，而CRF统计了全局概率，在做归一化时考虑了数据在全局的分布`，而不是仅仅在局部归一化，这样就解决了MEMM中的标记偏置的问题。使得序列标注的解码变得最优解。HMM、MEMM属于有向图，所以考虑了x与y的影响，但没将x当做整体考虑进去（这点问题应该只有HMM）。CRF属于无向图，没有这种依赖性，克服此问题。











