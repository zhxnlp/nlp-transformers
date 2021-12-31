@[toc]
## 一、BP神经网络
### 1.1 为何要引出BP神经网络
1. 逻辑回归对于如今越来越复杂的任务效果越来越差，主要是难以处理线性不可分的数据，LR处理线性不可分，一般是特征变换和特征组合，将低维空间线性不可分的数据在高维空间中线性可分
2. 改良方式有几种，本质上都是对原始输入特征做文章。但都是针对特定场景设计。如果实际场景中特征组合在设计之外，模型无能为力
	- 人工组合高维特征，将特征升维至高维空间。但是会耗费较多人力，而且需要对业务理解很深
	- 自动交叉二阶特征，例如FM模型。缺点是只能进行二阶交叉
	- SVM+核方法：可以将特征投影到高维空间。缺点是核函数种类有限，升维具有局限性，运算量巨大。

3.构建BP神经网络（也叫MLP多层感知器），<font color='red'>用线性变换+非线性函数激活的方式进行特征变换。以分类为目的进行学习的时候，网络中的参数会以分类正确为目的自行调整，也就是自动提取合适的特征。</font >（回归问题，去掉最后一层的激活函数就行。输出层激活函数只是为了结果有概率意义）神经网络最大的惊喜就是自动组合特征。
### 1.2 BP神经网络基本原理
- MLP网络中，每个节点都接收前一层所有节点的信号的加权和，累加后进行激活，再传入下一层的节点。这个过程和动物神经元细胞传递信号的过程类似，所以叫神经网络，各节点称为神经元。
- 每层神经元个数称为神经网络的宽度，层数称为神经网络的深度。所以多层神经网络称为深度神经网络DNN。
- 神经元数据过多会造成过拟合和运算量过大。层中神经元过多，相当于该层转换后的特征维度过高，会造成维数灾难。
- DNN仍然使用交叉熵损失函数。
### 1.3 神经网络的多分类
- 机器学习模型相对简单，参数较少。可以通过训练N个二分类模型来完成多分类。而深度学习中，模型参数较多，训练多个模型不实际。而且多分类任务的前层特征变换是一致的，没必要训练多个模型。一般是输出层采用softmax函数来完成。
- softmax做多分类只适用于类别互斥且和为1的情况。如果和不为1，可以加一个其它类。不互斥时不能保证分母能归一化。
### 1.4 二分类使用softmax还是sigmoid好？

参考[《速通8-DNN神经网络学习笔记》](https://blog.csdn.net/qq_56591814/article/details/120876730)
- softmax等于分别学习w1和w2，而sigmoid等于学这两个的差值就行了。sigmoid是softmax在二分类上的特例。二分类时sigmoid更好。因为我们只关注w1和w2的差值，但是不关心其具体的值。
- softmax的运算量很大，因为要考虑别的概率值。一般只在神经网络最后一层用（多分类时）。中间层神经元各算各的，不需要考虑别的w数值，所以中间层不需要softmax函数。
- softmax函数分子：通过指数函数，将实数输出映射到零到正无穷。softmax函数分母：将所有结果相加，进行归一化。
- softmax和sigmoid一样有饱和区，x变化几乎不会引起softmax输出的变化，而且饱和区导数几乎为0，无法有效学习。所以需要合适的初始化，控制前层输出的值域。
- 两个函数的代码实现如下图
```python
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1.0/(1+np.exp(-x))
	 
sigmoid_inputs = np.arange(-10,10)
sigmoid_outputs=sigmoid(sigmoid_inputs)
print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
print("Sigmoid Function Output :: {}".format(sigmoid_outputs))
 
plt.plot(sigmoid_inputs,sigmoid_outputs)
plt.xlabel("Sigmoid Inputs")
plt.ylabel("Sigmoid Outputs")
plt.show()
```
```python
def softmax(x):
    orig_shape=x.shape
    if len(x.shape)>1:
        #Matrix
        #shift max whithin each row
        constant_shift=np.max(x,axis=1).reshape(1,-1)
        x-=constant_shift
        x=np.exp(x)
        normlize=np.sum(x,axis=1).reshape(1,-1)
        x/=normlize
    else:
        #vector
        constant_shift=np.max(x)
        x-=constant_shift
        x=np.exp(x)
        normlize=np.sum(x)
        x/=normlize
    assert x.shape==orig_shape
    return x
 
softmax_inputs = np.arange(-10,10)
softmax_outputs=softmax(softmax_inputs)
print("Sigmoid Function Input :: {}".format(softmax_inputs))
print("Sigmoid Function Output :: {}".format(softmax_outputs))
# 画图像
plt.plot(softmax_inputs,softmax_outputs)
plt.xlabel("Softmax Inputs")
plt.ylabel("Softmax Outputs")
plt.show()
```
### 1.5 梯度下降和链式求导
### 1.6度量学习
## 二、矩阵求导术
线性代数参考[《线性代数一（基本概念）》](https://blog.csdn.net/qq_16555103/article/details/84838943)
### 2.1 标量对向量求导
例如标量z对向量X求导，就是看X各元素变化对z的影响。所以结果是z对X各元素求导，结果也是一个同尺寸向量。
### 2.2 向量对向量求导
<font color='red'>列向量对行向量求导，结果是一个矩阵，方便后面进行链式求导中的导数连乘。</font>（向量既可以写成列的形式，也可以写成行的形式。列向量可以直接对列向量求导，但是维数太多不便于操作，而且没法进行连乘，一般没人这么干。）
- m×1维列向量$y$对n×1维列向量$x$求导，等于$y$的每个元素$y_{i}$分别对$x$求导，得到m×1维导数。而导数每个元素都是标量对向量求导，结果是n×1维列向量。所以最终结果是mn×1维的列向量。行向量亦然。
- 神经网络中一般都用列向量。，<font color='red'>如果列向量$y$对列向量$x$求导，结果太长不便于计算。一般是$y$对$x$的转置求导，即$\frac{\partial y}{\partial x^{T}}$，最后结果是一个m×n的矩阵。
### 2.3 标量对矩阵的矩阵
向量可以看成某一维为1的矩阵（行为1或列为1），同理，标量对m×n维矩阵求导，是对矩阵中每个元素求导，结果也是一个m×n维的结果。
### 2.4 向量求导及链式法则
对于$$x_{3}=Wx_{2}$$
展开之后有：
$$\begin{pmatrix}
x_{31}\\ 
x_{32}\\ 
...\\ 
x_{3m}\end{pmatrix}=\begin{pmatrix}
m_{11} & m_{12} &... &m _{1n} \\ 
m_{21} &m _{22} &...  &m_{2n} \\ 
 ...& ... & ... &... \\ 
 m_{m1}&m_{m2}  &... & m_{mn} 
\end{pmatrix}\times \begin{pmatrix}
x_{21}\\ 
x_{22}\\ 
...\\ 
x_{2n}\end{pmatrix}$$
所以$\frac{\partial x_{3}}{\partial x_{2}^{T}}=W$，$\frac{\partial x_{3}}{\partial W}=x_{2}^{T}$
- 即之前列向量对列向量转置求导，推出结果是mn的矩阵，但是各元素的具体值不知道。这里直接求出，具体值为矩阵W
- 求导要一直盯着尺寸看。

公式1——标量对向量求导链式法则：
向量$x_{1}...x_{n}$为神经网络各层的输入，最终输出结果是标量z。存在依赖关系：$x_{1}\rightarrow x_{2}\rightarrow x_{3}...\rightarrow x_{n}\rightarrow z$，则有：
<font color='red'>$$\frac{\partial z}{\partial x_{1}}=(\frac{\partial x_{n} }{\partial x_{n-1}^{T}}\cdot \frac{\partial x_{n-1} }{\partial x_{n-2}^{T}}...\frac{\partial x_{2} }{\partial x_{1}^{T}})^{T}\frac{\partial z}{\partial x_{n}}$$
- 假如$x_{n}、x_{n-1}、x_{n-2}$分别是m、n、k维的列向量，则上式右边第一项分别是m×n和n×k的矩阵，这两个矩阵才有相乘的可能，结果是m×k的矩阵。所以必须是列向量对列向量的转置求导
- 假如$x_{1}$是h维列向量，<font color='red'>则括号内最终结果是m×h维矩阵，$\frac{\partial z}{\partial x_{n}}$结果是m维列向量，无法直接相乘，所以括号内的矩阵必须转置。

公式2——标量对矩阵求导链式法则：
W为矩阵，$x$和$y$是向量，有$y=Wx$。且标量$z=f(y)$，则：
$$\frac{\partial z}{\partial W}=\frac{\partial z }{\partial y}\cdot x^{T}$$
参照上面写的：$\frac{\partial x_{3}}{\partial W}=x_{2}^{T}$

### 2.5 BP反向传播
Loss对任意层$W^k$求导有：
$$\frac{\partial Loss}{\partial W^{k}}=\frac{\partial Loss}{\partial d_{j}^{L}}\cdot \frac{\partial d_{j}^{L}}{\partial W^{k}}$$
对于一个L层的神经网络，j表示L层第j维数据（也就是第j个神经元）
第一项$\frac{\partial Loss}{\partial d_{j}^{L}}=y_{j}'-y_{j}$，是一个标量。(即loss对最后一层某个神经元导数为一个标量，推导见P119）
又因为：
$$d^{k}=W^{k}\cdot a^{k-1}$$
$$a^{k-1}=f(d^{l-1}+w_{0}^{k-1})$$
后一项是标量对矩阵求导，根据公式2有： 
$$\frac{\partial d_{j}^{L}}{\partial W^{k}}= \frac{\partial d_{j}^{L}}{\partial d^{k}}\cdot (a^{k-1})^{T}$$
>将$d_{j}^{L}$看做是由向量$d^{L}$映射成标量

上式右边第一项是标量对向量求导，根据公式1有：
$$\frac{\partial d_{j}^{L}}{\partial d^{k}}=\frac{\partial d_{j}^{L}}{\partial a^{L-1}}\cdot (\frac{\partial a^{L-1}}{\partial (d^{L-1})^{T}}\cdot \frac{\partial d^{L-1}}{\partial (a^{L-2})^{T}}...\frac{\partial d^{k+1}}{\partial (a^{k})^{T}}\cdot \frac{\partial a^{k}}{\partial (d^{k})^{T}})^T$$
依次求三项导数有：
$$\frac{\partial d_{j}^{L}}{\partial a^{L-1}}=W^L$$
$$\frac{\partial a^{m}}{\partial (d^{m})^{T}}=f'(d^m)$$
$$\frac{\partial d^{m+1}}{\partial (a^{m})^{T}}=W^{m+1}$$
将以上结果联合起来就是：
$$\frac{\partial Loss}{\partial W^{k}}=\frac{\partial Loss}{\partial d_{j}^{L}}\cdot \frac{\partial d_{j}^{L}}{\partial W^{k}}=(y'_{j}-y_{j})\cdot (a^{k-1})^{T}\cdot W^{L}\cdot f'(d^{L-1})\cdot  W^{L-1}\cdot f'(d^{L-2})... W^{k+1}\cdot f'(d^{k})$$

第m层激活函数得导数有：
$$a^{m}=f(d^{m}+w_{0}^{m})$$
展开后为：
$$\begin{pmatrix}
a_{1}\\ 
a_{2}\\ 
...\\ 
a_{M}\end{pmatrix}=\begin{pmatrix}
f(d_{1}+w_{0}^{1})\\ 
f(d_{2}+w_{0}^{2})\\ 
...\\ 
f(d_{M})+w_{0}^{M}\end{pmatrix}$$
其实是省去了上标层数m，其中第m层共有M个神经元。相当于向量d数乘之后得到向量a，每个元素按位操作。$a_{1}=f(d_{1}+w_{0}^{1})$,$a_{1}$只和$d_{1}$有关，和向量d其它元素无关，对d其它分量结果为0。

$$\frac{\partial a^{m}}{\partial (d^{m})^{T}}=f'(d^m)$$
列向量对行向量求导，结果是一个矩阵。所以有：
$$f'(d^m)=\begin{pmatrix}
f'(d_{1}+w_{0}^{1}) &0  &...  &0 \\ 
 0& f'(d_{2}+w_{0}^{2}) & ... & 0\\ 
0 &...  & ... &... \\ 
0& 0 & ... & f'(d_{M}+w_{0}^{M})
\end{pmatrix}$$
- 如果激活函数f的导数f'>1，经过多次连乘，最后结果会非常大，即梯度爆炸。会产生震荡甚至溢出。
- 如果f'<1，梯度消失，W几乎不会更新。
### 2.5 激活函数及其导数
1. sigmoid函数：$sigmoid(d)=\frac{1}{1+e^{-d}}$导数为：
$f'=f(1-f)=0.25-(f-0.5)^2$,后一项非负，当f=0.5时有最大值0.25。所以其值域为（0，0.25）
2. softmax函数及其导数，参考[《Softmax函数及其导数》](https://blog.csdn.net/cassiePython/article/details/80089760)
![在这里插入图片描述](https://img-blog.csdnimg.cn/7b1f7e77f53045e0bf79b29e00d463d7.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_19,color_FFFFFF,t_70,g_se,x_16)

3. relu函数：$f(x)=max(0,x)$。其导数w为：
$$f'(x)=\left\{\begin{matrix}
0 ,x<0\\ 
1,x>0\end{matrix}\right.$$
4. Tanh函数
$$f(d)=Tanh(d)=\frac{e^{d}-e^{-d}}{e^{d}+e^{-d}}=sigmoid(2d-1)$$
导数为：$$\frac{\partial a}{\partial d}=1-f^{2}$$
Tanh值域（-1，1)，导数值域（0,1）。
## 三、神经网络调优
海量的数据和强大的算力为深度学习的发展提供了条件。但是也带来一些新的问题：
- 网络太深带来梯度消失和梯度爆炸
- 损失函数太复杂，有大量的极小点和鞍点，如果学习方法不合适，损失函数可能无法降低
- 参数过多造成过拟合
### 3.1 激活函数得选型
合适的激活函数需要满足的条件
1. 零均值输出：例如tanh，零均值输出，有正有负。W各个维度更新方向可以不同，避免单向更新学习慢的问题（各维度只能同增或者同减，走Z字路线。比如loss极小值在左下方，只能先左后下）这一点softmax、sigmoid、relu都不满足，都是非负的
2. 适当的线性。激活函数对输入的变化不宜过于激烈，否则输入轻微变化造成输出剧烈变化，稳定性不好。softmax、sigmoid、relu、tanh都有近似线性的区域可以满足
3. 导数不宜过大或过小，否则梯度消失或者爆炸。这也是relu成为神经网络首选的原因。tanh导数是0-1，比sigmoid好一点。
4. 导数的单调性。避免导数有正有负造成学习震荡，例如三角函数
5. 有值域的控制。避免多次激活后输出太大。例如$y=x^2$
6. 没有超参数，计算简单。Maxout、PReLu函数有超参数，设置本身依赖经验。而且ReLu计算足够简单
### 3.2 Relu激活函数及其变体
参考[《从ReLU到GELU，一文概览神经网络的激活函数》](https://blog.csdn.net/Sophia_11/article/details/103998468?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163769303916780357222745%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163769303916780357222745&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-7-103998468.first_rank_v2_pc_rank_v29&utm_term=gelu&spm=1018.2226.3001.4187)
线性整流函数Relu:$$f(x)=max(0,x)$$
- d≥0时，f(d)=d，导数为1，不存在梯度消失或者梯度爆炸，也不存在饱和区
- d＜0时，f(d)=0，梯度为0 ，神经元死亡
梯度消失和激活函数饱和困扰业界多年，直到RELU的出现。

神经元真死和假死：
真死：
- 无论w和x各维度如何变化，恒有$d_{j}^{(l)}=w_{j}^{(l)}a^{(l-1)}$≤0，神经元死亡。（l表示神经网络层数，j表示某层中神经元节点j）。$a^{(l-1)}≥0$为上一层神经网络的输出。
- 所以是<font color='red'>$w_{j}^{(l)}$各元素都是很大的负数时，恒有d≤0。这是参数初始化错误或者lr过大导致权重$w_{j}^{(l)}$更新过大造成的。因此用Relu做激活函数，lr不宜过大，权重初始化要合理

假死：饱和形式之一
- 恰巧造成$w_{j}^{(l)}a_{j}^{(l-1)}≤0$，重新训练时有可能会恢复正常。神经元大多数时是假死。
- 适当假死可以提升神经网络效果：
	- 如果没有神经元假死，$a_{j}^{(l-1)}≥0$，f(d)=d，Relu作为激活函数就是纯线性的，失去意义。
	- 假死神经元就是对某些输入特征不进行输出，达到<font color='red'>特征选择的作用（门控决策）

LRelu： d＜0时，f(d)=k，k为超参数，在0-1之间
PRelu： d＜0时，f(d)=k，k是可学习的参数。
Maxout：$w_{j}^{(l)}(1)$到$w_{j}^{(l)}(n)$有多个参数，分别和上一层输出相乘，即每个神经元节点j有多个输入，最终输出选最大值。(P133-134)

### 3.3 高斯误差线性单元激活函数gelu
>参考[《超越ReLU却鲜为人知，3年后被挖掘》](https://mp.weixin.qq.com/s/XqPKsM8yq8rbFFvvJKCtkQ)
参考[《GELU 论文》](https://arxiv.org/pdf/1606.08415.pdf)

#### 3.3.1 GELU概述
BERT、RoBERTa、ALBERT 等目前业内顶尖的 NLP 模型都使用了这种激活函数。另外，在 OpenAI 声名远播的无监督预训练模型 GPT-2 中，研究人员在所有编码器模块中都使用了 GELU 激活函数。在计算机视觉、自然语言处理和自动语音识别等任务上，使用 GELU 激活函数比使用ReLU 或 ELU 效果更好。

随着网络深度的不断增加，<font color='red'>利用 Sigmoid 激活函数来训练被证实不如非平滑、低概率性的 ReLU 有效（Nair & Hinton, 2010），因为 ReLU 基于输入信号做出门控决策。

深度学习中为了解决过拟合，会随机正则化（如在隐层中加入噪声）或采用 dropout 机制。<font color='red'>这两个选择是和激活函数割裂的。非线性和 dropout 共同决定了神经元的输出，而随机正则化在执行时与输入无关。

由此提出高斯误差线性单元（Gaussian Error Linear Unit，GELU）。<font color='red'>GELU 与随机正则化有关，因为它是自适应 Dropout 的修正预期（Ba & Frey, 2013）</font >。这表明神经元输出的概率性更高。

#### 3.3.2 GELU数学表示
<font color='red'>Dropout、ReLU 等机制都希望将「不重要」的激活信息规整为零。即对于输入的值，我们根据它的情况乘上 1 或 0。</font >或者说，对输入x乘上一个伯努利分布 Bernoulli(Φ(x))，其中Φ(x) = P(X ≤ x)。（x服从于标准正态分布 N(0, 1)）

对于一部分Φ(x)，它直接乘以输入 x，而对于另一部分 (1 − Φ(x))，它们需要归零。随着 x 的降低，它被归零的概率会升高。对于 ReLU 来说，这个界限就是 0。

我们经常希望神经网络具有确定性决策，这种想法催生了 GELU 激活函数的诞生。具体来说可以表示为：Φ(x) × Ix + (1 − Φ(x)) × 0x = xΦ(x)。可以理解为，<font color='red'>不太严格地说，上面这个表达式可以按当前输入 x 比其它输入大多少来缩放 x。</font >

高斯概率分布函数通常根据损失函数计算，因此研究者定义高斯误差线性单元（GELU）为：
$$GELU(x)=xP(X\leqslant x)=x\Phi (x)$$
上面这个函数是无法直接计算的，因此可以通过另外的方法来逼近这样的激活函数，研究者得出来的表达式为：
$$GELU(x)=0.5x(1+tanh[\sqrt{\frac{2}{\pi }}(x+0.044175x^{3})])$$
或：$$GELU(x)=x\sigma (1.702x)$$
其中 σ() 是标准的 sigmoid 函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/adbc2fffd2e04be5adf7ccc5702ecd40.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)

当 x 大于 0 时，输出为 x；但 x=0 到 x=1 的区间除外，这时曲线更偏向于 y 轴。
没能找到该函数的导数，所以我使用了 WolframAlpha 来微分这个函数。结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/338d92546ede439195600449445c66a2.png)
微分的GELU函数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/0c8f7ea9572e4068946a53279375f94c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_18,color_FFFFFF,t_70,g_se,x_16)
GELU 的近似实现方式有两种，借助 tanh() 和借助σ()。我们在 GPT-2 的官方代码中也发现，更多研究者采用了 tanh() 的实现方式尽管它看起来要比 xσ(1.702x) 复杂很多。

```python
# GPT-2 的 GELU 实现
def gelu(x):
	return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
```
### 3.4 Xavier权重初始化
参考《苏神文章解析（6篇）》
[《浅谈Transformer的初始化、参数化与标准化》](https://kexue.fm/archives/8620)
[《从几何视角来理解模型参数的初始化策略》](https://kexue.fm/archives/7180)
- 如果两个神经元参数完全相同，则梯度也相同，更新幅度一致，最后输出也相同。看上去是两个神经元，但其实相当于就一个，是互相冗余的。所以 <font color='red'>初始化W矩阵时要破坏其对称性，独立均匀分布的随机初始化就可以做到。（正态分布采样容易集中在均值附近，可能造成权重对称）均匀分布有两个参数$\mu/ \sigma$ ，均值和方差。
-  真实场景中，特征都是客观事实，一般都是＞0。如果W矩阵各元素都是＞0，则relu函数不起作用。都＜0，神经元都死了。所以 <font color='red'>W矩阵各元素需要有正有负，所以一般均匀分布选择均值为0。$\sigma$ 越大，w采样到大值的可能性越大,即$W_{i,j}^{l}\propto \sigma _{i,j}^{l}$
- <font color='red'>从均值为0、方差为1/m的随机分布中独立重复采样，这就是Xavier初始化（无激活函数时）relu做激活函数时，方差为2/m
- NTK参数化：均值为0、方差为1的随机分布来初始化，但是将输出结果除以$\sqrt{m}$。利用NTK参数化后，所有参数都可以用方差为1的分布初始化，这意味着每个参数的量级大致都是相同的O(1)级别，让我们更平等地处理每一个参数，于是我们可以设置较大的学习率。
- 对于$d_{j}^{(l)}=\sum_{i=1}^{M(l-1)}w_{i,j}^{(l)}a_{i}^{(l-1)}$，M(l-1)表示上一层神经元个数。<font color='red'>$d_{j}^{(l)}$不宜过大,$W_{i,j}^{l}\propto \frac{1}{M^{l-1}},\sigma _{i,j}^{l}\propto \frac{1}{M^{l-1}}$</font >。否则relu前向传播时，输出会层层放大而溢出。softmax、sigmoid、tanh进入饱和区，导数基本为0。
其它推导见P137




