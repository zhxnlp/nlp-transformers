@[toc]
## 一、 Boosting算法原理
- Bagging：<font color='red'>通过Bootstrap 的方式对全样本数据集进行抽样得到抽样子集，对不同的子集使用同一种基本模型进行拟合，然后投票得出最终的预测。
- Bagging主要通过降低方差的方式减少预测误差</font>
- Boosting：<font color='red'>使用同一组数据集进行反复学习，得到一系列简单模型，然后组合这些模型构成一个预测性能十分强大的机器学习模型。
- Boosting通过不断减少偏差的形式提高最终的预测效果，与Bagging有着本质的不同。

在概率近似正确（PAC）学习的框架下：

1. 弱学习：识别准确率略高于1/2（即准确率仅比随机猜测略高的学习算法）
2. 强学习：识别准确率很高并能在多项式时间内完成的学习算法
3. 强可学习和弱可学习是等价的，弱可学习算法，能提升至强可学习算法

- 提升方法：从弱学习算法出发，反复学习，得到一系列弱分类器(又称为基本分类器)，再通过一定的形式去组合这些弱分类器构成一个强分类器。而弱可学习算法比强可学习算法容易得多。
- 大多数的Boosting方法都是<font color='red'>通过改变训练数据集的概率分布(训练数据不同样本的权值)，针对不同概率分布的数据调用弱分类算法学习一系列的弱分类器。

Boosting方法关键点：
1. 每一轮学习应该如何改变数据的概率分布
2. 如何将各个弱分类器组合起来

## 二、 Adaboost算法
### 2.1 Adaboost算法原理
Adaboost解决上述的两个问题的方式是：
1. <font color='red'>提高那些被前一轮分类器错误分类的样本的权重</font>，而降低那些被正确分类的样本的权重。错误分类样本权重的增大而在后一轮的训练中“备受关注”。
2. 各个弱分类器<font color='red'>通过采取加权多数表决的方式组合</font>。分类错误率低的弱分类器权重高，分类错误率较大的弱分类器权重低。
3. Adaboost等Boosting模型增加了计算的复杂度和计算成本，用降低偏差的方式去减少总误差，但是过程中引入了方差，可能出现过拟合

4. <font color='deeppink'>Boosting方式无法做到现在流行的并行计算的方式进行训练，因为每一步迭代都要基于上一部的基本分类器。
5. Adaboost算法是由基本分类器组成的加法模型，损失函数为指数损失函数。

Adaboost算法：
- 输入：二分类的训练数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，特征$x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}$，类别$y_{i} \in \mathcal{Y}=\{-1,+1\}$，$\mathcal{X}$是特征空间，$\mathcal{Y}$是类别集合，其中每个样本点由特征与类别组成。
- 输出：最终分类器$G(x)$。

(1) 初始化训练集样本的权值分布：$$D_{1}=\left(w_{11}, \cdots, w_{1 i}, \cdots, w_{1 N}\right), \quad w_{1 i}=\frac{1}{N}, \quad i=1,2, \cdots, N$$其中权值是均匀分布，使得第一次没有先验信息的条件下每个样本在基本分类器的学习中作用一样。

(2) 对于学习轮次m=1,2,...,M
1. 使用具有权值分布$D_m$的训练数据集进行学习，得到基本分类器：$G_{m}(x): \mathcal{X} \rightarrow\{-1,+1\}$
2. 计算$G_m(x)$在训练集上的分类误差率$$e_{m}=\sum_{i=1}^{N} P\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)=\sum_{i=1}^{N} w_{m i} I\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)$$
$w_{m i}$代表了在$G_m(x)$中分类错误的样本权重和，这点直接说明了权重分布$D_m$与$G_m(x)$的分类错误率$e_m$有直接关系。
3. 计算$G_m(x)$的系数$\alpha_{m}=\frac{1}{2} \log \frac{1-e_{m}}{e_{m}}$，这里的log是自然对数ln
	- $\alpha_{m}$表示了$G_m(x)$在最终分类器的重要性程度。
	- 当$e_{m} \leqslant \frac{1}{2}$时，$\alpha_{m} \geqslant 0$，并且$\alpha_m$随着$e_m$的减少而增大，因此分类错误率越小的基本分类器在最终分类器的作用越大！                       
4. 更新训练数据集的权重分布
   $$D_{m+1}=(w_{m+1,1}, \cdots, w_{m+1, i}, \cdots, w_{m+1, N})$$
   $$w_{m+1, i}=\left\{\begin{array}{ll}
\frac{w_{m i}}{Z_{m}} \mathrm{e}^{-\alpha_{m}}, & G_{m}\left(x_{i}\right)=y_{i} \\
\frac{w_{m i}}{Z_{m}} \mathrm{e}^{\alpha_{m}}, & G_{m}\left(x_{i}\right) \neq y_{i}
\end{array}\right.$$ 
   $$Z_{m}=\sum_{i=1}^{N} w_{m i} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right)$$                                       
   这里的$Z_m$是规范化因子，使得$D_{m+1}$成为概率分布。
    
   从上式可以看到：被基本分类器$G_m(x)$错误分类的样本的权重扩大，被正确分类的样本权重减少，二者相比相差$\mathrm{e}^{2 \alpha_{m}}=\frac{1-e_{m}}{e_{m}}$倍。     

(3) 构建基本分类器的线性组合$f(x)=\sum_{m=1}^{M} \alpha_{m} G_{m}(x)$，得到最终的分类器                       

$$
\begin{aligned}
G(x) &=\operatorname{sign}(f(x)) \\
&=\operatorname{sign}\left(\sum_{m=1}^{M} \alpha_{m} G_{m}(x)\right)
\end{aligned}
$$
$$sign(x)=\begin{cases}
1 & \text{ if } x\geqslant 0 \\ 
-1 & \text{ if } x< 0 
\end{cases}   $$                       
线性组合$f(x)$实现了将M个基本分类器的加权表决，系数$\alpha_m$标志了基本分类器$G_m(x)$的重要性，值得注意的是：所有的$\alpha_m$之和不为1。$f(x)$的符号决定了样本x属于哪一类,其绝对值表示分类的确信度。

<font color='red'>简单来说：计算M个基本分类器，每个分类器的错误率、模型权重及样本权重
1. 均匀初始化样本权重$D_{1}$
2. 对于轮次m，针对当前权重$D_{m}$ 学习分类器 $G_{m}(x)$，并计算其分类错误率$e_{m}$。
3. 计算分类器$G_m(x)$的<font color='red'>权重系数$\alpha_{m}=\frac{1}{2} \log \frac{1-e_{m}}{e_{m}}$</font>。$e_{m} \leqslant \frac{1}{2}$时，$\alpha_{m} \geqslant 0$，并且$\alpha_m$随着$e_m$的减少而增大，因此<font color='red'>分类错误率越小的基本分类器在最终分类器的作用越大！
4. 更新权重分布 $$w_{m+1, i}=\left\{\begin{array}{ll}
\frac{w_{m i}}{Z_{m}} \mathrm{e}^{-\alpha_{m}}, & G_{m}\left(x_{i}\right)=y_{i} \\
\frac{w_{m i}}{Z_{m}} \mathrm{e}^{\alpha_{m}}, & G_{m}\left(x_{i}\right) \neq y_{i}
\end{array}\right.$$ 
一般来说$\alpha_{m} \geqslant 0，e^0=1$。被基本分类器$G_m(x)$错误分类的样本的权重扩大，被正确分类的样本权重减少。$e_{m}$减小，$\alpha_{m}$增大，${w_{m+1, i}}$增大。即错误率越低的分类器，分类器权重和错误样本权重都越大，感觉上越准确准确的分类器学习力度越大。
5. 基本分类器加权组合表决
6. 总结：Adaboost不改变训练数据，而是改变其权值分布，使每一轮的基学习器学习不同权重分布的样本集，最后加权组合表决。

### 2.2 Adaboost算法举例
下面，我们使用一组简单的数据来手动计算Adaboost算法的过程：(例子来源http://www.csie.edu.tw)                                                               

训练数据如下表，假设基本分类器的形式是一个分割$x<v$或$x>v$表示，<font color='red'>阈值v由该基本分类器在训练数据集上分类错误率$e_m$最低确定。</font>                                                
$$
\begin{array}{ccccccccccc}
\hline \text { 序号 } & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\
\hline x & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 \\
y & 1 & 1 & 1 & -1 & -1 & -1 & 1 & 1 & 1 & -1 \\
\hline
\end{array}
$$                                          
解：                        
初始化样本权值分布
$$
\begin{aligned}
D_{1} &=\left(w_{11}, w_{12}, \cdots, w_{110}\right) \\
w_{1 i} &=0.1, \quad i=1,2, \cdots, 10
\end{aligned}
$$                                  
1. <font color='red'>对m=1:                      
   - 在权值分布$D_1$的训练数据集上，遍历每个结点并计算分类误差率$e_m$，阈值取v=2.5时分类误差率最低，那么基本分类器为：
   $$
   G_{1}(x)=\left\{\begin{array}{ll}
   1, & x<2.5 \\
   -1, & x>2.5
   \end{array}\right.
   $$                         
   - 样本7.8.9分错，$G_1(x)$在训练数据集上的误差率为$e_{1}=P\left(G_{1}\left(x_{i}\right) \neq y_{i}\right)=0.1*3=0.3$。                                        
   - 计算$G_1(x)$的系数：$\alpha_{1}=\frac{1}{2} \log \frac{1-e_{1}}{e_{1}}=0.4236$               
   - 更新训练数据的权值分布：                  
   $$
   \begin{aligned}
   D_{2}=&\left(w_{21}, \cdots, w_{2 i}, \cdots, w_{210}\right) \\
   w_{2 i}=& \frac{w_{1 i}}{Z_{1}} \exp \left(-\alpha_{1} y_{i} G_{1}\left(x_{i}\right)\right), \quad i=1,2, \cdots, 10 \\
   D_{2}=&(0.07143,0.07143,0.07143,0.07143,0.07143,0.07143,\\
   &0.16667,0.16667,0.16667,0.07143) \\
   f_{1}(x) &=0.4236 G_{1}(x)=array([ 0.4236,  0.4236,  0.4236, -0.4236, -0.4236, -0.4236, -0.4236,
       -0.4236, -0.4236, -0.4236])
   \end{aligned}
   $$
<font color='deeppink'>权重为[0.07143×7,0.16667×3]

2. <font color='red'>对于m=2：                   
   - 在权值分布$D_2$的训练数据集上，遍历每个结点并计算分类误差率$e_m$，阈值取v=8.5时分类误差率最低，那么基本分类器为：                  
   $$
   G_{2}(x)=\left\{\begin{array}{ll}
   1, & x<8.5 \\
   -1, & x>8.5
   \end{array}\right.
   $$                       
   - 样本4.5.6分错，$G_2(x)$在训练数据集上的误差率为$e_2 = 0.07143*3=0.2143$                    
   - 计算$G_2(x)$的系数：$\alpha_2 = 0.6496$                        
   - 更新训练数据的权值分布：                  
   $$
   \begin{aligned}
   D_{3}=&(0.0455,0.0455,0.0455,0.1667,0.1667,0.1667\\
   &0.1060,0.1060,0.1060,0.0455) \\
   f_{2}(x) &=0.4236 G_{1}(x)+0.6496 G_{2}(x)=array([ 1.0732,  1.0732,  1.0732,  0.226 ,  0.226 ,  0.226 ,  0.226 ,
        0.226 ,  0.226 , -1.0732])
   \end{aligned}
   $$                   
   权重为[0.00455×4,0.1060×3,0.16667×3]。
3. <font color='deeppink'>对m=3：                          
   - 在权值分布$D_3$的训练数据集上，遍历每个结点并计算分类误差率$e_m$，阈值取v=5.5时分类误差率最低，那么基本分类器为：                     
   $$
   G_{3}(x)=\left\{\begin{array}{ll}
   1, & x>5.5 \\
   -1, & x<5.5
   \end{array}\right.
   $$                               
   - 样本1.2.3.10分错，$G_3(x)$在训练数据集上的误差率为$e_3 =0.0455*4= 0.1820$                       
   - 计算$G_3(x)$的系数：$\alpha_3 = 0.7514$                                 
   - 更新训练数据的权值分布：                    
   $D_{4}=(0.125,0.125,0.125,0.102,0.102,0.102,0.065,0.065,0.065,0.125)$                       
$$f_{3}(x)=0.4236 G_{1}(x)+0.6496 G_{2}(x)+0.7514 G_{3}(x)=array([ 0.3218,  0.3218,  0.3218, -0.5254, -0.5254, -0.5254,  0.9774,
        0.9774,  0.9774, -0.3218])$$
        分类器$\operatorname{sign}\left[f_{3}(x)\right]$在训练数据集上的误分类点的个数为0。                                
最终分类器为：$G(x)=\operatorname{sign}\left[f_{3}(x)\right]=\operatorname{sign}\left[0.4236 G_{1}(x)+0.6496 G_{2}(x)+0.7514 G_{3}(x)\right]$

==可以看到每次样本权重和都为1，分类器错误率越来越低，分类器权重越来越高。最终分类器结果就是基分类器结果乘以其权重的加和==
### 2.3 Adaboos代码举例
数据集：CI的机器学习库里的[葡萄酒数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)，该数据集包含了178个样本和13个特征，从不同的角度对不同的化学特性进行描述，最终预测红酒属于哪一个类别。(案例来源《python机器学习(第二版》)

```python
# 引入数据科学相关工具包：
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline
import seaborn as sns

# 加载训练数据：         
wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', 
                'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
# 数据查看：
print("Class labels",np.unique(wine["Class label"]))
wine.head()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/8e3e3b19c1f04d1b9257419b42646479.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

下面对数据做简单解读：

Class label：分类标签
Alcohol：酒精
Malic acid：苹果酸
Ash：灰
Alcalinity of ash：灰的碱度
Magnesium：镁
Total phenols：总酚
Flavanoids：黄酮类化合物
Nonflavanoid phenols：非黄烷类酚类
Proanthocyanins：原花青素
Color intensity：色彩强度
Hue：色调
OD280/OD315 of diluted wines：稀释酒OD280 OD350
Proline：脯氨酸

```python
# 数据预处理
# 仅仅考虑2，3类葡萄酒，去除1类
wine = wine[wine['Class label'] != 1]
y = wine['Class label'].values
X = wine[['Alcohol','OD280/OD315 of diluted wines']].values

# 将分类标签变成二进制编码：
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# 按8：2分割训练集和测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)  # stratify参数代表了按照y的类别等比例抽样
```

```python
# 使用单一决策树建模
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',random_state=1,max_depth=1)
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train,y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train,y_train_pred)
tree_test = accuracy_score(y_test,y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train,tree_test))

Decision tree train/test accuracies 0.916/0.875
```

```python
# 使用sklearn实现Adaboost(基分类器为决策树)
'''
AdaBoostClassifier相关参数：
base_estimator：基本分类器，默认为DecisionTreeClassifier(max_depth=1)
n_estimators：终止迭代的次数
learning_rate：学习率
algorithm：训练的相关算法，{'SAMME'，'SAMME.R'}，默认='SAMME.R'
random_state：随机种子
'''
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)
ada = ada.fit(X_train,y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train,y_train_pred)
ada_test = accuracy_score(y_test,y_test_pred)
print('Adaboost train/test accuracies %.3f/%.3f' % (ada_train,ada_test))

Adaboost train/test accuracies 1.000/0.917
```
结果分析：单层决策树似乎对训练数据欠拟合，而Adaboost模型正确地预测了训练数据的所有分类标签，而且与单层决策树相比，Adaboost的测试性能也略有提高。然而，为什么模型在训练集和测试集的性能相差这么大呢？我们使用图像来简单说明下这个道理！

```python
# 画出单层决策树与Adaboost的决策边界：
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2,sharex='col',sharey='row',figsize=(12, 6))
for idx, clf, tt in zip([0, 1],[tree, ada],['Decision tree', 'Adaboost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],X_train[y_train==0, 1],c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],X_train[y_train==1, 1],c='red', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.tight_layout()
plt.text(0, -0.2,s='OD280/OD315 of diluted wines',ha='center',va='center',fontsize=12,transform=axarr[1].transAxes)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/e7ee7c34eacb4218b8e52011424f4055.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
从上面的决策边界图可以看到：
- Adaboost模型的决策边界比单层决策树的决策边界要复杂的多。也就是说，<font color='deeppink'>Adaboost试图用增加模型复杂度而降低偏差的方式去减少总误差，但是过程中引入了方差，可能出现过拟合
- 与单个分类器相比，Adaboost等Boosting模型增加了计算的复杂度，在实践中需要仔细思考是否愿意为预测性能的相对改善而增加计算成本
- <font color='deeppink'>Boosting方式无法做到现在流行的并行计算的方式进行训练，因为每一步迭代都要基于上一部的基本分类器。

## 三、 前向分步算法
Adaboost的算法内容：通过计算M个基本分类器，每个分类器的错误率、样本权重以及模型权重。我们可以认为：Adaboost每次学习单一分类器以及单一分类器的参数(权重)。
抽象出Adaboost算法的整体框架逻辑，构建集成学习的一个非常重要的框架----前向分步算法，有了这个框架，我们不仅可以解决分类问题，也可以解决回归问题。
### 3.1加法模型                                
在Adaboost模型中，我们把每个基本分类器合成一个复杂分类器的方法是每个基本分类器的加权和，即：$f(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)$，其中，$b\left(x ; \gamma_{m}\right)$为即基本分类器，$\gamma_{m}$为基本分类器的参数，$\beta_m$为基本分类器的权重，显然这与第二章所学的加法模型。为什么这么说呢？大家把$b(x ; \gamma_{m})$看成是即函数即可。                       
在给定训练数据以及损失函数$L(y, f(x))$的条件下，学习加法模型$f(x)$就是：                        
$$
\min _{\beta_{m}, \gamma_{m}} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{M} \beta_{m} b\left(x_{i} ; \gamma_{m}\right)\right)
$$                      
通常这是一个复杂的优化问题，很难通过简单的凸优化的相关知识进行解决。前向分步算法可以用来求解这种方式的问题，它的基本思路是：因为学习的是加法模型，如果从前向后，每一步只优化一个基函数及其系数，逐步逼近目标函数，那么就可以降低优化的复杂度。具体而言，每一步只需要优化：                    
$$
\min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, \beta b\left(x_{i} ; \gamma\right)\right)
$$                                   
### 3.2 前向分步算法                          
给定数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，$x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}$，$y_{i} \in \mathcal{Y}=\{+1,-1\}$。损失函数$L(y, f(x))$，基函数集合$\{b(x ; \gamma)\}$，我们需要输出加法模型$f(x)$。                         
   - 初始化：$f_{0}(x)=0$                           
   - 对m = 1,2,...,M:                     
      - (a) 极小化损失函数：
      $$
      \left(\beta_{m}, \gamma_{m}\right)=\arg \min _{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\beta b\left(x_{i} ; \gamma\right)\right)
      $$                        
      得到参数$\beta_{m}$与$\gamma_{m}$                                           
      - (b) 更新：                          
      $$
      f_{m}(x)=f_{m-1}(x)+\beta_{m} b\left(x ; \gamma_{m}\right)
      $$                                       
   - 得到加法模型：                           
   $$
   f(x)=f_{M}(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)
   $$                                                     

这样，前向分步算法将同时求解从m=1到M的所有参数$\beta_{m}$，$\gamma_{m}$的优化问题简化为逐次求解各个$\beta_{m}$，$\gamma_{m}$的问题。                           
### 3.3 前向分步算法与Adaboost的关系                              
由于这里不是我们的重点，我们主要阐述这里的结论，不做相关证明，具体的证明见李航老师的《统计学习方法》第八章的3.2节。Adaboost算法是前向分步算法的特例，<font color='deeppink'>Adaboost算法是由基本分类器组成的加法模型，损失函数为指数损失函数。

## 四、梯度提升决策树(GBDT)
- GBDT 的全称是 Gradient Boosting Decision Tree，梯度提升树。GBDT使用的决策树是CART回归树。为什么不用CART分类树呢？因为GBDT每次迭代要拟合的是梯度值，是连续值所以要用回归树
- CART假设决策树都是二叉树，内部节点特征取值为“是”和“否”，等价于递归二分每个特征。对回归树用平方误差最小化准则（回归树中的样本标签是连续数值，所以再使用熵之类的指标不再合适），对分类树用基尼系数最小化准则，进行特征选择生成二叉树
- 回归问题没有分类错误率可言，，<font color='deeppink'>用每个样本的残差表示每次使用基函数预测时没有解决的那部分问题
### 4.1 Decision Tree：CART回归树
最小二乘回归树生成算法见《统计学习方法》P82，算法5.5：
![在这里插入图片描述](https://img-blog.csdnimg.cn/15618f981d2f4b29a6d81ffceca8cfdb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
### 4.2 回归提升树算法
提升法：加法模型+前向分步算法。
提升树：以决策树为基函数（基本分类器）的提升方法，可表示为决策树的加法模型。
决策树：分类问题是二叉分类树，回归问题是二叉回归树

- Adaboost：加法模型+前向分步算法的分类树模型
- GBDT：     加法模型+前向分步算法的回归树模型

分类误差率
- Adaboost算法：使用了分类错误率修正样本权重以及计算每个基本分类器的权重
- GBDT：回归问题没有分类错误率可言，，<font color='deeppink'>用每个样本的残差表示每次使用基函数预测时没有解决的那部分问题

根据以上两点得到<font color='deeppink'>回归问题提升树算法：
                                 
输入：数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y} \subseteq \mathbf{R}$
输出：最终的提升树$f_{M}(x)$                             
   - 初始化$f_0(x) = 0$                        
   - 对m = 1,2,...,M：                  
      - 计算每个样本的残差:$r_{m i}=y_{i}-f_{m-1}\left(x_{i}\right), \quad i=1,2, \cdots, N$                                    
      - 拟合残差$r_{mi}$学习一棵回归树，得到$T\left(x ; \Theta_{m}\right)$                        
      - 更新$f_{m}(x)=f_{m-1}(x)+T\left(x ; \Theta_{m}\right)$
   - 得到最终的回归问题的提升树：$f_{M}(x)=\sum_{m=1}^{M} T\left(x ; \Theta_{m}\right)$                         
   
下面我们用一个实际的案例来使用这个算法：(案例来源：李航老师《统计学习方法》P168，此处省略)                                                             

### 4.3 梯度提升决策树算法(GBDT)
 <font color='red'>GBDT：利用损失函数的负梯度作为回归问题提升树算法中的残差的近似值，拟合回归树。
- 提升树利用加法模型和前向分步算法实现学习的过程，当损失函数为平方损失和指数损失时，每一步优化是相当简单的，也就是我们前面探讨的提升树算法和Adaboost算法。对于一般的损失函数而言，往往每一步的优化不是那么容易
- 针对这一问题，Freidman提出了梯度提升算法(gradient boosting)，利用损失函数的负梯度在当前模型的值$-\left[\frac{\partial L\left(y, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{m-1}(x)}$作为回归问题提升树算法中的残差的近似值，拟合回归树。**与其说负梯度作为残差的近似值，不如说残差是负梯度的一种特例。**

以下开始具体介绍梯度提升算法：                      
输入训练数据集$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in \mathcal{X} \subseteq \mathbf{R}^{n}, y_{i} \in \mathcal{Y} \subseteq \mathbf{R}$和损失函数$L(y, f(x))$，输出回归树$\hat{f}(x)$                              
   - 初始化$f_{0}(x)=\arg \min _{c} \sum_{i=1}^{N} L\left(y_{i}, c\right)$                     
   - 对于m=1,2,...,M：                   
      - 对i = 1,2,...,N计算：$r_{m i}=-\left[\frac{\partial L\left(y_{i}, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{m-1}(x)}$                
      - 对$r_{mi}$拟合一个回归树，得到第m棵树的叶结点区域$R_{m j}, j=1,2, \cdots, J$                           
      - 对j=1,2,...J，计算：$c_{m j}=\arg \min _{c} \sum_{x_{i} \in R_{m j}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+c\right)$                      
      - 更新$f_{m}(x)=f_{m-1}(x)+\sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$                    
   - 得到回归树：$\hat{f}(x)=f_{M}(x)=\sum_{m=1}^{M} \sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$

下面，我们来使用一个具体的案例来说明GBDT是如何运作的([案例来源](https://blog.csdn.net/zpalyq110/article/details/79527653) )：                             
下面的表格是数据：                           
![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-hBCnWMyY-1638484890166)(./6.png)\]](https://img-blog.csdnimg.cn/14f12b8bf27d43ad90e52d67258a25b0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
                                     
学习率：learning_rate=0.1，迭代次数：n_trees=5，树的深度：max_depth=3                                       
平方损失的负梯度为：
$$
-\left[\frac{\left.\partial L\left(y, f\left(x_{i}\right)\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f(x)=f_{t-1}(x)}=y-f\left(x_{i}\right) 
$$                        
$c=(1.1+1.3+1.7+1.8)/4=1.475，f_{0}(x)=c=1.475$                                                      
![](https://img-blog.csdnimg.cn/3319d5746aed4188b03e57d37cb53cca.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
                                  
学习决策树，分裂结点：                                          
![在这里插入图片描述](https://img-blog.csdnimg.cn/3c1ff3f51eb7480e99248dd3fb4dba73.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
                              
![在这里插入图片描述](https://img-blog.csdnimg.cn/b9bcefd41c2342d99935c6070e6fbaa7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
                           
![在这里插入图片描述](https://img-blog.csdnimg.cn/69b7a5687482439a8025ba693cb33558.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
                                 
对于右节点，只有2，3两个样本，那么根据下表我们选择年龄30进行划分：                            
![在这里插入图片描述](https://img-blog.csdnimg.cn/7d63133f988e46648ccaa30269007385.png)
                            
![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-DgdpHG6v-1638484890170)(./13.png)\]](https://img-blog.csdnimg.cn/cce3670868d54729933a6769d75f6a29.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
                             

因此根据$\Upsilon_{j 1}=\underbrace{\arg \min }_{\Upsilon} \sum_{x_{i} \in R_{j 1}} L\left(y_{i}, f_{0}\left(x_{i}\right)+\Upsilon\right)$：                                
$$
\begin{array}{l}
\left(x_{0} \in R_{11}\right), \quad \Upsilon_{11}=-0.375 \\
\left(x_{1} \in R_{21}\right), \quad \Upsilon_{21}=-0.175 \\
\left(x_{2} \in R_{31}\right), \quad \Upsilon_{31}=0.225 \\
\left(x_{3} \in R_{41}\right), \quad \Upsilon_{41}=0.325
\end{array}
$$                                      
这里其实和上面初始化学习器是一个道理，平方损失，求导，令导数等于零，化简之后得到每个叶子节点的参数$\Upsilon$,其实就是标签值的均值。
最后得到五轮迭代：                         
![在这里插入图片描述](https://img-blog.csdnimg.cn/ba161ee53e4b487fa7509dca014222c8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
                               
最后的强学习器为：$f(x)=f_{5}(x)=f_{0}(x)+\sum_{m=1}^{5} \sum_{j=1}^{4} \Upsilon_{j m} I\left(x \in R_{j m}\right)$。                           
其中：
$$
\begin{array}{ll}
f_{0}(x)=1.475 & f_{2}(x)=0.0205 \\
f_{3}(x)=0.1823 & f_{4}(x)=0.1640 \\
f_{5}(x)=0.1476
\end{array}
$$                                
预测结果为：                       
$$
f(x)=1.475+0.1 *(0.2250+0.2025+0.1823+0.164+0.1476)=1.56714
$$                                      
为什么要用学习率呢？这是Shrinkage的思想，如果每次都全部加上（学习率为1）很容易一步学到位导致过拟合。    

### 4.3 GBDT代码示例

```python
下面我们来使用sklearn来使用GBDT
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

'''
GradientBoostingRegressor参数解释：
loss：{‘ls’, ‘lad’, ‘huber’, ‘quantile’}, default=’ls’：‘ls’ 指最小二乘回归. ‘lad’ (最小绝对偏差) 是仅基于输入变量的顺序信息的高度鲁棒的损失函数。. ‘huber’ 是两者的结合. ‘quantile’允许分位数回归（用于alpha指定分位数）
learning_rate：学习率缩小了每棵树的贡献learning_rate。在learning_rate和n_estimators之间需要权衡。
n_estimators：要执行的提升次数。
subsample：用于拟合各个基础学习者的样本比例。如果小于1.0，则将导致随机梯度增强。subsample与参数n_estimators。选择会导致方差减少和偏差增加。subsample < 1.0
criterion：{'friedman_mse'，'mse'，'mae'}，默认='friedman_mse'：“ mse”是均方误差，“ mae”是平均绝对误差。默认值“ friedman_mse”通常是最好的，因为在某些情况下它可以提供更好的近似值。
min_samples_split：拆分内部节点所需的最少样本数
min_samples_leaf：在叶节点处需要的最小样本数。
min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等。
max_depth：各个回归模型的最大深度。最大深度限制了树中节点的数量。调整此参数以获得最佳性能；最佳值取决于输入变量的相互作用。
min_impurity_decrease：如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。
min_impurity_split：提前停止树木生长的阈值。如果节点的杂质高于阈值，则该节点将分裂
max_features{‘auto’, ‘sqrt’, ‘log2’}，int或float：寻找最佳分割时要考虑的功能数量：

如果为int，则max_features在每个分割处考虑特征。

如果为float，max_features则为小数，并 在每次拆分时考虑要素。int(max_features * n_features)

如果“auto”，则max_features=n_features。

如果是“ sqrt”，则max_features=sqrt(n_features)。

如果为“ log2”，则为max_features=log2(n_features)。

如果没有，则max_features=n_features。
'''

X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
mean_squared_error(y_test, est.predict(X_test))

5.009154859960321
```

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
reg.score(X_test, y_test)

0.43848663277068134
```
GradientBoostingRegressor与GradientBoostingClassifier函数的各个参数的意思！参考文档：
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html?highlight=gra#sklearn.ensemble.GradientBoostingClassifier


