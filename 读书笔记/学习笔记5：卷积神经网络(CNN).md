@[toc]
## 一、CNN模型原理
### 1.1 图像
- 图像具有平移不变性和旋转不变性。即对图像的平移或者轻微旋转不改变其类别。图像可以用像素点来表示，存储为一个三维矩阵（长×宽×channels）
- 黑白图片channels=1，即每个像素点只有灰度值。彩色图像channels=3，每个像素点由RGB三原色组成，对应一个三维向量，值域[0，255]。一般0表示白色，255表示黑色
### 1.2 DNN图像分类的问题
如果直接将图像根据各像素点的向量作为图片特征输入模型，例如LR、SVM、DNN等模型进行分类，理论上可行，但是面临以下问题：
- 图像的平移旋转、手写数字笔迹的变化等，会造成输入图像特征矩阵的剧烈变化，影响分类结果。即不抗平移旋转等。
- 一般图像像素很高，如果直接DNN这样全连接处理，计算量太大，耗时太长；参数太多需要大量训练样本
### 1.3 卷积原理
- 人类认识图片的原理：不纠结图像每个位置具体的像素值，重点是每个区域像素点组成的几何形状，以及几何形状的相对位置和搭配。也就是图像中更抽象的轮廓信息。
- 模型需要对平移、旋转、笔迹变化等不敏感且参数较少。

卷积原理：
-  <font color='red'>CNN通过卷积核对图像的各个子区域进行特征提取，而不是直接从像素上提取特征。子区域称为感受野。

- 卷积运算：图片感受野的像素值与卷积核的像素值进行按位相乘后求和，加上偏置之后过一个激活函数（一般是Relu）得到特征图Feature Map。

- <font color='red'>特征图：卷积结果不再是像素值，而是感受野形状和卷积核的匹配程度，称之为特征图Feature Map。卷积后都是输出特定形状的强度值，与卷积核形状差异过大的感受野输出为0（经过Relu激活），所以卷积核也叫滤波器Filter。

卷积的特点：
- 使用一个多通道卷积核对多通道图像卷积，结果仍是单通道图像。要想保持多通道结果，就得使用多个卷积核。同一个通道的卷积结果是一类特征（同一个卷积核计算的结果）。

- 卷积核的参数为待学习参数，可以通过模型训练自动得到。
- 图像识别中有很多复杂的识别任务，如果每个图像对应一个卷积核，那么卷积核会很大，参数过多。另外复杂图像形状各异，但是基本元素种类差不多。所以CNN使用多个不同尺寸的卷积核进行基本形状的提取
- 假设图像大小为$N*N$矩阵，卷积核的尺寸为$K*K$矩阵，图像边缘像素填充:P，卷积的步伐为S，那么经过一层这样的<font color='red'>卷积后出来的图像为：W=(N-K+2P)/S+1。当步幅为1，padding=1时，卷积后图像尺寸不变。
### 1.4 池化原理
<font color='red'>池化层的作用：缩减图像尺寸；克服图像图像平移、轻微旋转的影响；间接增大后续卷积的感受野；降低运算量和参数量
- 消除相邻感受野的信息冗余现象（相邻感受野形状差异不大）
- 池化操作对子区域内的轻微改变不敏感，所以可以克服图像平移。轻微旋转的影响
- 缩减特征图，增大后续卷积操作的感受野，以便后续提取更宏观的特征。（比如经过2×2池化，池化后图像每个像素对应于原图像2×2的感受野，此时在用3×3卷积，那么卷积后每个像素对应于6×6的感受野）
- 降低运算量和参数量

池化的特点：
- 只需要设定池化层大小和池化标准（最大池化或均值池化、中位数池化），就可以进行池化计算，没有参数需要学习
- 最大池化提取特征能力较强，但是容易被噪声干扰。均值池化相对稳定，对噪声不敏感
- 池化在各个通道上独立进行
- 池化步长一般会参考感受野尺寸。当二者相等时，池化时没有交集

池化技巧：
&#8195;&#8195;如果有20个卷积层，max池化和ave池化混用，怎么安排？应该是前面的层用最大池化，尽可能的提取特征；后面层用ave池化，减少尺寸抗平移。
&#8195;&#8195;因为<font color='deeppink'>一开始就用平均池化，把特征平均掉了，就很难恢复了。所以是先提取特征再去噪。训练样本足够多的时候，不太容易被噪声所影响，直接用max池化。（足够的样本可以平均噪声）

而在不同场景用的池化操作也不一样：
1. 人脸识别：公司打卡系统 。对特征要求高，需要把每个人的五官特点提取出来（max pool）
2. 人脸检测：画面是否有人 。要求低，大概轮廓出来即可（ave pool）

### 1.5 Flatten
- 卷积-池化输出是一个多通道的而为特征，而最后softmax分类，softmax只能作用于一个向量。所以需要对卷积0池化结果进行拉平操作，也就是Flatten。
- Flatten没有参数，例如一个7×7×10的矩阵，会被拉平成490维向量。
![在这里插入图片描述](https://img-blog.csdnimg.cn/a5ef3ebb7f704a0f8fb4dda4dc7c4d7e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)
### 1.6 卷积池化总结
CNN使用不同卷积核提取图像中特定的形状特征，配合池化操作进一步缩减图像尺寸，稳定特征。对低级特征进行卷积计算，得到相对高级的特征。所以多层卷积-池化，层层堆叠可以提取复杂特征。
需要注意的是：
- 一个CNN模型只能处理一种尺寸的图像，所以实际中需要将输入图像全部处理成同一尺寸
- 为了防止数值溢出和激活函数饱和造成的梯度消失或者梯度爆炸，输入图像像素值会归一化至[0,1]
- CNN中每个通道代表同一类特征（同一个卷积核计算的结果），所以可以<font color='deeppink'>对同一个通道的数值进行批归一化。分别计算n个通道上的n组均值和方差。
- 对于28×28×1的图像，如果全连接并保持图像尺寸不变，则参数量为28×28×1×28×28×1=614656个。如果进行3×3卷积并保持尺寸不变，参数量为28×28×1×3×3。可以理解为<font color='deeppink'>隐藏层每个神经元只与输入层9个神经元相连，其它连接都被剪枝，各个位置参数共享</font>（隐藏层还是576神经元，但是前后层都是9*9相连）</font>所以<font color='red'>CNN是通过权重共享和局部感受野（剪枝）对DNN全连接进行简化。
## 二、卷积池化计算
一个常见的CNN例子如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/9b5798fb03ed4920be2a6222473f626d.png)
### 2.1. 初识卷积
微积分中卷积的表达式为：

$S

(t) = \int x(t-a)w(a) da

$$

　　　　离散形式是：$$s(t) = \sum\limits_ax(t-a)w(a)$$

　　　　这个式子如果用矩阵表示可以为：$$s(t)=(X*W)(t)$$

　　　　其中星号表示卷积。

　　　　如果是二维的卷积，则表示式为：$$s(i,j)=(X*W)(i,j) = \sum\limits_m \sum\limits_n x(i-m,j-n) w(m,n)$$

　　　　在CNN中，虽然我们也是说卷积，但是我们的卷积公式和严格意义数学中的定义稍有不同,比如对于二维的卷积，定义为：$$s(i,j)=(X*W)(i,j) = \sum\limits_m \sum\limits_n x(i+m,j+n) w(m,n)$$

　　　　这个式子虽然从数学上讲不是严格意义上的卷积，但是大牛们都这么叫了，那么我们也跟着这么叫了。后面讲的CNN的卷积都是指的上面的最后一个式子。

　　　　其中，我们叫W为我们的卷积核，而X则为我们的输入。如果X是一个二维输入的矩阵，而W也是一个二维的矩阵。但是如果X是多维张量，那么W也是一个多维的张量。
### 2.2. CNN中的卷积层
图像卷积:对输入的图像的不同局部的矩阵和卷积核矩阵各个位置的元素相乘，然后相加得到。

　　　　举个例子如下:
- 输入二维的3x4的矩阵
- 卷积核是一个2x2的矩阵
- 卷积步长为1（一次移动一个像素来卷积）

1. 首先我们对输入的左上角2x2局部和卷积核卷积，即各个位置的元素相乘再相加，得到的输出矩阵S的$S_{00}$的元素，值为$aw+bx+ey+fz$。
2. 我们将输入的局部向右平移一个像素，现在是(b,c,f,g)四个元素构成的矩阵和卷积核来卷积，这样我们得到了输出矩阵S的$S_{01}$的元素
3. 同样的方法，我们可以得到输出矩阵S的$S_{02}，S_{10}，S_{11}， S_{12}$的元素。
图像卷积，回想我们的上一节的卷积公式，其实就是对输入的图像的不同局部的矩阵和卷积核矩阵各个位置的元素相乘，然后相加得到。

#### 2.2.1 二维卷积：
举例如下:
- 输入二维的3x4的矩阵，卷积核是一个2x2的矩阵。步幅S=1。首先我们对输入的左上角2x2局部和卷积核卷积，即各个位置的元素相乘再相加，得到的输出矩阵S的$S_{00}$的元素，值为$aw+bx+ey+fz$。
- 接着我们将输入的局部向右平移一个像素，现在是(b,c,f,g)四个元素构成的矩阵和卷积核来卷积，这样我们得到了输出矩阵S的$S_{01}$的元素
- 同样的方法，我们可以得到输出矩阵S的$S_{02}，S_{10}，S_{11}， S_{12}$的元素。
![在这里插入图片描述](https://img-blog.csdnimg.cn/07c3043832834aeeb273b4a6dc3313b9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_12,color_FFFFFF,t_70,g_se,x_16)
最终我们得到卷积输出的矩阵为一个2x3的矩阵S。

假设图像大小为$N*N$矩阵，卷积核的尺寸为$K*K$矩阵，图像边缘像素填充:P，卷积的步伐为S，那么经过一层这样的<font color='red'>卷积后出来的图像为：W=(N-K+2P)/S+1
#### 2.2.2 三维卷积
 输入是多维的情况，在斯坦福大学的cs231n的课程上，有一个[动态的例子](https://cs231n.github.io/convolutional-networks/)


- 输入3个7x7的矩阵。（原输入是3个5x5的矩阵，padding=1）
- 卷积核：
	- <font color='red'>两个卷积核，卷积核的个数是即模板的个数。
	- 卷积核也是三维，且<font color='red'>最后一维和输入维数第三维channel数一样。</font>卷积核W0的单个子矩阵维度为3x3，加上最后一维为3，最终是一个3x3x3的张量。
	- 步幅为2

- 张量的卷积：两个张量的3个子矩阵卷积后，再把卷积的结果相加后再加上偏置。

每个卷积核卷积的结果是一个3x3的矩阵，卷积的结果是一个3x3x2的张量。把上面的卷积过程用数学公式表达出来就是：$$s(i,j)=(X*W)(i,j) + b = \sum\limits_{k=1}^{n\_in}(X_k*W_k)(i,j) +b$$
　　　　其中，$n\_in$为输入矩阵的个数，或者是张量的最后一维的维数。$X_k$代表第k个输入矩阵（channel个）。$W_k$代表卷积核的第k个子卷积核矩阵。$s(i,j)$即卷积核$W$对应的输出矩阵的对应位置元素的值。
　　　　
- 激活:对于卷积后的输出，一般会通过ReLU激活函数，将输出的张量中的小于0的位置对应的元素值都变为0。
#### 2.2.3 卷积计算公式
&#8195;&#8195;对图像的每个像素进行编号，用$x_{i,j}$表示图像的第行第列元素；用$w_{m,n}$表示<font color='red'>卷积核filter</font>第m行第n列权重，用$w_b$表示filter的偏置项；用$a_{i,j}$表示<font color='red'>特征图Feature Map</font>的第i行第j列元素；用$f$表示激活函数(这个例子选择relu函数作为激活函数)。使用下列公式计算卷积：
$$a_{i,j}=f(\sum_{m=0}^{2}\sum_{n=0}^{2}w_{m,n}x_{i+m,j+n}+w_{b})$$
&#8195;&#8195;如果卷积前的图像深度为D，那么相应的filter的深度也必须为D。我们扩展一下上式，得到了深度大于1的卷积计算公式：
$$a_{d,i,j}=f(\sum_{d=0}^{D-1}\sum_{m=0}^{F-1}\sum_{n=0}^{F-1}w_{d,m,n}x_{d,i+m,j+n}+w_{b})$$
&#8195;&#8195; W2是卷积后Feature Map的宽度；W1是卷积前图像的宽度；F是filter的宽度；P是Zero Padding数量。D是深度（卷积核个数）；F是卷积核filter的矩阵维数；$w_{d,m,n}$表示filter的第d层第m行第n列权重；$a_{d,i,j}$表示Feature Map图像的第d层第i行第j列像素。
&#8195;&#8195;每个卷积层可以有多个卷积核filter。每个filter和原始图像进行卷积后，都可以得到一个Feature Map。因此，卷积后Feature Map的深度(个数)和卷积层的filter个数是相同的。

### 2.3 CNN中的池化层
- <font color='red'> 池化，就是对输入张量的各个子矩阵进行压缩，将输入子矩阵的每nxn个元素变成一个元素，所以需要一个池化标准。
- 常见的池化标准有2个，MAX或者是Average。即取对应区域的最大值或者平均值作为池化后的元素值。

2x2最大池化，步幅为2时，池化操作如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/821b9e32e4e842519dc6412f1eb8f2c8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_11,color_FFFFFF,t_70,g_se,x_16)
###  2.4 CNN前向传播算法
输入：1个图片样本，CNN模型的层数L和所有隐藏层的类型，对于卷积层，要定义卷积核的大小K，卷积核子矩阵的维度F，填充大小P，步幅S。对于池化层，要定义池化区域大小k和池化标准（MAX或Average），对于全连接层，要定义全连接层的激活函数（输出层除外）和各层的神经元个数。

　　　　输出：CNN模型的输出$a^L$

　　　　&#8195;&#8195;1) 根据输入层的填充大小P，填充原始图片的边缘，得到输入张量$a^1$。

　　　　&#8195;&#8195;2）初始化所有隐藏层的参数$W,b$　　

　　　　&#8195;&#8195;3）for $l$=2 to $L-1$（层数$l$）:

&#8195;&#8195;&#8195;&#8195;a) 如果第$l$层是<font color='deeppink'>卷积层</font>,则输出为（*表示卷积，而不是矩阵乘法）
$$ a^l= ReLU(z^l) = ReLU(a^{l-1}*W^l +b^l)$$　　
&#8195;&#8195;&#8195;&#8195;（这里要定义卷积核个数，卷积核中每个子矩阵大小，填充padding（以下简称P）和填充padding（以下简称P））
 　　　　b) 如果第$l$层是<font color='deeppink'>池化层</font>,则输出为：$$a^l= pool(a^{l-1})$$
 　　　　 需要定义池化大小和池化标准,池化层没有激活函数

　　　　　　&#8195;&#8195;&#8195;&#8195;c) 如果第$l$层是<font color='deeppink'>全连接层</font>,则输出为：$$ a^l= \sigma(z^l) = \sigma(W^la^{l-1} +b^l)$$

　　　　&#8195;&#8195;4)对于<font color='deeppink'>输出层第L层</font>: $$ a^L= softmax(z^L) = softmax(W^La^{L-1} +b^L)$$
### 2.5 CNN反向传播算法
>参考[《卷积神经网络(CNN)》](https://blog.csdn.net/m0_64375823/article/details/121584188)

## 三、深入卷积层
### 3.1 1×1卷积
<font color='red'> 1×1卷积作用是改变通道数，降低运算量和参数量。同时增加一次非线性变化，提升网络拟合能力。
- 对于一个$28×28×f_{1}$的图像，进行$f_{2}个1×1$卷积操作，得到$28×28×f_{2}$的图像，且参数量仅有$f_{1}×f_{2}$（忽略偏置）。
	- $f_{1}>f_{2}$时起到降维的作用，降低其它卷积操作的运算量。但是降维太厉害会丢失很多信息。
	-  $f_{1}<f_{2}$时起到升维作用（增加通道数），可以让后续卷积层提取更加丰富的特征
- 增加一次非线性变化，提升网络拟合能力。

所以可以先用1×1卷积改变通道数，再进行后续卷积操作，这个是Depth wise提出的。
### 3.2 VGGnet：小尺寸卷积效果好
卷积的尺寸决定卷积的视野，越大则提取的特征越宏观。但是大尺寸卷积，参数量和运算量都很大，而<font color='deeppink'>多个小尺寸卷积可以达到相同的效果，且参数量更小。还可以多次进行激活操作，提高拟合能力。</font>例如：
一个5×5卷积参数量25，可以替换成两个3×3卷积。，参数量为18。每个3×3卷积可以替换成3×1卷积加1×3卷积，参数量为12。
### 3.3 inception宽度卷积核和GoogLeNet
![在这里插入图片描述](https://img-blog.csdnimg.cn/7517f9b7b00f497cab5f4c57eca04345.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)

实际CNN识别中，会遇到识别物体尺寸不一的情况。不同尺寸的物体需要不同尺寸的卷积核来提取特征。如果增加网络深度来处理，会造成：
- 训练集不够大，则过拟合
- 深层网络容易梯度消失，模型难以优化
- 简单堆叠较大的卷积层浪费计算资源

<font color='deeppink'>为了使卷积层适应不同的物体尺寸，一般会在同一层网络中并列使用多种尺寸的卷积，以定位不同尺寸的物体。此时增加了网络宽度，而不会增加其深度。

2016年google的inception模型首先提出，结构如下：
>图片参考：[《深入解读GoogLeNet网络结构（附代码实现）》](https://blog.csdn.net/qq_37555071/article/details/108214680)
![在这里插入图片描述](https://img-blog.csdnimg.cn/827565c5ccd84d84a06236424462e4c5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_17,color_FFFFFF,t_70,g_se,x_16)


- 上图使用三种尺寸卷积核进行特征提取，并列了一个最大池化（抗平移），池化步长为1，不会改变输入尺寸。最后多个同尺寸的卷积结果进行级联，得到新的图像。（并列在一起，增加了通道数）
- 多个卷积核级联造成通道数过多，所以可以在卷积前、3×3池化后分别进行1×1卷积进行降维，同时提高网络非线性程度。
- 最终输出和输入图像尺寸相同，但是通道数可以不一样。

多个inception堆叠就是GoogLeNet:
![在这里插入图片描述](https://img-blog.csdnimg.cn/85f04938e7a94fd5a31b34f02f24fe54.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)

- 第一个红色框住的模块叫stem（根源），就是传统的卷积。
- 后面九个模块都是inception，再过一个全连接层，最后过softmax进行分类。

这样会有一个问题：	<font color='deeppink'>网络太深造成梯度消失，前面的层基本就没有什么信息了（梯度消失学不到）。所以中间层引入两个辅助分类器，并配以辅助损失函数。防止前层网络信息丢失。</font >具体地：

- 三个黄色和椭圆模块是做三次分类。即在3.6.9层inception时输出做分类。
- 三个分类器的任务完全一样，$loss=w_{1}loss_{1}+ w_{2}loss_{2}
+w_{3}loss_{3}$。$w_{3}$的系数应该高一些，具体权重可以查。辅助分类器只用来训练，不用于推断
- 训练时三个分类器一起训练，使用的时候只用最后一层。最后一个inception使用全局平均池化


GoogleNet知识点：
1. inception
2. 深层网络可以从中间抽取loss来训练，防止过拟合。
3. 启发：网络太深，涉及梯度消失时，都可以这样搞：中间层可以抽出loss来一起学习。（shortcut也可以，一个道理，可能还好一点，比较方便）。

GoogleNet：

![在这里插入图片描述](https://img-blog.csdnimg.cn/bc3f9203cabc442da8688cba3f12b10d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_16,color_FFFFFF,t_70,g_se,x_16)
### 3.4 Depth wise和Pointwise降低运算量
- 传统卷积：一个卷积核卷积图像的所有通道，参数过多，运算量大。
![在这里插入图片描述](https://img-blog.csdnimg.cn/b79e31d49ba14ab69f17f0fc9817b01a.png)
运算量（忽略偏置）：$28*28*128*3*3*256=231211008$
参数量（忽略偏置）：$128*3*3*256=294912$
- Depth wise卷积：一个卷积核只卷积一个通道。输出图像通道数和输入时不变。缺点是每个通道独立卷积运算，没有利用同一位置上不同通道的信息
- Pointwise卷积：使用多个1×1标准卷积，将Depth wise卷积结果的各通道特征加权求和，得到新的特征图
![在这里插入图片描述](https://img-blog.csdnimg.cn/dac8bfddaf3645e8b109cfa7adad7214.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_17,color_FFFFFF,t_70,g_se,x_16)
运算量（忽略偏置）：$28*28*128*3*3+28*28*128*256=26593280$
参数量（忽略偏置）：$3*3*128+128*256=33920$
### 3.5 SENet、CBAM特征通道加权卷积
#### 3.5.1 SENet
>可参考[《CNN卷积神经网络之SENet及代码》](https://blog.csdn.net/qq_41917697/article/details/114100031)
SENet：卷积操作中，每个通道对应一类特征。而不同特征对最终任务结果贡献是不一样的，所以考虑调整各通道的权重。
1. SE模块，对各通道中所有数值进行全局平均，此操作称为Squeeze。比如28×28×128的图像，操作后得到128×1的向量。
2. 此向量输入全连接网络，经过sigmoid输出128维向量，每个维度值域为（0,1），表示各个通道的权重
3. 在正常卷积中改为各通道加权求和，得到最终结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/323821f98d6f49c1ac526604c3e5090b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)

- Squeeze建立channel间的依赖关系；Excitation重新校准特征。二者结合强调有用特征抑制无用特征
- 能有效提升模型性能，提高准确率。几乎可以无脑添加到backbone中。根据论文，SE block应该加在Inception block之后，ResNet网络应该加在shortcut之前，将前后对应的通道数对应上即可
#### 3.5.2 CBAM
>参考[《CBAM重点干货和流程详解及Pytorch实现》](https://blog.csdn.net/qq_36584673/article/details/116088055)
>
除了通道权重，CBAM还考虑空间权重，即：图像中心区域比周围区域更重要，由此设置不同位置的空间权重。CBAM将空间注意力和通道注意力结合起来。

Channel attention module：
![在这里插入图片描述](https://img-blog.csdnimg.cn/cad6ce18b4b943ed8dfb452acdd1d01b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)
- 输入特征图F，经过两个并行的最大值池化和平均池化将C×H×W的特征图变成C×1×1的大小
- 经过一个共享神经网络Shared MLP(Conv/Linear，ReLU，Conv/Linear)，压缩通道数C/r (reduction=16)，再扩张回C，得到两个激活后的结果。
- 最后将二者相加再接一个sigmoid得到权重channel_out，再加权求和。

<font color='deeppink'>此步骤与SENet不同之处是加了一个并行的最大值池化，提取到的高层特征更全面，更丰富。

Channel attention module：

将上一步得到的结果通过最大值池化和平均池化分成两个大小为H×W×1的张量，然后通过Concat操作将二者堆叠在一起(C为2)，再通过卷积操作将通道变为1同时保证H和W不变，经过一个sigmoid得到spatial_out，最后spatial_out乘上一步的输入变回C×H×W，完成空间注意力操作

总结：
- 实验表明：通道注意力在空间注意力之前效果更好
- 加入CBAM模块不一定会给网络带来性能上的提升，受自身网络还有数据等其他因素影响，甚至会下降。如果网络模型的泛化能力已经很强，而你的数据集不是benchmarks而是自己采集的数据集的话，不建议加入CBAM模块。要根据自己的数据、网络等因素综合考量。
### 3.6 inception几个改进版
google对inception进行改造，出现了inception1→inception2→inception3→Xception→inception4→inception ResNetV1→inception →ResNetV2。
#### 3.6.1 Inception2
![在这里插入图片描述](https://img-blog.csdnimg.cn/2f0616b8df2448a1bc6da9c944ee866b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)
<font color='deeppink'>变种A是基础版的5×5改成两个3×3，B是3×3拆成两个，C是拆成的两个并联。

对卷积核进行了几种改造。但是设计思想都是：
1.大量应用1×1卷积核和n×1卷积核（1×3和3×1）
2.大量应用小卷积核（没有超过3乘3的）
3.并联卷积
#### 3.6.2 Inception3
最大贡献：标签平滑防止过拟合

- 对于逻辑回归来说，单个样本$loss=-ylogy’-(1-y)log(1-y’)$。y’∈（0,1）是一个闭区间,预测值y’只能无限逼近0和1，但是永远取不到0或1。即单个样本没有极小值。
- 这样在拟合的时候随着梯度下降，y’不断向0或1逼近，w会不断增大。而如果标签y=1做平滑改成y=0.97，y’就可以取到这个值，w就不会无限增大，所以避免了过拟合。
- 也可以看做对标签适当注入噪声防止过拟合。（LR可以看做二分类的softmax，所以此处也适用）
- 加正则项主要是让模型在测试集上的效果尽可能和训练集效果一样好，标签平滑让模本本身有一个好的性能（防止标签打错等噪声）。
#### 3.6.3 Xception、inception4
- Xception：3×3正常卷积变成Depth wise（上一节讲过）
- inception4是改变了stem
![在这里插入图片描述](https://img-blog.csdnimg.cn/6ea077128fb4406f94ac4a06d886a130.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)
#### 3.6.4 inception ResNetV1&2
![在这里插入图片描述](https://img-blog.csdnimg.cn/435581b3b8bc4bf381e1c7867620f024.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)
inception ResNetV1&2最主要思想就是与shortcut结合。inception模块的输出和模块输入直接按位相加。即对应channel对应位置的元素相加。这样就要求输出的channel和尺寸要和输出的一致。而一般不池化尺寸可以不变，channel数通过最后一层的1×1卷积核来调整。
	至于中间的细节，右侧的构造为啥是这样的，都是试验碰出来的，没必要纠结。


#### 3.6.5 Resnet&Renext
>参考[《CNN卷积神经网络之ResNeXt》](https://blog.csdn.net/qq_41917697/article/details/115905056)

Resnet是微软何凯明搞出来的（93年的人）。主要也是借鉴shortcut思想，因为网络太深必然会碰到梯度消失的问题。然后就是一堆小卷积核，每两层抄一次近道是试验出来的效果。抄近道就必须保持前后的channel数一致。
最重要的部分就是：
![在这里插入图片描述](https://img-blog.csdnimg.cn/0ec2991998ce4df88784a23e188d1423.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_10,color_FFFFFF,t_70,g_se,x_16)

然后Resnet又借鉴inception把网络都做宽了，就是Renext：

ResNeXt是ResNet和Inception的结合体，因此你会觉得与InceptionV4有些相似，但却更简洁，同时还提出了一个新的维度： cardinality （基数），在不加深或加宽网络增加参数复杂度的前提下提高准确率，还减少了超参数的数量。
![在这里插入图片描述](https://img-blog.csdnimg.cn/80147e3181fe476b9ab70650be2a8f72.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_17,color_FFFFFF,t_70,g_se,x_16)
进一步进行了等效转换的，采用了分组卷积的方法。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2f910437e1d542d791482b0ef794d1c2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA6K-75Lmm5LiN6KeJ5bey5pil5rex77yB,size_20,color_FFFFFF,t_70,g_se,x_16)




