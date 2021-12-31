@[toc]
>推荐文章[《PyTorch 学习笔记汇总（完结撒花）》](https://zhuanlan.zhihu.com/p/265394674)
## 一、基础介绍
### 1.1PyTorch 简介：
- Torch是一个有大量机器学习算法支持的科学计算框架，是一个与Numpy类似的张量（Tensor） 操作库，其特点是特别灵活，但因其采用了小众的编程语言是Lua，所以流行度不高。
- PyTorch是一个基于Torch的Python开源机器学习库，提供了两个高级功能： 
	- 具有强大的GPU加速的张量计算（如Numpy） 
	-  包含自动求导系统的深度神经网络
	- PyTorch，通过反向求导技术，可以让你零延迟地任意改变神经网络的行为，而且其实现速度快
	- 底层代码易于理解 +命令式体验 +自定义扩展
	- 缺点，PyTorch也不例外，对比TensorFlow，其全面性处于劣势。例如目前PyTorch还不支持快速傅里 叶、沿维翻转张量和检查无穷与非数值张量等
### 1.2 静态图和动态图
为了能够计算权重梯度和数据梯度,神经网络需记录运算的过程,并构建出计算图。
1. 静态图：tensorflow和caffe。先构建模型对应的静态图，再输入张量。执行引擎会根据输入的张量进行计算,最后输出深度学习模型的计算结果。
	- 静态图的前向和反向传播路径在计算前已经被构建,所以是已知的。计算图在实际发生计算之前已经存在
	- 执行引擎可以在计算之前对计算图进行优化,比如删除冗余的运算合并两个运算操作等
	- 执行效率较高：不用每次计算都重新构建计算图,减少了计算图构建的时间消耗
	- 不够灵活：因为静态计算图在构建完成之后不能修改，使用条件控制(比如循环和判断语句)会不大方便
	- 代码调试较慢：构建时只能检查静态参数，如输入输出形状。执行时的问题无法在构件图时预先排查 
	- ==计算图中直接集成了优化器==，求出权重张量梯度，直接执行优化器的计算图，更新权重的张量值
2. 动态图：在计算过程中逐步构建计算图。牺牲执行效率但是更灵活
	- 反向传播路径只有在构建完计算图时才能获得
	- 条件控制语句很简单
	- 调试方便：可以实时输出模型的中间张量
	- 优化器绑定在权重张量上：反向传播后，优化器根据绑定的梯度长量更新权重张量。
	- 强大的可扩展性。例如自由定制张量计算、CPU/GPU异构计算、并行计算环境、设置不同模型层的学习率等。

### 1.3 pytorch主要模块
下面介绍主要模块。具体都可以参考[官方文档](https://pytorch.org/docs/stable/nn.html)。
1. torch模块：包含<font color='red'>激活函数</font>和主要的张量操作
2. torch.Tensor模块：定义了张量的数据类型（整型、浮点型等）另外张量的某个类方法会返回新的张量，==如果方法后缀带下划线，就会修改张量本身==。比如Tensor.add是当前张量和别的张量做加法，返回新的张量。如果是ensor.add_就是将加和的张量结果赋值给当前张量。
3. torch.cuda:定义了CUDA运算相关的函数。如检查CUDA是否可用及序号，清除其缓存、设置GPU计算流stream等
4. torch.nn：神经网络模块化的核心，包括卷积神经网络nn.ConvNd和全连接层（线性层）nn.Linear等，以及一系列的<font color='red'>损失函数</font>。
5. torch,nn.functional:定义神经网络相关的函数，例如卷积函数、池化函数、log_softmax函数等部分激活函数。torch.nn模块一般会调用torch.nn.functional的函数。
6. torch.nn.init:权重初始化模块。包括均匀初始化torch.nn.init.uniform_和正态分布归一化torch.nn.init.normal_。（_表示直接修改原张量的数值并返回）
7. torch.optim：定义一系列优化器，如optim.SGD、optim.Adam、optim.AdamW等。以及学习率调度器torch.optim.lr_scheduler。并可以实现多种学习率衰减方法等。具体参考[官方教程](https://pytorch.org/docs/stable/optim.html)。
8. torch.autograd：自动微分算法模块。定义一系列自动微分函数，例如torch.autograd.backward反向传播函数和torch.autograd.grad求导函数（一个标量张量对另一个张量求导）。以及设置不求导部分。
9. torch.distributed：分布式计算模块。设定并行运算环境
10. torch.distributions：强化学习等需要的策略梯度法（概率采样计算图）  无法直接对离散采样结果求导，这个模块可以解决这个问题
11. torch.hub：提供一系列预训练模型给用户使用。torch.hub.list获取模型的checkpoint，torch.hub.load来加载对应模型。
12. torch.random：保存和设置随机数生成器。manual_seed设置随机数种子，initial_seed设置程序初始化种子。set_rng_state设置当前随机数生成器状态，get_rng_state获取前随机数生成器状态。设置统一的随机数种子，可以测试不同神经网络的表现，方便进行调试。
13. torch.jit：动态图转静态图，保存后被其他前端支持（C++等）。关联的还有torch.onnx（深度学习模型描述文件，用于和其它深度学习框架进行模型交换）
除此之外还有一些辅助模块：
-  torch.utils.benchmark：记录深度学习模型中各模块运行时间，通过优化运行时间，来优化模型性能
- torch.utils.checkpoint：以计算时间换空间，优化模型性能。因为反向传播时，需要保存中间数据，大大增加内存消耗。此模块可以记录中间数据计算过程，然后丢弃中间数据，用的时候再重新计算。这样可以提高batch_size，使模型性能和优化更稳定。
- torch.utils.data：主要是Dataset和DataLoader。
- torch.utils.tensorboard：pytorch对tensorboard的数据可视化支持工具。显示模型训练过程中的
损失函数和张量权重的直方图，以及中间输出的文本、视频等。方便调试程序。
## 二、 张量
pytorch提供专门的torch.Tensor类，根据张量的数据格式和存储设备（CPU/GPU）来存储张量。
Tensors 类似于 NumPy 的 ndarrays ，同时 Tensors 可以使用 GPU 进行计算。
详细的张量操作参考：[torch.Tensor](https://pytorch.org/docs/stable/tensors.html)、[张量创建和运算： torch](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)
### 2.1.张量的创建方式：
- python列表、ndarray数组转为张量
```python
torch.tensor([[1., -1.], [1., -1.]])#python列表转为张量，子列表长度必须一致
torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))#ndarray数组转为张量
x_np = torch.from_numpy(np_array)
```
- 利用函数创建张量
```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
```
- 常见的构造Tensor的函数：

|                                  函数 | 功能                   |
| ------------------------------------: | ---------------------- |
|                      Tensor(sizes) | 基础构造函数           |
|                        tensor(data) | 类似于np.array         |
|                        ones(sizes) | 全1                    |
|                       zeros(sizes) | 全0                    |
|                         eye(sizes) | 对角为1，其余为0       |
|                    arange(s,e,step) | 从s到e，步长为step     |
|                 linspace(s,e,steps) | 从s到e，均匀分成step份 |
|                  randn(sizes) |    标准正态分布                    |
| rand（size）|  [0,1)j均匀分布
| normal(mean,std)| 正态分布
| uniform(from,to) | /均匀分布 
| randint(a,b,(sizes)) |     从a到b形状为size的整数张量
|                         randperm(m) | 随机排列               |

- 创建类似形状的张量：

```python
t=torch.randn(3,3)
torch.zeros_like(t)#zeros还可以换成其它构造函数ones、randdeng
#如果t是整型，构造函数生成浮点型会报错
```

### 2.2 张量类型和维度
- 访问dtype属性可以查看张量的类型。shape属性可以查看张量的形状
```python
a=torch.tensor([[1., -1.], [1., -1.]])
print(a.dtype,a.type(),a.shape)

torch.float32 torch.FloatTensor torch.Size([2, 2])
```
- pytorch不同数据类型之间可以用to转换，或者.int()方法
```python
#浮点型转整型
torch.randn(3,3).to(torch.int)
torch.randn(3,3).int()
```
- 张量的维度
```python
t=torch.randn(3,4).to(torch.int)
t.nelement()#获取元素总数
t.ndimension()#获取张量维度
t.shape#张量形状
```
- 改变张量的维度可以用view方法，指定n-1维，最后一维写-1

```python
t.view(4,3)
t.view(-1,3)
t.view(12)#tensor([0, 0, 0, 0, -1, 1, 0, 2, 0, 2, 0, 0], dtype=torch.int32)
```
另外还有reshape和contiguous方法。
### 2.3 张量的存储设备
两个张量只有在同一设备上才可以运算（CPU或者同一个GPU）

```python
nvidia-smi#可以查看GPU的信息
!nvidia-smi#colab上命令是这个
torch.randn(3,3,device='cuda:0').device#在0号cuda上创建张量，查看张量存储设备
device(type='cuda', index=0)

torch.randn(3,3,device='cuda:0').cpu().device#cuda 0上的张量复制到CPU上
device(type='cpu')

torch.randn(3,3,device='cuda:0').cuda(1）
torch.randn(3,3,device='cuda:0').to('cuda:1')
```

```python
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 495.44       Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   47C    P0    28W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
### 2. 4 索引和切片
等同numpy的操作。如：

```python
t=torch.randn(3,4,5)
t[:,1:-1,1:3])
t>0#得到一个掩码矩阵
t[t>0]
```
<font color='red'>筛选出t中大于0的元素，最终得到一个一维向量
如果不想改变原张量的数值，可以先用clone得到张量的副本，再进行索引和切片的赋值操作。

### 2.5 函数运算和极值排序sort
所有运算符、操作符见文档：[《Creation Ops》](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops)
```python
t.mean()#对所有维度求均值
t.mean(0)#对第0维元素求均值
t.mean([0,1])#对0,1两维元素求均值
```
- argmax和argmin可以根据传入的维度，求的该维度极大极小值对应的序号。
- max和min会得到一个元组，包括极值位置和极值。
- sort默认从小到大排序。从大到小需要设置descending=True。需要传入排序的维度，返回排序后的张量和各元素在原始张量的位置

```python
t=torch.randint(1,100,(3,4))

tensor([[20, 95,  9, 94],
        [97, 61, 80, 67],
        [76, 66, 64, 65]])
        
t.max(0)

torch.return_types.max(
values=tensor([97, 95, 80, 94]),
indices=tensor([1, 0, 1, 0]))

t.argmax(0)
tensor([1, 0, 1, 0])

t.sort(-1,descending=True)
torch.return_types.sort(
values=tensor([[95, 94, 20,  9],
        [97, 80, 67, 61],
        [76, 66, 65, 64]]),
indices=tensor([[1, 3, 0, 2],
        [0, 2, 3, 1],
        [0, 1, 3, 2]]))
```
==函数后面加下划线是原地操作，改变被调用的张量的值==
### 2.6 矩阵乘法@和张量的缩并einsum
矩阵乘法可以用a.mm(b)或者torch.mm(a,b)或者a@b三种形式。
==还是a@b最好用==

```python
t=torch.randint(1,100,(3,4))
q=torch.randint(1,100,(4,3))
#三种写法都是得到3×3的矩阵
t.mm(q)
torch.mm(t,q)
t@q
tensor([[12151,  9911,  8876],
        [16752, 18098, 15618],
        [14844, 15675, 13614]])
```
- ==一个batch矩阵的乘法，需要用bmm函数==。即两个批次的矩阵乘法，是沿着批次方向分别对两个矩阵做乘法，最后将矩阵组合在一起。
- 比如一个b×m×k的矩阵和一个b×k×n的矩阵，做张量相乘，得到b×m×n的张量。

```python
a = torch.randn(2,3,4) # 随机产生张量
c = torch.randn(2,4,3)
a.bmm(c) # 批次矩阵乘法的结果
torch.bmm(a,c)
a@b
```
如果是3维以上张量的乘积，称为缩并。需要用到爱因斯坦求和约定。对应函数为torch.einsum。
### 2.7 张量的拼接和分割split
torch.stack:传入张量列表和维度，将张量沿此维度进行堆叠（新建一个维度来堆叠）
torch.cat:传入张量列表和维度，将张量沿此维度进行堆叠
两个都是拼接张量，torch.stack会新建一个维度来拼接，后者维度预先存在，沿着此维度堆叠就行。

```python
t1 = torch.randn(3,4) # 随机产生三个张量
t2 = torch.randn(3,4)
t3 = torch.randn(3,4)
 
torch.stack([t1,t2,t3], -1).shape# 沿着最后一个维度做堆叠，返回大小为3×4×3的张量
torch.Size([3, 4, 3])
-----------------------------------------------------------------------------
torch.cat([t1,t2,t3], -1).shape # 沿着最后一个维度做拼接，返回大小为3×14的张量
torch.Size([3, 12])
```

```python
torch.split(tensor, split_size_or_sections, dim=0)
```
torch.split函数，有三个参数。将张量沿着指定维度进行分割。
第二个参数可以是整数n或者列表list。前者表示这个维度等分成n份（最后一份可以是剩余的）。或者表示分成列表元素值来分割。

torch.chunk函数和slpit函数类似
```python
t = torch.randint(1, 10,(3,6)) # 随机产生一个3×6的张量
tensor([[8, 9, 5, 3, 6, 7],
        [1, 4, 2, 2, 7, 1],
        [5, 2, 5, 7, 2, 7]])
------------------------------------------------------------------------------        
t.split([1,2,3], -1) # 把张量沿着最后一个维度分割为三个张量
(tensor([[8],
         [1],
         [5]]),
 tensor([[9, 5],
         [4, 2],
         [2, 5]]),
 tensor([[3, 6, 7],
         [2, 7, 1],
         [7, 2, 7]]))
------------------------------------------------------------------------------         
t.split(3, -1) # 把张量沿着最后一个维度分割，分割大小为3，输出的张量大小均为3×3
(tensor([[8, 9, 5],
         [1, 4, 2],
         [5, 2, 5]]),
 tensor([[3, 6, 7],
         [2, 7, 1],
         [7, 2, 7]]))
         
t.chunk(3, -1) # 把张量沿着最后一个维度分割为三个张量，大小均为3×2
(tensor([[8, 9],
         [1, 4],
         [5, 2]]),
 tensor([[5, 3],
         [2, 2],
         [5, 7]]),
 tensor([[6, 7],
         [7, 1],
         [2, 7]]))
​
```
### 2.8 张量扩增(unsqueeze)、压缩(squeeze)和广播
- 张量可以任意扩增一个维度大小为1 的维度，数据不变。反过来这些维度大小为1的维度也可以压缩掉。

```python
t = torch.rand(3, 4) # 随机生成一个张量

t.unsqueeze(-1).shape # 扩增最后一个维度
torch.Size([3, 4, 1])

t.unsqueeze(-1).unsqueeze(1).shape  # 继续扩增一个维度
torch.Size([3, 1, 4, 1])

t = torch.rand(1,3,4,1) # 随机生成一个张量，有两个维度大小为1
t.squeeze().shape # 压缩所有大小为1的维度
torch.Size([3, 4])
```
- 两个不同维度的张量做四则运算，需要先把维度数目少的张量扩增到和另一个一致（unsqueeze方法），再进行运算。运算时，将扩增的维度进行复制，到最后维度一致再运算。

```python
t1 = torch.randn(3,4,5) 
t2 = torch.randn(3,5) 
t2 = t2.unsqueeze(1) # 张量2的形状变为3×1×5
print(t2)
tensor([[[ 0.7188, -1.1053, -0.1161, -2.2889, -0.8046]],

        [[ 0.1434, -2.8369, -1.5712,  1.1490,  0.7161]],

        [[-0.8259,  1.8744, -0.7918, -0.4208,  1.6935]]])
        
t3 = t1 + t2 #将t2沿着第二个维度复制4次，最后形状为(3,4,5) 
print(t3)
tensor([[[ 1.6212, -1.0232,  1.9735, -2.3579, -2.8416],
         [ 1.3389, -0.7377, -0.8453, -2.2385, -1.4370],
         [ 1.4433, -1.8982, -0.0669, -2.8503, -1.0240],
         [-0.0498, -2.2708,  0.4583, -0.3370, -2.7074]],

        [[ 1.7768, -2.4552,  0.3409, -0.7948,  1.9718],
         [ 0.1147, -3.2569, -1.4112,  1.3465,  0.2129],
         [ 0.8951, -3.5355, -0.3349,  1.4523,  0.2659],
         [ 0.6704, -2.3110, -1.1827,  0.8700,  2.9844]],

        [[-0.3561,  0.7850, -0.9848, -0.8666,  0.0758],
         [-0.1744,  1.3592, -1.7955, -0.0697,  3.8696],
         [-2.5559,  2.6479, -0.1718, -0.2446,  1.7351],
         [ 0.5748,  1.2866, -1.3801,  0.0290,  1.0740]]])
```

## 三. PyTorch 自动微分
### 3.1 autograd 自动求导和冻结参数
1. autograd 软件包为 Tensors 上的所有操作提供自动微分，是 PyTorch 中所有神经网络的核心。
2. 设置torch.Tensor 类的属性<font color='red'> .requires_grad = True</font>，则表示该张量会加入到计算图中，作为叶子节点参与计算，自动跟踪针对 tensor的所有操作。计算的中间结果都是requires_grad = True。
3. 每个张量都有一个<font color='red'> grad_fn方法</font>，保存创建该张量的运算的导数信息、计算图信息。
4. 调用<font color='red'> Tensor.backward() </font>传入最后一层的神经网络梯度。grad_fn方法的<font color='red'> next.functions属性</font>，包含连接该张量的其它张量的grad_fn。不断反向传播回溯中间张量计算节点，可以得到所有张量的梯度。该张量的梯度将累积到<font color='red'> .grad 属性 </font>中。如果Tensor 是标量，则backward()不需要指定任何参数。否则，需要指定一个gradient 参数来指定张量的形状。
5.  <font color='red'> with torch.no_grad() </font>: 包装的代码块部分，停止跟踪历史记录（和使用内存）。
6. 张量绑定的梯度在不清空的情况下会不断累积。可用来一次性求很多batch的累积梯度。

Tensor 和 Function 互相连接并构建一个非循环图，它保存整个完整的计算过程的历史信息。每个张量都有一个 .grad_fn 属性保存着创建了张量的 Function 的引用，（如果用户自己创建张量，则g rad_fn 是 None ）。

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
print(z, out)

tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)
tensor(27., grad_fn=<MeanBackward0>)
```

```python
out.backward()
print(x.grad)

tensor([[4.5000, 4.5000],
[4.5000, 4.5000]])
```
- 冻结参数

在 torch.nn 中，不计算梯度的参数通常称为冻结参数。 如果事先知道您不需要这些参数的梯度，则“冻结”模型的一部分很有用（通过减少自动梯度计算，这会带来一些表现优势）。

例如加载一个预训练的 resnet18 模型，并冻结所有参数，仅修改分类器层以对新标签进行预测。

```python
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# 冻结网络的所有参数
for param in model.parameters():
    param.requires_grad = False
```

假设我们要在具有 10 个标签的新数据集中微调模型。 在 resnet 中，分类器是最后一个线性层model.fc。 我们可以简单地将其替换为充当我们的分类器的新线性层（默认情况下未冻结）。

```python
model.fc = nn.Linear(512, 10)
# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
```
现在，除了model.fc的参数外，模型中的所有参数都将冻结。 计算梯度的唯一参数是model.fc的权重和偏差。（torch.no_grad()中的上下文管理器可以使用相同的排除功能。）
### 3.2 雅克比向量积
### 3.3 计算图
Autograd 在由函数对象组成的有向无环图（DAG）中记录张量、所有已执行的操作（以及由此产生的新张量）。 在此 DAG 中，叶子是输入张量，根是输出张量。 通过从根到叶跟踪此图，可以使用链式规则自动计算梯度。

1. 在正向传播中，Autograd 同时执行两项操作：
	- 根据张量和function计算结果张量
	- 在 DAG 中维护操作的梯度函数。

2. 当在 DAG 根目录上调用.backward()时，开始回传梯度，然后：
	- 从每个.grad_fn计算梯度，将它们累积在各自的张量的.grad属性中
	- 使用链式规则，一直传播到叶子张量。
 
下面是我们示例中 DAG 的直观表示。 在图中，箭头指向前进的方向。 节点代表正向传播中每个操作的反向函数。 蓝色的叶节点代表我们的叶张量a和b：
![在这里插入图片描述](https://img-blog.csdnimg.cn/9c6e2816a72e456bb5fb582f6817fc12.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_12,color_FFFFFF,t_70,g_se,x_16)

