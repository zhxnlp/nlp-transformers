@[toc]
[《PyTorch 学习笔记汇总（完结撒花）》](https://zhuanlan.zhihu.com/p/265394674)
## 一、nn.Module
### 1.1 nn.Module的调用
pytorch通过继承nn.Module类，定义子模块的实例化和前向传播，实现深度学习模型的搭建。其构建代码如下：
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, *kargs): # 定义类的初始化函数，...是用户的传入参数
        super(Model, self).__init__()#调用父类nn.Module的初始化方法
        ... # 根据传入的参数来定义子模块
    
    def forward(self, *kargs): # 定义前向计算的输入参数，...一般是张量或者其他的参数
        ret = ... # 根据传入的张量和子模块计算返回张量
        return ret
```
- __init\__方法初始化整个模型
-  super(Model, self).__init\__():调用父类nn.Module的初始化方法，初始化必要的变量和参数
- 定义前向传播模块
### 1.2 线性回归的实现

```python
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim#输入的特征数

        self.weight = nn.Parameter(torch.randn(ndim, 1)) # 定义权重
        self.bias = nn.Parameter(torch.randn(1)) # 定义偏置

    def forward(self, x):
        # 定义线性模型 y = Wx + b
        return x.mm(self.weight) + self.bias
```
- 定义权重和偏置self.weight和self.bias。采用标准正态分布torch.randn进行初始化。
- self.weight和self.bias是模型的参数，<font color='red'>使用nn.Parameter包装，表示将这些初始化的张量转换为模型的参数。只有参数才可以进行优化（被优化器访问到）

实例化方法如下：

```python
lm = LinearModel(5) # 定义线性回归模型，特征数为5
x = torch.randn(4, 5) # 定义随机输入，迷你批次大小为4
lm(x) # 得到每个迷你批次的输出
```
1. 使用model.named_parameters()或者model.parameters()获取模型参数的生成器。区别是前者包含参数名和对应的张量值，后者只含有张量值。
2. 优化器optimzer直接接受参数生成器作为参数，反向传播时根据梯度来优化生成器里的所有张量。
3. model.train()的作用是启用 Batch Normalization 和 Dropout。model.eval()的作用是不启用 Batch Normalization 和 Dropout。
4. named_buffers和buffers获取张量的缓存（不参与梯度传播但是会被更新的参数，例如BN的均值和方差）register_buffers可以加入这种张量
5. 使用apply递归地对子模块进行函数应用（可以是匿名函数lambda）

对于model.train()和model.eval()用法和区别进一步可以参考：[《Pytorch：model.train()和model.eval()用法和区别》](https://zhuanlan.zhihu.com/p/357075502?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn)

对于上面定义的线性模型来举例:
```python
lm.named_parameters() # 获取模型参数（带名字）的生成器
#<generator object Module.named_parameters at 0x00000279A1809510>
list(lm.named_parameters()) # 转换生成器为列表

[('weight',
  Parameter containing:
  tensor([[-1.0407],
          [ 0.0427],
          [ 0.4069],
          [-0.7064],
          [-1.1938]], requires_grad=True)),
 ('bias',
  Parameter containing:
  tensor([-0.7493], requires_grad=True))]
  
lm.parameters() # 获取模型参数（不带名字）的生成器
list(lm.parameters()) # 转换生成器为列表

[Parameter containing:
 tensor([[-1.0407],
         [ 0.0427],
         [ 0.4069],
         [-0.7064],
         [-1.1938]], requires_grad=True),
 Parameter containing:
 tensor([-0.7493], requires_grad=True)]

lm.cuda()#模型参数转到GPU上
list(lm.parameters()) # 转换生成器为列表
```

- model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是在训练中随机去除神经元，用一部分网络连接来训练更新参数。如果被删除的神经元（叉号）是唯一促成正确结果的神经元。一旦我们移除了被删除的神经元，它就迫使其他神经元训练和学习如何在没有被删除神经元的情况下保持准确。<font color='red'>这种dropout提高了最终测试的性能，但它对训练期间的性能产生了负面影响，因为网络是不全的

- 在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变（model.eval()时，框架会自动把BN和Dropout固定住，不会取平均，直接使用在训练阶段已经学出的mean和var值）

## 二、损失函数
pytorch损失函数有两种形式：
- <font color='red'> torch.nn.functional调用的函数形式</font>.传入神经网络预测值和目标值来计算损失函数
- <font color='red'> torch.nn库里面的模块形式</font>。新建模块的实例，调用模块化方法计算
最后输出的是标量，对一个批次的损失函数的值有两种归约方式：求和和求均值。

1. 回归问题一般调用torch.nn.MSEloss模块。使用默认参数创建实例，输出的是损失函数对一个batch的均值。

```python
import torch
import torch.nn as nn
mse = nn.MSELoss() # 初始化平方损失函数模块
#class torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
t1 = torch.randn(5, requires_grad=True) # 随机生成张量t1
tensor([ 0.6582,  0.0529, -0.9693, -0.9313, -0.7288], requires_grad=True)

t2 = torch.randn(5, requires_grad=True) # 随机生成张量t2
tensor([ 0.8095, -0.3384, -0.9510,  0.1581, -0.1863], requires_grad=True)

mse(t1, t2) # 计算张量t1和t2之间的平方损失函数
tensor(0.3315, grad_fn=<MseLossBackward>)
```
2. 二分类问题:
	- 使用 <font color='red'> torch.nn.BCELoss</font>二分类交叉熵损失函数。输出的是损失函数的均值。接受两个张量。前一个是正分类标签的概率值（预测值必须经过 <font color='deeppink'> nn.Sigmoid()</font>输出概率），后者是二分类标签的目标数据值（1是正分类）。两个都必须是浮点类型。
	-  <font color='red'> torch.nn.BCEWithLogitsLoss</font>：自动在损失函数内部实现sigmoid函数的功能，可以增加计算的稳定性。因为概率接近0或1的时候，二分类交叉熵损失函数接受的对数部分容易接近无穷大，造成数值不稳定。使用torch.nn.BCEWithLogitsLoss可以避免此种情况

```python
t1s = torch.sigmoid(t1)
t2 = torch.randint(0, 2, (5, )).float() # 随机生成0，1的整数序列，并转换为浮点数
bce=torch.nn.BCELoss()
(t1s, t2) # 计算二分类的交叉熵
bce_logits = nn.BCEWithLogitsLoss() # 使用交叉熵对数损失函数
bce_logits(t1, t2) # 计算二分类的交叉熵，可以发现和前面的结果一致
```

3. 多分类问题
	-  <font color='red'> torch.nn.NLLLoss</font>:负对数损失函数，计算之前预测值必须经过softmax函数输出概率值（<font color='deeppink'> torch.nn.functional.log_softmax或torch.nn.LogSoftmax(dim=dim)函数</font>）
	- <font color='red'>  torch.nn.CrossEntropyLoss</font>:交叉熵损失函数，内部已经整合softmax输出概率，不需要再另外对预测值进行softmax计算。
```python

N=10 # 定义分类数目
t1 = torch.randn(5, N, requires_grad=True) # 随机产生预测张量
t2 = torch.randint(0, N, (5, )) # 随机产生目标张量
t1s = torch.nn.functional.log_softmax(t1, -1) # 计算预测张量的LogSoftmax
nll = nn.NLLLoss() # 定义NLL损失函数
nll(t1s, t2) # 计算损失函数
ce = nn.CrossEntropyLoss() # 定义交叉熵损失函数
ce(t1, t2) # 计算损失函数，可以发现和NLL损失函数的结果一致
```
## 三、优化器
>完整文档参考：[《torch.optim 》](https://pytorch.org/docs/stable/optim.html)
#### 3.1.1  SGD优化器
 以波士顿房价问题举例，构建SGD优化器。第一个参数是模型的参数生成器（lm.parameters()调用），第二个参数是学习率。训练时通过 optim.step()进行优化计算。
```python
from sklearn.datasets import load_boston
boston = load_boston()

lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6) # 定义优化器
data = torch.tensor(boston["data"], requires_grad=True, dtype=torch.float32)
target = torch.tensor(boston["target"], dtype=torch.float32)

for step in range(10000):
    predict = lm(data) # 输出模型预测结果
    loss = criterion(predict, target) # 输出损失函数
    if step and step % 1000 == 0 :
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad() # 清零梯度
    loss.backward() # 反向传播
    optim.step()
```

```python
torch.optim.SGD(params,lr=<required parameter>,momentum=0,
    dampening=0,weight_decay=0,nesterov=False)

#momentum：动量因子
#dampening：动量抑制因子
#nesterov：设为True时使用nesterov动量
```
#### 3.1.2 Adagrad优化器

```python
torch.optim.Adagrad(
    params,lr=0.01,lr_decay=0,weight_decay=0,
    initial_accumulator_value=0,eps=1e-10)

#lr_decay：学习率衰减速率
#weight_decay：权重衰减
#initial_accumulator_value：梯度初始累加值
```

### 3.2 分层学习率
对不同参数指定不同的学习率：

```python
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```
这意味着model.base的参数将使用 的默认学习率1e-2， model.classifier的参数将使用 的学习率1e-3，0.9所有参数将使用动量 。
### 3.3 学习率调度器torch.optim.lr_scheduler

```python
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#没经过30的个迭代周期，学习率降为原来的0.1倍。每个epoch之后学习率都会衰减。
or epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

大多数学习率调度器都可以称为背靠背（也称为链式调度器）。结果是每个调度器都被一个接一个地应用于前一个调度器获得的学习率。

例子：

```python
model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler1.step()
    scheduler2.step()
```
## 四 、数据加载torch.utils.data
>本节也可以参考[《编写transformers的自定义pytorch训练循环（Dataset和DataLoader解析和实例代码）》](https://blog.csdn.net/qq_56591814/article/details/120467968)

4.1 DataLoader参数

PyTorch 数据加载实用程序的核心是torch.utils.data.DataLoader 类。它代表一个 Python 可迭代的数据集，支持：
- map类型和可迭代类型数据集
- 自定义数据加载顺序
- 自动batching
- 单进程和多进程数据加载

```python
train_loader = DataLoader(dataset=train_data, batch_size=6, shuffle=True ，num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False，num_workers=4)
```
下面看看dataloader代码：
```python
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None,*, prefetch_factor=2,persistent_workers=False)
    self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

```
- dataset:Dataset类，PyTorch已有的数据读取接口，决定数据从哪里读取及如何读取；
- batch_size：批大小；默认1
- num_works:是否多进程读取数据；默认0使用主进程来导入数据。大于0则多进程导入数据，加快数据导入速度
- shuffle：每个epoch是否乱序；默认False。输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。一般shuffle训练集即可。
- drop_last:当样本数不能被batchsize整除时，是否舍弃最后一批数据；
- collate_fn:将得到的数据整理成一个batch。默认设置是False。如果设置成True，系统会在返回前会将张量数据（Tensors）复制到CUDA内存中。
- batch_sampler，批量采样，和batch_size、shuffle等参数是互斥的，一般采用默认None。batch_sampler，但每次返回的是一批数据的索引（注意：不是数据），应该是每次输入网络的数据是随机采样模式，这样能使数据更具有独立性质。所以，它和一捆一捆按顺序输入，数据洗牌，数据采样，等模式是不兼容的。
- sampler，默认False。根据定义的策略从数据集中采样输入。如果定义采样规则，则洗牌（shuffle）设置必须为False。
- pin_memory，内存寄存，默认为False。在数据返回前，是否将数据复制到CUDA内存中。
- timeout，是用来设置数据读取的超时时间的，但超过这个时间还没读取到数据的话就会报错。
- worker_init_fn（数据类型 callable），子进程导入模式，默认为Noun。在数据导入前和步长结束后，根据工作子进程的ID逐个按顺序导入数据。

想用随机抽取的模式加载输入，可以设置 sampler 或 batch_sampler。如何定义抽样规则，可以看sampler.py脚本，或者这篇帖子：[《一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系》](https://blog.csdn.net/aiwanghuan5017/article/details/102147809)


### 4.2 两种数据集类型
DataLoader 构造函数最重要的参数是dataset，它表示要从中加载数据的数据集对象。PyTorch 支持两种不同类型的数据集：
- map-style datasets：映射类型数据集。每个数据有一个对应的索引，通过输入具体的索引，就可以得到对应的数据

其构造方法如下：

```python
class Dataset(object):
    def __getitem__(self, index):
        # index: 数据缩索引（整数，范围为0到数据数目-1）
        # ...
        # 返回数据张量

    def __len__(self):
        # 返回数据的数目
        # ...
```
主要重写两个方法：
- __getitem\__:python内置的操作符方法，对应索引操作符[]。通过输入整数索引，返回具体某一条数据。具体的内部逻辑根据数据集类型决定
- __len\__：返回数据总数

更具体的可以参考[《torch.utils.data.Dataset》](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

- iterable-style datasets：可迭代数据集：实现__iter__()协议的子类的实例。不需要__getitem\__和__len\__方法，其实类似python的迭代器
- 不同于映射，索引之间相互独立。多线程载入时，多线程独立分配索引。迭代中索引右前后关系，需要考虑如何分割数据。
- 这种类型的数据集特别适用于随机读取代价高昂甚至不可能的情况，以及批量大小取决于获取的数据的情况。
- 在调用iter(dataset)时可以返回从数据库、远程服务器甚至实时生成的日志中读取的数据流

```python
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
            assert end > start, \
"this example code only works with end >= start"
            self.start = start
            self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程数据载入
            iter_start = self.start
            iter_end = self.end
        else:  # 多进程，分割数据
        	   #根据不同工作进程序号worker_id，设置不同进程数据迭代器取值范围。保证不同进程获取不同的迭代器。
            per_worker = int(math.ceil((self.end - self.start) \
                            / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
```

更多详细信息，请参阅[IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)
### 4.3 数据加载顺序和 Sampler
- 对于iterable-style datasets，数据加载顺序完全由用户定义的 iterable 控制。这允许更容易地实现块读取和动态批量大小（例如，通过每次产生批量样本）。
- map 类型数据，torch.utils.data.Sampler 类用于指定数据加载中使用的索引/键的序列。它们表示数据集索引上的可迭代对象。例如，在随机梯度下降 (SGD) 的常见情况下，Sampler可以随机排列索引列表并一次产生一个，或者为小批量 SGD 产生少量索引。

将根据shufflea的参数自动构建顺序或混洗采样器DataLoader。或者，用户可以使用该sampler参数来指定一个自定义Sampler对象，该对象每次都会生成下一个要获取的索引/键。

一次Sampler生成批量索引列表的自定义可以作为batch_sampler参数传递。也可以通过batch_size和 drop_last参数启用自动批处理。
### 4.4 批处理和collate_fn
经由参数 batch_size，drop_last和batch_sampler，DataLoader支持批处理数据
当启用自动批处理时，每次都会使用数据样本列表调用 collat​​e_fn。预计将输入样本整理成一个批次，以便从数据加载器迭代器中产生。

例如，如果每个数据样本由一个 3 通道图像和一个完整的类标签组成，即数据集的每个元素返回一个元组 (image, class_index)，则默认 collat​​e_fn 将此类元组的列表整理成单个元组一个批处理图像张量和一个批处理类标签张量。特别是，默认 collat​​e_fn 具有以下属性：
- 它总是预先添加一个新维度作为批次维度。

- 它会自动将 NumPy 数组和 Python 数值转换为 PyTorch 张量。

- 它保留了数据结构，例如，如果每个样本是一个字典，它输出一个具有相同键集但批量张量作为值的字典（如果值不能转换为张量，则为列表）。列表 s、元组 s、namedtuple s 等也是如此。

用户可以使用自定义 collat​​e_fn 来实现自定义批处理，例如，沿着除第一个维度之外的维度进行整理，填充各种长度的序列，或添加对自定义数据类型的支持。


## 五、模型的保存和加载
### 5.1 模块、张量的序列化和反序列化
- PyTorch模块和张量本质是torch.nn.Module和torch.tensor类的实例。PyTorch自带了一系列方法， <font color='red'>可以将这些类的实例转化成字成串</font >。所以这些实例可以通过Python序列化方法进行序列化和反序列化。
- 张量的序列化： <font color='red'>本质上是把张量的信息，包括数据类型和存储位置、以及携带的数据，转换为字符串，然后使用Python自带的文件IO函数进行存储</font >。当然也是这个过程是可逆的。

```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=2)
torch.load(f, map_location=None, pickle_module=pickle, **pickle_load_args)
```

- torch.save参数
	1. pytorch中可以被序列化的对象，包括模型和张量
	2. 存储文件路径
	3. 序列化的库，默认pickle
	4. pickle协议，版本0-4
- torch.load函数
	1. 文件路径
	2. 张量存储位置的映射（默认CPU，也可以是GPU）
	3. pickle参数，和save时一样。

	如果模型保存在GPU中，而加载的当前计算机没有GPU，或者GPU设备号不对，可以使用map_location='cpu'。

PyTorch默认有两种模型保存方式：
- 保存模型的实例
- 保存模型的状态字典state_dict：state_dict包含模型所有参数名和对应的张量，通过调用load_state_dict可以获取当前模型的状态字典,载入模型参数。
### 5.2 state_dict保存模型参数
```python
torch.save(model.state_dict(), PATH)
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```
- 保存模型状态字典state_dict ：<font color='red'>只保存模型学习到的参数，与模块关联较小，即不依赖版本。
- PyTorch 中最常见的模型保存使‘.pt’或者是‘.pth’作为模型文件扩展名
- 在运行推理之前，务必调用 model.eval() 去设置 dropout 和 batch normalization 层为评
估模式。如果不这么做，可能导致 模型推断结果不一致

### 5.2 保存/加载完整模型
以 Python `pickle 模块的方式来保存模型。这种方法的缺点是：
- 序列化数据受 限于某种特殊的类而且需要确切的字典结构。当在其他项目使用或者重构之后，您的代码可能会以各种方式中断。
- PyTorch模块的实现依赖于具体的版本。所依一个版本保存的模块序列化文件，在另一个版本可能无法载入。

```python
torch.save(model, PATH)
# 模型类必须在此之前被定义
model = torch.load(PATH)
model.eval()
```
### 5.3  Checkpoint 用于推理/继续训练
- 在训练时，不仅要保存模型相关的信息，还要保存优化器相关的信息。因为可能要从检查点出发，继续训练。所以可以保存优化器本身的状态字典，存储包括当前学习率、调度器等信息。
- 最新记录的训练损失，外部的 torch.nn.Embedding 层等等都可以保存。
- PyTorch 中常见的保存checkpoint 是使用 .tar 文件扩展名。
-  要加载项目，首先需要初始化模型和优化器，然后使用 torch.load() 来加载本地字典

一个模型的检查点代码如下：
```python
torch.save({
'epoch': epoch,
'model_state_dict': model.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
'loss': loss,
...
}, PATH)
```
加载

```python
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()#或model.train()
```
或者是：

```python
save_info = { # 保存的信息
    "iter_num": iter_num,  # 迭代步数 
    "optimizer": optimizer.state_dict(), # 优化器的状态字典
    "model": model.state_dict(), # 模型的状态字典
}
# 保存信息
torch.save(save_info, save_path)
# 载入信息
save_info = torch.load(save_path)
optimizer.load_state_dict(save_info["optimizer"])
model.load_state_dict(sae_info["model"])
```

### 5.4 在一个文件中保存多个模型
```python
torch.save({
'modelA_state_dict': modelA.state_dict(),
'modelB_state_dict': modelB.state_dict(),
'optimizerA_state_dict': optimizerA.state_dict(),
'optimizerB_state_dict': optimizerB.state_dict(),
...
}, PATH)
```
加载
```python
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)
checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
modelA.eval()
modelB.eval()
```
当保存一个模型由多个 torch.nn.Modules 组成时，例如GAN(对抗生成网络)、sequence-to-
sequence (序列到序列模型), 或者是多个模 型融合, 可以采用与保存常规检查点相同的方法。
换句话说，保存每个模型的 state_dict 的字典和相对应的优化器。如前所述，可以通 过简单地
将它们附加到字典的方式来保存任何其他项目，这样有助于恢复训练。

## 六、TensorBoard的安装和使用
>pip install tensorflow-tensorboard
pip install tensorboard
安装完之后import tensorboard时报错ImportError: TensorBoard logging requires TensorBoard version 1.15 or above
试了几种方法。最后关掉ipynb文件，新建一个ipynb文件复制代码运行就好了。

### 6.1 TensorBoard用法示例：

```python
from sklearn.datasets import load_boston
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

#定义线性回归模型
class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim

        self.weight = nn.Parameter(torch.randn(ndim, 1)) # 定义权重
        self.bias = nn.Parameter(torch.randn(1)) # 定义偏置

    def forward(self, x):
        # 定义线性模型 y = Wx + b
        return x.mm(self.weight) + self.bias

boston = load_boston()
lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6)

data = torch.tensor(boston["data"], requires_grad=True, dtype=torch.float32)
target = torch.tensor(boston["target"], dtype=torch.float32)
writer = SummaryWriter() # 构造摘要生成器，定义TensorBoard输出类

for step in range(10000):
    predict = lm(data)
    loss = criterion(predict, target)
    writer.add_scalar("Loss/train", loss, step) # 输出损失函数
    writer.add_histogram("Param/weight", lm.weight, step) # 输出权重直方图
    writer.add_histogram("Param/bias", lm.bias, step) # 输出偏置直方图
    if step and step % 1000 == 0 :
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad()
    loss.backward()
    optim.step()
    
writer.close()
```
训练完之后，在当前目录下面会生成一个文件夹runs。runs下面还有一个文件夹（名字和训练时间、主机名称有关）

- from torch.utils.tensorboard import SummaryWriter是从tensorboard构造一个摘要写入器SummaryWriter。实例化之后调用实例化方法添加要写入摘要的张量信息。
- add_scalar：添加标量数据，比如loss、acc等
- add_histogram：添加直方图
- add_graph()：创建Graphs，Graphs中存放了网络结构
- 运行tensorboard-logdir./run命令，启动tensorboard服务器。默认端口6006。访问http://127.0.0.1:6006可以看到tensorboard网页界面。

### 6.2 具体函数
#### 6.2.1 SummaryWriter
```python
 writer = SummaryWriter(log_dir=None, comment='',
    purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')
```
- log_dir：tensorboard文件的存放路径，默认是创建runs文件夹
- flush_secs：表示写入tensorboard文件的时间间隔
- purge_step:可视化数据不是实时写入，而是有个队列。积累的数据超过队列限制的时候，触发数据文件写入。如果写入的可视化数据崩溃，purge_step步数之后的数据将会被舍弃
- max_queue:写入磁盘之前内存中最多可以保留的事件（数据）的数量
- filaname_suffix:可视化数据文件的后缀，默认为空字符串

#### 6.2.2  add_scalar()和add_scalars()
- add_scalar()
```python
add_scalar(tag, scalar_value, global_step=None, walltime=None)
```

于在tensorboard中加入loss，其中常用参数有：
	- tag：不同图表的标签，如下图所示的Train_loss。
	- scalar_value：标签的值，浮点数
	- global_step：当前迭代步数，标签的x轴坐标
	- walltime：迭代时间函数。如果不传入，方法内部使用time.time()返回一个浮点数代表时间
```python
writer.add_scalar('Train_loss', loss, (epoch*epoch_size + iteration))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/ee5a72c90ce742c981ba80b6c5cb4877.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
- add_scalars()
```python
add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
```
和上一个方法类似，通过传入一个主标签（main_tag），然后传入键值对是标签和标量值的一个字典（tag_scalar_dict），对每个标量值进行显示。

#### 6.2.3 add_histogram()
显示张量分量的直方图和对应的分布
```python
add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None)
```
- bins：产生直方图的方法，可以是tensorflow、auto、fd
- max_bins:最大直方图分段数
#### 6.2.4 add_graph
传入pytorch模块及输入，显示模块对应的计算图
- model：pytorch模型
- input_to_model：pytorch模型的输入

```python
if Cuda:
    graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor).cuda()
else:
    graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor)
writer.add_graph(model, (graph_inputs,))
```
#### 6.2.5 add_pr_curve
显示准确率-召回率曲线（Prediction-Recall Curve）。
```python
add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127,
    weights=None, walltime=None)
```
-  labels：目标值
- predictions：预测值
- num_thresholds：曲线中间插值点数
- weights：每个点的权重
#### 6.2.6 tensorboard --logdir=
完成tensorboard文件的生成后，可在命令行调用该文件，tensorboard网址。

```python
#打开cmd命令
tensorboard --logdir=.\Chapter2\runs --bind_all
#TensorBoard 2.2.2 at http://DESKTOP-OHLNREI:6006/ (Press CTRL+C to quit)
```
#### 6.2.7  add_image、add_vide、add_audio、add_text


### 6.3 tensorboard界面简介
右上方三个依次是：
- SCALARS：损失函数图像
- DISTRIBUTIONS：权重分布（随时间）
- HISTOGRAMS：权重直方图分布

![DISTRIBUTIONS](https://img-blog.csdnimg.cn/39cf2f69f92948a1ab248ff51adb72f3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![HISTOGRAMS](https://img-blog.csdnimg.cn/f9f9756f47954534a608cc9c6b6ba73b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
权重分布和直方图应该是随着训练一直变化，直到分布稳定。如果一直没有变化，可能模型结构有问题或者反向传播有问题。

Scalars：这个面板是最常用的面板，主要用于将神经网络训练过程中的acc（训练集准确率）val_acc（验证集准确率），loss（损失值），weight（权重）等等变化情况绘制成折线图。
- Ignore outlines in chart scaling（忽略图表缩放中的轮廓），可以消除离散值
- data downloadlinks：显示数据下载链接，用来下载图片
- smoothing：图像的曲线平滑程度，值越大越平滑。每个mini-batch的loss不一定下降，smoothing越大时，代表平均的mini-batch越多。
- Horizontal Axis：水平轴表示方式。
	- STEP：表示迭代次数
	- RELATIVE：表示按照训练集和测试集的相对值
	- WALL：表示按照时间。
