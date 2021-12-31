@[toc]
# 一、Dataset和DataLoader加载数据集
## 1.torch.utils.data
torch.utils.data主要包括以下三个类： 
1. class torch.utils.data.Dataset
其他的数据集类必须是torch.utils.data.Dataset的子类,比如说torchvision.ImageFolder. 
2. class torch.utils.data.sampler.Sampler(data_source) 
参数: data_source (Dataset) – dataset to sample from 
作用: 创建一个采样器, class torch.utils.data.sampler.Sampler是所有的Sampler的基类, 其中,iter(self)函数来获取一个迭代器,对数据集中元素的索引进行迭代,len(self)方法返回迭代器中包含元素的长度. 
3. class torch.utils.data.DataLoader

## 2. 加载数据流程
pytorch中加载数据的顺序是：
1. 加载数据，提取出feature和label，并转换成tensor
2. 创建一个dataset对象
3. 创建一个dataloader对象，dataloader类的作用就是实现数据以什么方式输入到什么网络中
4. 循环dataloader对象，将data,label拿到模型中去训练
代码一般是这么写的：

```python
# 定义学习集 DataLoader
train_data = torch.utils.data.DataLoader(各种设置...) 
# 将数据喂入神经网络进行训练
for i, (input, target) in enumerate(train_data): 
    循环代码行......
```
## 3. Dataset
Dataset是我们用的数据集的库，是Pytorch中所有数据集加载类中应该继承的父类。其中父类中的两个私有成员函数必须被重载，否则将会触发错误提示。其中__len__应该返回数据集的大小，而__getitem__应该编写支持数据集索引的函数

```python
class Dataset(object):
    def __init__(self):
        ...       
    def __getitem__(self, index):
        return ...    
    def __len__(self):
        return ...
```
上面三个方法是最基本的，其中__getitem__是最主要的方法，它规定了如何读取数据。其主要作用是能让该类可以像list一样通过索引值对数据进行访问。

```python
class FirstDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self):
        # 初始化，定义你用于训练的数据集(文件路径或文件名列表)，以什么比例进行sample（多个数据集的情况），每个epoch训练样本的数目，预处理方法等等
        #也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        pass
    def __getitem__(self, index):
         #从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
         #预处理数据（例如torchvision.Transform）。
         #返回数据对（例如图像和标签）。
         #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # 定义为数据集的总大小。
```
图片加载的dataset可以参考帖子：[《带你详细了解并使用Dataset以及DataLoader》](https://blog.csdn.net/qq_33431368/article/details/105463045)
人民币二分类参考：[《pytorch - 数据读取机制中的Dataloader与Dataset》](https://blog.csdn.net/qq_37388085/article/details/102663166)
## 4. dataloader类及其参数
dataloader类调用torch.utils.Data.DataLoader，实际过程中数据集往往很大，通过DataLoader加载数据集使用mini-batch的时候可以使用多线程并行处理，这样可以加快我们准备数据集的速度。Datasets就是构建这个工具函数的实例参数之一。一般可以这么写：

```python
train_loader = DataLoader(dataset=train_data, batch_size=6, shuffle=True ，num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False，num_workers=4)
```
下面看看dataloader代码：
```python
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None)
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

## 5. dataloader内部函数
### 5.1 __next__函数
DataLoader__next__函数用for循环来遍历数据进行读取。
```python
def __next__(self): 
        if self.num_workers == 0:   
            indices = next(self.sample_iter)  
            batch = self.collate_fn([self.dataset[i] for i in indices]) # this line 
            if self.pin_memory: 
                batch = _utils.pin_memory.pin_memory_batch(batch) 
            return batch
```
仔细看可以发现，前面还有一个self.collate_fn方法，这个是干嘛用的呢?在介绍前我们需要知道每个参数的意义：

- indices: 表示每一个iteration，sampler返回的indices，即一个batch size大小的索引列表
- self.dataset[i]: 前面已经介绍了，这里就是对第i个数据进行读取操作，一般来说self.dataset[i]=(img, label)

看到这不难猜出collate_fn的作用就是将一个batch的数据进行合并操作。默认的collate_fn是将img和label分别合并成imgs和labels，所以如果你的__getitem__方法只是返回 img, label,那么你可以使用默认的collate_fn方法，但是如果你每次读取的数据有img, box, label等等，那么你就需要自定义collate_fn来将对应的数据合并成一个batch数据，这样方便后续的训练步骤。

### 5.2 DataLoaderIter函数
```python
def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))
        super(DataLoader, self).__setattr__(attr, val)

def __iter__(self):
        return _DataLoaderIter(self)

def __len__(self):
        return len(self.batch_sampler)
```
当代码运行到要从torch.utils.data.DataLoader类生成的对象中取数据的时候，比如：
```python
train_data=torch.utils.data.DataLoader(...)
for i, (input, target) in enumerate(train_data):
```
就会调用DataLoader类的__iter__方法：return DataLoaderIter(self)，此时牵扯到DataLoaderIter类：

```python
def __iter__(self)：
	 if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
	 else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)
```
- SingleProcessDataLoaderIter：单线程数据迭代，采用普通方式来读取数据
- MultiProcessingDataLoaderIter：多进程数据迭代，采用队列的方式来读取。 

MultiProcessingDataLoaderIter继承的是BaseDataLoaderIter,开始初始化，然后Dataloader进行初始化，然后进入 next __（）方法 随机生成索引，进而生成batch，最后调用 _get_data() 方法得到data。idx, data = self._get_data()， data = self.data_queue.get(timeout=timeout)

****
总结一下：
1. 调用了dataloader 的__iter__() 方法, 产生了一个DataLoaderIter
2. 反复调用DataLoaderIter 的__next__()来得到batch, 具体操作就是, 多次调用dataset的__getitem__()方法 (如果num_worker>0就多线程调用), 然后用collate_fn来把它们打包成batch. 中间还会涉及到shuffle , 以及sample 的方法等,
3. 当数据读完后, next()抛出一个StopIteration异常, for循环结束, dataloader 失效.

DataLoaderIter的源码及详细解读参考：[《PyTorch源码解读之torch.utils.data.DataLoader》](https://blog.csdn.net/u014380165/article/details/79058479)


## 6. dataloader循环
ataloader本质上是一个可迭代对象，但是dataloader不能像列表那样用索引的形式去访问，而是使用迭代遍历的方式。

```python
for i in dataLoader:
	print(i.keys())
```
也可以使用enumerate(dataloader)的形式访问。
在计算i的类型时，发现其为一个字典，打印这个字典的关键字可得到

```python
for i in dataLoader:
	print(i.keys())
```
```python
dict_keys(['text', 'audio', 'vision', 'labels'])
```
同理，计算 **i[‘text’]**发现其为一个张量，打印该张量信息
```python
print(i['text'].shape)  #64*39*768
```
此时的64恰好就是我们设置的batchsize，并且最后一个i值的text的shape为24*39*768，即24个数据
# 二、代码示例
## 1. transformer单句文本分类（HF教程）
### 1.1使用Trainer训练
GLUE榜单包含了9个句子级别的分类任务，分别是：

- CoLA (Corpus of Linguistic Acceptability) 鉴别一个句子是否语法正确.
- MNLI (Multi-Genre Natural Language Inference) 给定一个假设，判断另一个句子与该假设的关系：entails, contradicts 或者 unrelated。
- MRPC (Microsoft Research Paraphrase Corpus) 判断两个句子是否互为paraphrases.
- QNLI (Question-answering Natural Language Inference) 判断第2句是否包含第1句问题的答案。
- QQP (Quora Question Pairs2) 判断两个问句是否语义相同。
- RTE (Recognizing Textual Entailment)判断一个句子是否与假设成entail关系。
- SST-2 (Stanford Sentiment Treebank) 判断一个句子的情感正负向.
- STS-B (Semantic Textual Similarity Benchmark) 判断两个句子的相似性（分数为1-5分）。
- WNLI (Winograd Natural Language Inference) Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not.

加载数据集
```python
from datasets import load_dataset
raw_datasets = load_dataset("glue","sst2")
```
预处理数据
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]
```

定义评估函数
```python
import numpy as np
from datasets import load_metric

metric = load_metric("glue","sst2")#改成"accuracy"效果一样吗？

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```
加载模型
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```
配置 Trainer参数：

```python
from transformers import TrainingArguments，Trainer

args = TrainingArguments(
    "ft-sst2",                          # 输出路径，存放检查点和其他输出文件
    evaluation_strategy="epoch",        # 定义每轮结束后进行评价
    learning_rate=2e-5,                 # 定义初始学习率
    per_device_train_batch_size=16,     # 定义训练批次大小
    per_device_eval_batch_size=16,      # 定义测试批次大小
    num_train_epochs=2,                 # 定义训练轮数
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```
开始训练：
```python
trainer.train()
```
训练完毕后，执行以下代码，得到模型在验证集上的效果：
```python
trainer.evaluate()
```

```python
{'epoch': 2,
 'eval_loss': 0.9351930022239685，
 'eval_accuracy'': 0.7350917431192661
 }
```

### 1.2 使用 PyTorch进行训练
重新启动笔记本以释放一些内存，或执行以下代码：
```python
del model
del pytorch_model
del trainer
torch.cuda.empty_cache()
```
首先，我们需要定义数据加载器，我们将使用它来迭代批次。 在这样做之前，我们只需要对我们的 tokenized_datasets 应用一些后处理：
1. 删除与模型不期望的值相对应的列（此处为“text”列）
2. 将列“label”重命名为“labels”（因为模型期望参数被命名为标签）
3. 设置数据集的格式，以便它们返回 PyTorch 张量而不是列表。

tokenized_datasets 对每个步骤处理如下：

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence","idx"])#删除多余的“sebtence”列和“idx”列,否则会报错forward() got an unexpected keyword argument 'idx'
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")#列“label”重命名为“labels”，否则报错forward() got an unexpected keyword argument 'label'
tokenized_datasets.set_format("torch")#返回 PyTorch 张量，否则报错'list' object has no attribute 'size'
```
二三步也可以合并：

```python
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
tokenized_datasets.set_format(type='torch', columns=columns)
```
切出一部分数据集

```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```
定义dataloaders：

```python
from torch.utils.data import DataLoader
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```
定义模型：

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```
定义优化器optimizer 和学习率调度器scheduler：

```python
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

#默认使用的学习率调度器只是线性衰减从最大值（此处为 5e-5）到 0：
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
```
使用GPU进行训练：

```python
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```
使用 tqdm 库在训练步骤数上添加了一个进度条，并定义训练循环：

```python
from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()#设置train状态，启用 Batch Normalization 和 Dropout。
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
编写评估循环，在循环完成时计算最终结果之前累积每个批次的预测：

```python
metric= load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

### 1.3 句子对文本分类（rte）：

```python
dataset = load_dataset('glue', 'rte')
metric = load_metric('glue', 'rte')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)
def tokenize(examples):
    return tokenizer(examples['hypothesis'],examples['premiere'] truncation=True, padding='max_length')
dataset = dataset.map(tokenize, batched=True)
```
其它代码一样.更多文本分类参考datawhale-transformer教程4.1：[《文本分类》](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.1-%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)
### 1.4 更多示例
要查看更多微调示例，您可以参考：
🤗[Transformers Examples](https://github.com/huggingface/transformers/tree/master/examples)，其中包括在 PyTorch 和 TensorFlow 中训练所有常见 NLP 任务的脚本。
🤗  [Transformers Notebooks](https://huggingface.co/transformers/notebooks.html) ，其中包含各种笔记本，尤其是每个任务一个（查找如何在 xxx 上微调模型）。

## 2. 科大讯飞中文相似度代码赏析
转载自[《10分钟 杀入科大讯飞中文相似度 Top10！》](https://gitee.com/coggle/competition-baseline/blob/master/competition/%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9EAI%E5%BC%80%E5%8F%91%E8%80%85%E5%A4%A7%E8%B5%9B2021/%E4%B8%AD%E6%96%87%E9%97%AE%E9%A2%98%E7%9B%B8%E4%BC%BC%E5%BA%A6%E6%8C%91%E6%88%98%E8%B5%9B/bert-nsp-xunfei.ipynb)
### 2.1赛题解析
- 赛题名称：中文问题相似度挑战赛
http://challenge.xfyun.cn/topic/info?type=chinese-question-similarity&ch=dw-sq-1

- 赛题介绍
重复问题检测是一个常见的文本挖掘任务，在很多实际问答社区都有相应的应用。重复问题检测可以方便进行问题的答案聚合，以及问题答案推荐，自动QA等。由于中文词语的多样性和灵活性，本赛题需要选手构建一个重复问题识别算法。

- 赛题任务
本次赛题希望参赛选手对两个问题完成相似度打分。

- 训练集：约5千条问题对和标签。若两个问题是相同含义，标签为1；否则为0。
测试集：约5千条问题对。

- 训练集样例：
句子1：有哪些女明星被潜规则啦
句子2：哪些女明星被潜规则了
标签：1
句子1：泰囧完整版下载
句子2：エウテルペ完整版下载
标签：0
- 解题思路
赛题为经典的文本匹配任务，所以可以考虑使用Bert的NSP来完成建模。

### 2.2 代码实例
步骤1：读取数据集

```python
import pandas as pd
import codecs
train_df = pd.read_csv('train.csv', sep='\t', names=['question1', 'question2', 'label'])
```

```python
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
```

import re
并按照标签划分验证集：

```python
# stratify 按照标签进行采样，训练集和验证部分同分布
q1_train, q1_val, q2_train, q2_val, train_label, test_label =  train_test_split(
    train_df['question1'].iloc[:], 
    train_df['question2'].iloc[:],
    train_df['label'].iloc[:],
    test_size=0.1, 
    stratify=train_df['label'].iloc[:])
```

步骤2：文本进行tokenizer
使用Bert对文本进行转换，此时模型选择bert-base-chinese。

```python
# pip install transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_encoding = tokenizer(list(q1_train), list(q2_train), 
                           truncation=True, padding=True, max_length=100)
val_encoding = tokenizer(list(q1_val), list(q2_val), 
                          truncation=True, padding=True, max_length=100)
```

步骤3：定义dataset

```python
# 数据集读取
class XFeiDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = XFeiDataset(train_encoding, list(train_label))
val_dataset = XFeiDataset(val_encoding, list(test_label))
```

步骤4：定义匹配模型
使用BertForNextSentencePrediction完成文本匹配任务，并定义优化器。

```python
from transformers import BertForNextSentencePrediction, AdamW, get_linear_schedule_with_warmup
model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 单个读取到批量读取
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# 优化方法
optim = AdamW(model.parameters(), lr=1e-5)
```

```python
# 精度计算
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
```

步骤5：模型训练与验证
祖传代码：模型正向传播和准确率计算。

```python
# 训练函数
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        
        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 参数更新
        optim.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))
```
    
```python
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in val_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f"%(total_eval_loss/len(val_dataloader)))
    print("-------------------------------")
```
    
```python
for epoch in range(5):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()
    torch.save(model.state_dict(), f'model_{epoch}.pt')
```

```python
#打印输出看看
outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
print(outputs)
NextSentencePredictorOutput(loss=tensor(0.4528, device='cuda:0'), logits=tensor([[ 2.7850,  1.2451],
        [ 3.9663, -0.9795],
        [ 0.1072,  4.8910],
        [ 3.2274,  0.4685]], device='cuda:0'), hidden_states=None, attentions=None)
```

步骤6：对测试集进行预测
读取测试集数据，进行转换。

```python
test_df = pd.read_csv('test.csv', sep='\t', names=['question1', 'question2', 'label'])
test_df['label'] = test_df['label'].fillna(0)

test_encoding = tokenizer(list(test_df['question1']), list(test_df['question2']), 
                          truncation=True, padding=True, max_length=100)
test_dataset = XFeiDataset(test_encoding, list(test_df['label']))
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
```

对测试集数据进行正向传播预测，得到预测结果，并输出指定格式。

```python
def predict():
    model.eval()
    test_predict = []
    for batch in test_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs[0]
        logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        test_predict += list(np.argmax(logits, axis=1).flatten())
        
    return test_predict
    
test_label = predict()
pd.DataFrame({'label':test_label}).to_csv('submit.csv', index=None)
```
## 3. CCF BDCI 剧本角色情感识别
本节转自[《CCF BDCI 剧本角色情感识别：多目标学习开源方案》](https://mp.weixin.qq.com/s/xl-MAlI1KroZrmpWGttEuA)
### 3.1 赛事解析
1. 赛题名称
剧本角色情感识别 比赛链接：https://www.datafountain.cn/competitions/518
后台回复“爱奇艺”可以获取完整代码

2. 赛题背景
剧本对影视行业的重要性不言而喻。一部好的剧本，不光是好口碑和大流量的基础，也能带来更高的商业回报。剧本分析是影视内容生产链条的第一环，其中剧本角色的情感识别是一个非常重要的任务，主要是对剧本中每句对白和动作描述中涉及到的每个角色从多个维度进行分析并识别出情感。相对于通常的新闻、评论性文本的情感分析，有其独有的业务特点和挑战。

3. 赛题任务
本赛题提供一部分电影剧本作为训练集，训练集数据已由人工进行标注，参赛队伍需要对剧本场景中每句对白和动作描述中涉及到的每个角色的情感从多个维度进行分析和识别。该任务的主要难点和挑战包括：1）剧本的行文风格和通常的新闻类语料差别较大，更加口语化；2）剧本中角色情感不仅仅取决于当前的文本，对前文语义可能有深度依赖。

4. 数据简介
比赛的数据来源主要是一部分电影剧本，以及爱奇艺标注团队的情感标注结果，主要用于提供给各参赛团队进行模型训练和结果验证使用。

数据说明
训练数据：训练数据为txt格式，以英文制表符分隔，首行为表头，字段说明如下：
字段名称	类型	描述	说明
id	String	数据ID	-
content	String	文本内容	剧本对白或动作描写
character	String	角色名	文本中提到的角色
emotion	String	情感识别结果（按顺序）	爱情感值，乐情感值，惊情感值，怒情感值，恐情感值，哀情感值

备注：
- 本赛题的情感定义共6类（按顺序）：爱、乐、惊、怒、恐、哀；  
- 情感识别结果：上述6类情感按固定顺序对应的情感值，情感值范围是[0, 1, 2, 3]，0-没有，1-弱，2-中，3-强，以英文半角逗号分隔；  
- 本赛题不需要识别剧本中的角色名；  文件编码：UTF-8 无BOM编码

5. 评估标准
本赛题算法评分采用常用的均方根误差（RMSE）来计算评分，按照“文本内容+角色名”识别出的6类情感对应的情感值来统计。
图片score = 1/(1 + RMSE)

其中是yi,j预测的情感值，xi,j是标注的情感值，n是总的测试样本数。最终按score得分来排名。

6. 基于预训练模型的对目标学习

这个题目可操作的地方有很多，一开始见到这个比赛的时候见想到了multi outputs的模型构建，这里给大家分享下这个基线，希望有大佬能够针对这个思路优化上去~
### 3.2  代码示例
加载数据

```python
with open('data/train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    data = list()
    for line in tqdm(lines):
        sp = line.split('\t')
        if len(sp) != 4:
            print("ERROR:", sp)
            continue
        data.append(sp)

train = pd.DataFrame(data)
train.columns = ['id', 'content', 'character', 'emotions']

test = pd.read_csv('data/test_dataset.tsv', sep='\t')
submit = pd.read_csv('data/submit_example.tsv', sep='\t')
train = train[train['emotions'] != '']
```

提取情感目标
```python
train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])

train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
```

构建数据集
数据集的标签一共有六个：

```python
class RoleDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text=str(self.texts[item])
        label=self.labels[item]
        
        encoding=self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
		# print(encoding['input_ids'])
        sample = {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        for label_col in target_cols:
            sample[label_col] = torch.tensor(label[label_col], dtype=torch.float)
        return sample
```
        
模型构建

```python
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.out_love = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_joy = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fright = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_anger = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_sorrow = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        love = self.out_love(pooled_output)
        joy = self.out_joy(pooled_output)
        fright = self.out_fright(pooled_output)
        anger = self.out_anger(pooled_output)
        fear = self.out_fear(pooled_output)
        sorrow = self.out_sorrow(pooled_output)
        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }
```

6.4 模型训练
回归损失函数直接选取 nn.MSELoss()

```python
EPOCHS = 1 # 训练轮数

optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.MSELoss().to(device)
```

模型总的loss为六个目标值的loss之和

```python
def train_epoch(
  model, 
  data_loader, 
  criterion, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for sample in tqdm(data_loader):
        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss_love = criterion(outputs['love'], sample['love'].to(device))
        loss_joy = criterion(outputs['joy'], sample['joy'].to(device))
        loss_fright = criterion(outputs['fright'], sample['fright'].to(device))
        loss_anger = criterion(outputs['anger'], sample['anger'].to(device))
        loss_fear = criterion(outputs['fear'], sample['fear'].to(device))
        loss_sorrow = criterion(outputs['sorrow'], sample['sorrow'].to(device))
        loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow
        
        
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
	#return correct_predictions.double() / (n_examples*6), np.mean(losses)
    return np.mean(losses)
```

