# 微调预训练模型
@[toc]
本文翻译自 [Hugging Face主页](https://huggingface.co/)Resources下的 [course](https://huggingface.co/course/chapter3/1?fw=pt)

说明：有的文章将token、Tokenizer、Tokenization翻译为令牌、令牌器和令牌化。虽然从意义上来说更加准确，但是笔者感觉还是不够简单直接，不够形象。所以文中有些地方会翻译成分词、分词器和分词，有些地方又保留英文（有可能google翻译成标记、标记化没注意到）。有其它疑问可以留言或者查看原文。
## 1. 本章简介
在第 2 章中，我们探讨了如何使用分词器和预训练模型进行预测。 但是，如果您想为自己的数据集微调预训练模型怎么办？ 这就是本章的主题！ 你将学习：

- 如何从Model Hub 准备大型数据集
- 如何使用high-level Trainer API来微调模型
- 如何使用自定义训练循环custom training loop
- 如何利用 🤗 Accelerate 库在任何分布式设置上轻松运行该custom training loop

要将经过训练的checkpoint上传到 Hugging Face Hub，您需要一个 Huggingface.co 帐户：[创建一个帐户](https://huggingface.co/join)。
## 2. 处理数据
继续上一章的例子，下面是我们如何在 PyTorch中训练一个序列分类器sequence classifier，一次一个batch：

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",]
#设置每个batch都padding和截断，并返回PyTorch张量
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```
仅仅在两个句子上训练模型不会产生很好的结果。 所以需要准备更大的数据集来进行训练。

在本节中，我们将使用 William B. Dolan 和 Chris Brockett 在一篇论文中介绍的 MRPC（微软研究释义语料库）数据集作为示例。该数据集由 5,801 对句子组成，带有一个标签，表明它们是否是互为释义paraphrases（即，两个句子的意思是否相同）。 选择它是因为这是一个小数据集，因此很容易进行训练。
### 从Hub上下载dataset
Youtube 视频：[Hugging Face Datasets Overview](https://youtu.be/_BZearw7f0w)（pytorch）

Hub 不仅包含模型；还含有多个[datasets](https://huggingface.co/datasets)，这些datasets有很多不同的语言。我们建议您在完成本节后尝试加载和处理新数据集（[参考文档](https://huggingface.co/docs/datasets/loading_datasets.html#from-the-huggingface-hub)）。 

 MRPC 数据集是构成 [GLUE 基准](https://gluebenchmark.com/)的 10 个数据集之一。而GLUE 基准是一种学术基准，用于衡量 ML 模型在 10 个不同文本分类任务中的性能。

🤗 Datasets库提供了一个非常简单的命令来下载和缓存Hub上的dataset。 我们可以像这样下载 MRPC 数据集：

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

```python
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```
这样就得到一个DatasetDict对象，包含训练集、验证集和测试集，训练集中有3,668 个句子对，验证集中有408对，测试集中有1,725 对。每个句子对包含四列数据：'sentence1', 'sentence2', 'label'和 'idx'。

load_dataset命令下载并缓存数据集，默认在 ~/.cache/huggingface/dataset 中。您可以通过设置 HF_HOME 环境变量来自定义缓存文件夹。

和字典一样，raw_datasets 可以通过索引访问其中的句子对：

```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

```python
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
```
```python
import pandas as pd
validation=pd.DataFrame(raw_datasets['validation'])
validation
```
![validation](https://img-blog.csdnimg.cn/df0adbee66ba40dc862e64f5df5e0022.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
可见标签已经是整数，不需要再做任何预处理。通过raw_train_dataset的features属性可以知道每一列的类型：
```python
raw_train_dataset.features
```

```python
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```
label是 ClassLabel 类型，整数到label name的映射存储在names文件夹中。label=1表示这对句子互为paraphrases，label=0表示句子对意思不一致。

>✏️试试看！ 查看训练集的元素 15 和验证集的元素 87。 他们的标签是什么？
### 数据集预处理
YouTube视频[《Preprocessing sentence pairs》](https://youtu.be/0u3ioSwev3s)

通过tokenizer可以将文本转换为模型能理解的数字，我们可以像这样将每个句子对的两个句子分词处理：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```
然而，我们不能仅仅将两个序列传递给模型并预测这两个句子是否互为paraphrases。 我们需要将两个序列成对处理，并进行适当的预处理。 
幸运的是，tokenizer还可以按照 BERT 模型所期望的方式进行句子对处理：

```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

```python
{ 'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
token_type_ids用来区分前后两个句子。另外两个之前讲过。
>✏️快来试试吧！ 取训练集的第 15 个元素，分别对两个句子进行分词。 这两个结果有什么区别？

如果我们将 input_ids 中的 ID 解码回单词：

```python
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
```
可以得到：

```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
```
所以我们看到，当有两个句子时，模型期望输入的形式为 [CLS] 句子 1 [SEP] 句子 2 [SEP]。 将其与 token_type_ids 对齐：

```python
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```
可以看到，**输入中对应于[CLS]sentence1[SEP]的部分的token_type_ids 都是0**，而对应sentence2[SEP]的部分token_type_ids都是1。

请注意，如果您选择不同的checkpoint，您的tokenizer处理后输入中不一定会有 token_type_ids。例如，如果您使用 DistilBERT 模型，则不会返回它们。（因为DistilBERT是BERT的蒸馏模型，去掉了NSP——下句子预测任务 ）

在这里，BERT 使用token type IDs进行了预训练，在掩码语言建模MLM目标之上，进行NSP任务，对句子对之间的关​​系进行建模。（简写了部分原文，教程其它地方讲了）

只要tokenizer和model使用相同的checkpoint，就无需担心标记化输入中是否存在 token_type_ids。

将句子对列表传给tokenizer，就可以对整个数据集进行分词处理。因此，预处理训练数据集的一种方法是：

```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```
==这很有效，但它的缺点是返回字典（带有我们的键:input_ids、attention_mask 和 token_type_ids，以及列表中的列表的值）。 tokenization期间有足够的 RAM 来存储整个数据集时这种方法才有效（而 🤗 Datasets 库中的数据集是存储在磁盘上的 Apache Arrow 文件，因此您只需将请求加载的样本保存在内存中）。==

为了将数据保留为dataset，我们将使用更灵活的Dataset.map 方法。此方法可以完成更多的预处理而不仅仅是 tokenization。 map 方法是对数据集中的每个元素应用同一个函数，所以让我们定义一个函数来对输入进行tokenize预处理：

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```
这个函数接受一个字典（就像我们dataset的items）并返回一个带有键 input_ids、attention_mask 和 token_type_ids 的新字典。 

分批处理时字典包含多个样本（==每个键作为一个句子列表==），此时调用 map 函数可以使用batched=True选项 ，这将大大加快tokenization过程。 因为 🤗 Tokenizers 库中的Tokenizer用 Rust 编写，一次处理很多输入时这个分词器可以非常快。

在tokenization函数中省略了padding 参数，这是因为padding到该批次中的最大长度时的效率，会高于所有序列都padding到整个数据集的最大序列长度。 当输入序列长度很不一致时，这可以节省大量时间和处理能力！

以下是对整个数据集应用tokenization方法。 我们在 ==map 调用中使用了 batched=True，因此该函数一次应用于数据集的整个batch元素，而不是分别应用于每个元素。 这样预处理速度会更快。==

```python
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```
==🤗 Datasets库应用这种处理的方式是向数据集添加新字段，每个字段对应预处理函数返回的字典中的每个键。==

```python
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```
，Dataset.map函数进行预处理时可以设定num_proc 参数来进行多线程处理。 我们在这里没有这样做，因为 🤗 Tokenizers 库已经使用多个线程来更快地tokenize我们的样本。如果您没有使用由该库支持的fast tokenizer，这可以加快您的预处理？

==我们的 tokenize_function 返回一个包含 input_ids、attention_mask 和 token_type_ids 键的字典，因此这三个字段被添加到我们数据集的所有splits部分中。 请注意，如果我们的预处理函数为我们应用map的数据集中的现有键返回一个新值，我们也可以更改现有字段。==

最后，当我们将输入序列进行批处理时，要将所有输入序列填充到本批次最长序列的长度——我们称之为动态填充技术dynamic padding。



### Dynamic padding动态填充技术
youtube视频：[《what is Dynamic padding》](https://youtu.be/7q5NyFT8REg)
在 PyTorch 中，负责将一批样本放在一起的函数称为整理函数collate function。这是您在构建 DataLoader 时可以传递的参数，默认值是一个函数，它将您的样本转换为 PyTorch 张量并连接它们（如果您的元素是列表、元组或字典，则递归）。在我们的例子中这是不可能的，因为我们所拥有的输入不会都是相同的大小。我们特意推迟了填充，只在每批必要时应用它，避免过长的输入和大量的填充。这将大大加快训练速度，但请注意，如果您在 TPU 上进行训练，它可能会导致问题——TPU 更喜欢固定形状，即使这需要额外的填充。

为了在实践中做到这一点，我们必须定义一个 collat​​e 函数，它将对我们想要一起批处理的数据集的items应用正确的填充数量。幸运的是，🤗 Transformers 库通过 DataCollat​​orWithPadding 为我们提供了这样的功能。当您实例化它时，它需要一个tokenizer（以了解要使用哪个填充标记，以及模型希望填充在输入的左侧还是右侧），并且会执行您需要的所有操作：

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
为了测试这个小功能，从训练集中选取我们想要一起批处理的样本。这里需要删除 idx、sentence1 和 sentence2 列，因为不需要这些列并且它们包含字符串（不能创建张量）。查看批处理中每个输入的长度：

```python
samples = tokenized_datasets["train"][:8]
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}
[len(x) for x in samples["input_ids"]]
```

```python
[50, 59, 47, 67, 59, 50, 62, 32]
```
我们得到了不同长度的序列。动态填充意味着该批次中的序列都应该填充到 67 的长度。 如果没有动态填充，所有样本都必须填充到整个数据集中的最大长度，或者模型可以接受的最大长度。 让我们仔细检查我们的 data_collator 是否正确地动态填充批处理：

```python
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
```

```python
{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}
```
看起来不错！ 现在我们已经从原始文本变成了我们的模型可以处理的批处理数据batches，我们准备好对其进行微调！
>✏️快来试试吧！ 在 GLUE SST-2 数据集上复制预处理。 SST-2 数据集由单个句子而不是成对组成，但其余部分处理方式应该看起来相同。 对于更难的挑战，请尝试编写一个适用于任何 GLUE 任务的预处理函数。
## 3. 使用Trainer API微调模型
[open in colab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter3/section3.ipynb) 
colab上下载和运行很快，建议尝试。
🤗 Transformers 提供了一个 Trainer 类，可以用来在你的数据集上微调任何预训练模型。 数据预处理后，只需要再执行几个步骤来定义 Trainer。 最困难的部分可能是准备运行 Trainer.train 的环境，因为它在 CPU 上运行速度非常慢。 如果您没有设置 GPU，则可以在 Google Colab 上访问免费的 GPU 或 TPU。

下面的代码示例假定您已经执行了上一节中的示例：

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")#MRPC判断两个句子是否互为paraphrases
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
### 训练
Trainer 第一个参数是TrainingArguments 类，包含 Trainer 用于训练和评估的所有超参数。 唯一一个必须提供的参数是：保存model和checkpoint的目录（The only argument you have to provide is a directory where the trained model will be saved, as well as the checkpoints along the way）。 其它参数可以选取默认值。

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```
第二步：定义模型
和上一章一样，我们将使用 AutoModelForSequenceClassification 类，带有两个标签：
（其实就是根据自己的任务选择任务头task head，以便进行微调）
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```
在实例化此预训练模型后会收到警告。 这是因为 BERT 没有在句子对分类方面进行过预训练，所以预训练模型的head已经被丢弃，而是添加了一个适合序列分类的new head。 警告表明一些权重没有使用（对应于丢弃的预训练head部分），而其他一些权重被随机初始化（new head部分）， 最后鼓励您训练模型。

有了模型之后，就可以定义一个训练器Trainer，将迄今为止构建的所有对象传递给它。这些对象包括：model、training_args、训练和验证数据集、data_collator 和tokenizer。（这都是Trainer的参数）：

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```
像上面这样传递tokenizer时，参数data_collator 是之前定义的 DataCollatorWithPadding，所以此调用中的 data_collator=data_collator行可以跳过。（但是像之前一样写出这一步很重要It was still important to show you this part of the processing in section 2!）

要在我们的数据集上微调模型，我们只需要调用 Trainer 的 train方法：

```python
trainer.train()
```
训练完毕显示：

```python
The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, sentence2, idx.
***** Running training *****
  Num examples = 3668
  Num Epochs = 3
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 8
  Gradient Accumulation steps = 1
  Total optimization steps = 1377
  
Step   Training Loss
500    0.544700
1000   0.326500

TrainOutput(global_step=1377, training_loss=0.3773723704795865, metrics={'train_runtime': 379.1704, 'train_samples_per_second': 29.021, 'train_steps_per_second': 3.632, 'total_flos': 405470580750720.0, 'train_loss': 0.3773723704795865, 'epoch': 3.0})
#运行中只显示500 steps和1000 steps的结果，最终是1377 steps，最终loss是0.377
```
开始微调（在colab上用 GPU 6分钟左右），每 500 steps报告一次训练损失。 但是，它不会告诉您模型的表现如何。 这是因为：
1. 没有设置evaluation_strategy 参数，告诉模型多少个“steps”（eval_steps）或“epoch”来评估一次损失。
2. Trainer的compute_metrics 可以计算训练时具体的评估指标的值（比如acc、F1分数等等）。不设置compute_metrics 就只显示loss，不是一个直观的数字。
### Evaluation
compute_metrics 函数必须传入一个 EvalPrediction 对象作为参数。 EvalPrediction是一个具有预测字段和 label_ids 字段的元组。
compute_metrics返回的结果是字典，键值对类型分别是strings和floats（strings是metrics的名称，floats是具体的值）。

也就是[教程4.1](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.1-%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)说的：直接调用metric的compute方法，传入labels和predictions即可得到metric的值。也只有这样做才能在训练时得到acc、F1等结果（具体指标根据不同任务来定）：
```python
tokenized_datasets["validation"]
```

```python
Dataset({
    features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
    num_rows: 408
})
```

我们可以使用 Trainer.predict 命令获得模型的预测结果：

```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

```python
(408, 2) (408,)
```
predict 方法输出一个具有三个字段的元组，三个字段分别是predictions、label_ids 和 metrics。 metrics字段将只包含数据集传递的损失，以及一些time metrics （预测所需的总时间和平均时间）。
==compute_metrics 函数写好并将其传递给Trainer后，该字段也将包含compute_metrics 返回的metrics==。Once we complete our compute_metrics function and pass it to the Trainer, that field will also contain the metrics returned by compute_metrics.
![mrpc](https://img-blog.csdnimg.cn/7a920b0dddf147cf87b38fb18a0ad0a8.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

```python
metrics={'test_loss': 0.6269022822380066, 'test_runtime': 4.0653, 'test_samples_per_second': 100.362, 'test_steps_per_second': 12.545})
```
predictions是一个二维数组，形状为 408 x 2（验证集408组数据，每一组是两个句子）。 要预测结果与标签进行比较，我们需要在predictions第二个轴上取最大值的索引：
```python
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)
```
为了构建我们的 compute_metric 函数，我们将依赖 🤗 Datasets 库中的metric。 通过 load_metric 函数，我们可以像加载数据集一样轻松加载与 MRPC 数据集关联的metric。The object returned has a compute method we can use to do the metric calculation:

```python
from datasets import load_metric

metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

```python
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}#模型在验证集上的准确率为 85.78%，F1 分数为 89.97
```
每次训练时model head的随机初始化可能会改变最终的metric值，所以这里的最终结果可能和你跑出的不一样。 acc和F1 是用于评估 GLUE 基准的 MRPC 数据集结果的两个指标。 BERT 论文中的表格报告了基本模型的 F1 分数为 88.9。 那是un-cased模型，而我们目前使用的是cased模型，这说明了更好的结果。(cased就是指区分英文的大小写）

将以上内容整合到一起，得到 compute_metrics 函数：

```python
def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```
再设定每个epoch查看一次验证评估。所以下面就是我们设定compute_metrics参数之后的Trainer：

```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```
```python
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```
请注意，我们创建了一个新的 TrainingArguments，其evaluation_strategy 设置为“epoch”和一个新模型——否则，我们只会继续训练我们已经训练过的模型。 要启动新的训练运行，我们执行：

```python
trainer.train()
```
最终训练了6分33秒，比上一次稍微长了一点点。最后运行结果为：
```python
The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence1, sentence2, idx.
***** Running training *****
  Num examples = 3668
  Num Epochs = 3
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 8
  Gradient Accumulation steps = 1
  Total optimization steps = 1377
  
Epoch	Training Loss	Validation Loss	 Accuracy	   F1
1	       No log	      0.557327	     0.806373	0.872375
2	      0.552700	      0.458040	     0.862745	0.903448
3	      0.333900	      0.560826	     0.867647	0.907850
TrainOutput(global_step=1377, training_loss=0.37862846690325436, metrics={'train_runtime': 393.5652, 'train_samples_per_second': 27.96, 'train_steps_per_second': 3.499, 'total_flos': 405470580750720.0, 'train_loss': 0.37862846690325436, 'epoch': 3.0})
```
This time, it will report the validation loss and metrics at the end of each epoch on top of the training loss. Again, the exact accuracy/F1 score you reach might be a bit different from what we found, because of the random head initialization of the model, but it should be in the same ballpark.

The Trainer will work out of the box on multiple GPUs or TPUs and provides lots of options, like mixed-precision training (use fp16 = True in your training arguments). We will go over everything it supports in Chapter 10.

这次，模型训练时会在training loss之外，还报告每个 epoch 结束时的 validation loss和metrics。 同样，由于模型的随机头部(task head)初始化，您达到的准确准确率/F1 分数可能与我们发现的略有不同，但它应该在同一范围内。

Trainer 将在多个 GPU 或 TPU 上开箱即用，并提供许多选项，例如混合精度训练（在训练参数中使用 fp16 = True）。 我们将在第 10 章讨论它支持的所有内容。

使用 Trainer API 进行微调的介绍到此结束。 第 7 章将给出一个对最常见的 NLP 任务执行此操作的示例，但现在让我们看看如何在纯 PyTorch 中执行相同的操作。
## 4. 编写训练循环（不使用Trainer）
本节介绍不使用 Trainer 类的情况下进行训练，获得与上一节相同的结果。 数据预处理如下：

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
准备训练
在实际编写我们的训练循环之前，我们需要定义一些对象。 第一个是我们将用于批次迭代的数据加载器（the dataloaders we will use to iterate over batches）。 但是在我们定义这些dataloaders之前，我们需要对我们的 tokenized_datasets 应用一些后处理，以处理 Trainer 自动为我们做的一些事情。 具体来说，我们需要：

- 删除与模型不期望的值相对应的列（如sentence1 和sentence2 columns）。
- 将column label重命名为labels（因为模型期望 the argument to be named labels）。
- 设置数据集的格式，使其返回 PyTorch 张量而不是列表。

对于以上几步处理，可以这样设置tokenized_datasets：

```python
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names#检查结果是否只有我们的模型可以接受的列
```
```python
['attention_mask', 'input_ids', 'labels', 'token_type_ids']
```
结果如上所示，现在我们可以轻松定义我们的数据加载器dataloaders：

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```
为了快速检查数据处理中有没有错误，我们可以检查其中一个批次：

```python
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```

```python
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
```
由于我们为训练数据加载器设置了 shuffle=True ，并且我们是填充批次内的最大长度，所以您的最终结果可能和这个不一样。

数据预处理已完成，接着看模型。 我们像在上一节中所做的那样实例化它：

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```
正式训练之前，我们先传入一个batch看看：

```python
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

```python
tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```
提供labels后，所有 🤗 Transformers 模型都会返回损失，同时模型输出logits向量（每个batch输入两个sentences，所以张量大小为 8 x 2）。

离编写完整的训练循环还差两件事：优化器和学习率调节器（an optimizer and a learning rate scheduler）。 这次是复现Trainer所以使用Trainer的默认参数。 Trainer 使用的优化器是 AdamW，它与 Adam 相同，但对权重衰减正则化有所不同（参见 Ilya Loshchilov 和 Frank Hutter 的“[解耦权重衰减正则化 Decoupled Weight Decay Regularization ](https://arxiv.org/abs/1711.05101)”）：

```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```
最后，默认使用的learning rate scheduler是从最大值 (5e-5) 到 0 的线性衰减。 为了正确定义学习率调节器，我们需要知道我们将采取的训练步数training steps，即 epochs × training batches （training dataloader长度）。 Trainer 默认使用三个 epoch，则有：

```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print(num_training_steps)
```

```python
1377
```

5. polynomial

```python
def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
```

    return LambdaLR(optimizer, lr_lambda, last_epoch)
### The training loop
最后，如果希望使用GPU来训练，可以定义一个device，把我们的model和batches放在上面：

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```
```python
device(type='cuda')
```
为了查看训练进度，可以使用 tqdm 库设置一个进度条：

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
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
You can see that the core of the training loop looks a lot like the one in the introduction。 接着我们添加一个评估循环evaluation loop，否则训练时看不到模型如何运作的信息。
### The evaluation loop
上一节中，我们使用 🤗 Datasets 库提供的metric，以及metric.compute方法。但实际上metrics在我们用 add_batch方法遍历prediction loop（评估循环）时，可以accumulate batches。一旦我们accumulated all the batches，就可以用metric.compute方法得到最终结果。

以下是在prediction loop中实现所有这些的方法：

```python
from datasets import load_metric

metric= load_metric("glue", "mrpc")
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

```python
{'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}
```
同样，由于模型头部初始化和数据shuffle的随机性，您的结果会略有不同，但它们应该在同一个范围内。
>✏️快来试试吧！ 修改之前的训练循环，在 SST-2 数据集上微调您的模型。
### 使用 🤗 Accelerate 增强训练循环
之前的训练循环都是使用单个 CPU 或 GPU 运行。要想在多个 GPU 或 TPU 上启用分布式训练，可以使用 🤗 Accelerate 库，并做一些调整就行。 Starting from the creation of the training and validation dataloaders，手动设定训练循环如下：

```python
#单个CPU或GPU运行时
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
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
对比一下其中的改变：
```python
#多个CPU或GPU运行时
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer)

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```
1. 添加导入行from accelerate import Accelerator
2. 添加行accelerator = Accelerator()，实例化一个 Accelerator 对象，该对象将检查环境并初始化合适的分布式设置（initialize the proper distributed setup）。 
3. 删除两行。🤗 Accelerate 为您处理设备放置（handles the device placement），因此您可以删除将模型放在 device上的那一行（或者，如果您愿意，可以将它们更改为使用 Accelerate.device 而不是 device）。

4. 添加两行。将dataloader、model和优化器optimizer传入到accelerator.prepare， 大部分工作在这行完成。 这会将这些对象包装在合适的container中，以确保您的分布式训练按预期工作。 
5. 删除行batch = {k: v.to(device) for k, v in batch.items()}
这行是将 batch放在device上（同样，如果您想保留它，您可以将其更改为使用accelerator.device）
6. 将loss.backward() 替换为accelerator.backward(loss) 。
>⚠️ 为了从 Cloud TPU 提供的加速中受益，我们建议使用分词器的 `padding="max_length"` 和 `max_length` 参数将您的样本填充到固定长度。

为了便于复制使用，以下是 🤗 Accelerate 的完整训练循环：

```python
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
将上面代码放在 train.py 脚本中，这样就可在任何类型的分布式设置上运行脚本代码。 要在分布式设置中试用它，使用这行命令后，将提示您回答几个问题并将您的答案转储到此配置文件中：

```python
accelerate config
```
启动分布式训练：
```python
accelerate launch train.py
```
如果您想在 Notebook 中尝试此操作（例如，在 Colab 上使用 TPU 对其进行测试），只需将代码粘贴到 training_function 中并使用以下命令运行最后一个单元格：

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```
您可以在 🤗 Accelerate repo中找到更多的[examples](https://github.com/huggingface/accelerate/tree/main/examples)。
## 5. Fine-tuning总结：
在前两章中，您了解了模型和分词器tokenizers，现在您知道如何针对您自己的数据对它们进行微调。 回顾一下，在本章中，您：

- 了解 [Hub](https://huggingface.co/datasets)中的数据集
- 学习了如何加载和预处理数据集，包括使用动态填充dynamic padding 和collators
- 模型微调和评估
- 编写了一个较低级别的训练循环
- 使用 🤗 Accelerate 轻松调整您的训练循环，使其适用于多个 GPU 或 TPU

[章末测验](https://huggingface.co/course/chapter3/6?fw=pt)
