
@[toc]
## Transformers
🤗 Transformers为自然语言理解（NLU）和自然语言生成（NLG）提供通用架构（BERT、GPT-2、RoBERTa、XLM、DistilBert、XLNet...）  具有 100 多种语言的 32 多种预训练模型以及 Jax、PyTorch 和 TensorFlow 之间的深度互操作性。
[所有模型检查点](https://huggingface.co/models)都从 Huggingface.co [模型中心](https://huggingface.co/)无缝集成，用户和组织直接上传。当前检查点数量22752。

## Contents
文档分为五个部分：

- GET STARTED：包含快速浏览、安装说明和一些关于我们的理念和词汇表的有用信息。
- USING 🤗 TRANSFORMERS ：包含有关如何使用库的一般教程。
- ADVANCED GUIDES ：包含特定于给定脚本或库的一部分的更高级指南。
- RESEARCH 侧重于与如何使用库关系不大的教程，但更多地涉及transformers的一般研究
- 最后三个部分包含每个公共类和函数的文档，分为：
	- MAIN CLASSES ：transformers库的重要 API 的main classes。
	- models：transformers库实现的每个模型相关的类和函数。
	- INTERNAL HELPERS ：内部使用的类和函数的内部帮助程序。

该库目前包含以下68种模型的 Jax、PyTorch 和 Tensorflow 实现、预训练模型权重、使用脚本和转换实用程序（conversion utilities）。

下表列出每个模型的当前支持：
- 是否有PyTorch support、TensorFlow support或Flax Support
- 是否支持Tokenizer slow和Tokenizer fast
![model Support](https://img-blog.csdnimg.cn/b60d39d59c9e428e99e7eb80ea3e2f14.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
# GET STARTED
- [快速游览](https://huggingface.co/transformers/quicktour.html)
	- [开始使用管道执行任务](https://huggingface.co/transformers/quicktour.html#getting-started-on-a-task-with-a-pipeline)
	- [Under the hood：预训练模型](https://huggingface.co/transformers/quicktour.html#under-the-hood-pretrained-models)
- 安装（略）
	- [缓存模型](https://huggingface.co/transformers/installation.html#caching-models)
	- [您想在移动设备上运行 Transformer 模型吗？](https://huggingface.co/transformers/installation.html#do-you-want-to-run-a-transformer-model-on-a-mobile-device)
- 哲学
	- [主要概念](https://huggingface.co/transformers/philosophy.html#main-concepts)
- 词汇表
	- [一般规则](https://huggingface.co/transformers/glossary.html#general-terms)
	- [模型输入](https://huggingface.co/transformers/glossary.html#model-inputs)
## 快速浏览
让我们快速浏览一下 🤗 Transformers 库功能。 该库下载用于自然语言理解 (NLU) 任务的预训练模型，例如分析文本的情绪，以及自然语言生成 (NLG)，例如用新文本完成提示或翻译成另一种语言。

首先，我们将看到如何轻松利用管道 API 在推理中快速使用这些预训练模型。 然后，我们将进一步挖掘，看看该库如何让您访问这些模型并帮助您预处理数据。

>文档中提供的所有代码示例在 Pytorch 与 TensorFlow 的左上角都有一个开关。 如果不是，则该代码预计适用于两个后端，无需任何更改。

### 使用管道执行任务
>[Youtube视频](https://youtu.be/tiZFewofSLM)

在给定任务上使用预训练模型的最简单方法是使用 pipeline()。
Transformers 提供了以下开箱即用的任务：
- 情感分析Sentiment analysis：文本是正面的还是负面的？
- 文本生成Text generation：提供提示，模型将生成以下内容。
- 名称实体识别 (NER)：在输入句子中，用它代表的实体（人、地点等）标记每个单词
- 问答Question answering：为模型提供一些上下文和问题，从上下文中提取答案。
- 填充掩码文本Filling masked text：给定带有掩码单词的文本（例如，替换为 [MASK]），填空。
- 摘要Summarization：生成一个长文本的摘要。
- 翻译Translation：用另一种语言翻译文本。
- 特征提取Feature extraction：返回文本的张量表示。

让我们看看它如何用于情感分析（其他任务都在[task summary](https://huggingface.co/transformers/task_summary.html)中介绍）：

```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
```
首次输入此命令时，会下载并缓存预训练模型及其标记器。分词器的工作是预处理文本输入模型，然后模型负责进行预测。 管道将所有这些组合在一起，并对预测进行后处理以使其可读。 例如：

```python
classifier('We are very happy to show you the 🤗 Transformers library.')
[{'label': 'POSITIVE', 'score': 0.9998}]
```
你可以输入多个句子，这些句子批处理输入模型，返回一个像这样的字典列表：

```python
results = classifier(["We are very happy to show you the 🤗 Transformers library.",
           "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```
默认情况下，为此管道下载的模型称为“distilbert-base-uncased-finetuned-sst-2-english”。 我们可以查看它的[模型页面](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)以获取有关它的更多信息。 它使用 [DistilBERT 架构](https://huggingface.co/transformers/model_doc/distilbert.html)，并针对情感分析任务在名为 SST-2 的数据集上进行了微调。

假设我们想使用一个用法语预训练的模型。 我们可以搜索 [model hub](https://huggingface.co/models)，该中心收集了研究实验室在大量数据上预训练的模型，以及社区模型 community models （通常是特定数据集上那些大模型的微调版本）。 应用标签 “French” 和 “text-classification”会返回一个建议“nlptown/bert-base-multilingual-uncased-sentiment”。 让我们看看如何使用它。

您可以直接将要使用的模型名称“bert-base-multilingual-uncased-sentiment“”传递给 pipeline()：

```python
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
```
这个分类器现在可以处理英语、法语以及荷兰语、德语、意大利语和西班牙语的文本！ model除了接收模型名称，也可以传入保存预训练模型的本地文件夹（见下文）。 您还可以传递model object及其关联的tokenizer。

为此，==我们需要两个类==：
-  AutoTokenizer：下载与我们选择的模型相关联的标记器并实例化它
-  AutoModelForSequenceClassification（或 TFAutoModelForSequenceClassification）：下载模型本身

注意：如果我们在其他任务中使用该库，模型的类会发生变化。 [ task summary（任务摘要）](https://huggingface.co/transformers/task_summary.html) 总结了哪个类用于哪个任务。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```
现在，要下载我们之前找到的模型和标记器，我们只需要使用 from_pretrained() 方法（ model_name可以任意替换为其它的transformers模型）：

```python
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
```
如果预训练模型的数据集和您的数据集不够相似，则需要在您的的数据集上进行微调。 我们提供了示例脚本来执行此操作。 完成后，不要忘记在社区分享您的微调模型。

### Under the hood: 预训练模型
现在让我们看看在使用这些管道时会发生什么。
>[Youtube视频](https://youtu.be/AhChOFRegn4)

正如我们所见，模型和分词器是使用 from_pretrained 方法创建的：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
#### tokenizer预处理
分词器负责对您的文本进行预处理。 
- 使用与模型预训练时相同的分词器来分词：有多个规则可以管理文本分词，使用模型名称来实例化分词器，可以确保我们使用与模型相同的分词规则来预训练。

- 使用与模型预训练时相同的词汇表，将这些标记转换为数字，以便能够从中构建张量并将它们提供给模型。  from_pretrained 方法实例化tokenizer时就会下载对应的词汇表。

将文本输入分词器就可以执行以上步骤：

```python
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
```
这将返回一个字符串到整数列表的字典。 它包含：
- token的 id
- 对模型有用的其他参数，如attention_mask（注意力掩码，模型将使用它来更好地理解序列）：

```python
print(inputs)
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
句子列表可以直接传递给分词器。 如果是批处理句子，则
- 每个batch需要全部填充到相同的长度
- 过长句子截断

最终返回张量输入模型。可以在tokenizer中设定这些参数：

```python
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,        #填充
    truncation=True,     #截断
    max_length=512,
    return_tensors="pt" #将input_ids转为Pytorch张量。Pytorch最终只接受张量作为模型输入
)
```
填充一般只在句子的一侧填充（如右侧）。attention_mask=True或者0表示padding，计算注意力时需要忽略：

```python
for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")
input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
```
您可以在[此处](https://huggingface.co/transformers/preprocessing.html)了解有关tokenizers的更多信息。
#### 使用模型
数据预处理后输入模型。如果您使用的是 TensorFlow 模型，则可以将字典键直接传递给张量，对于 PyTorch 模型，您需要通过添加 ** 来解压字典。

```python
pt_outputs = pt_model(**pt_batch)
```
在 🤗 Transformers 中，所有输出都是包含模型的最终激活以及其他元数据metadata的对象。 此处更详细地描述了这些对象。 现在，让我们自己检查输出：

```python
print(pt_outputs)
SequenceClassifierOutput(loss=None, logits=tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
```
>所有🤗 Transformers 模型（PyTorch 或 TensorFlow）在最终激活函数（如 SoftMax）之前返回模型的激活值，因为这个最终激活函数通常与损失融合。

让我们用 SoftMax 来获得预测:

```python
from torch import nn
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
```

```python
print(pt_predictions)
tensor([[2.2043e-04, 9.9978e-01],
        [5.3086e-01, 4.6914e-01]], grad_fn=<SoftmaxBackward>)
```
如果除了输入之外还为模型提供标签，模型输出对象还将包含一个 loss 属性：

```python
import torch
pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
print(pt_outputs)
SequenceClassifierOutput(loss=tensor(0.3167, grad_fn=<NllLossBackward>), logits=tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
```
models是标准的 torch.nn.Module 或 tf.keras.Model，因此您可以在通常的训练循环中使用它们。 🤗 Transformers 还提供了一个 Trainer（或 TFTrainer）类来帮助您进行训练（处理诸如分布式训练、混合精度等）。 有关更多详细信息，请参阅[培训教程](https://huggingface.co/transformers/training.html)。
>Pytorch 模型输出是特殊的数据类，因此您可以在 IDE 中自动完成其属性。 它们的行为也像元组或字典（例如，您可以使用整数、切片或字符串进行索引），在这种情况下，未设置的属性（具有 None 值）将被忽略。

#### 模型的保存和PyTorch 、TensorFlow 相互加载
模型微调后进行保存：

```python
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```
然后，您可以使用 from_pretrained() 方法通过传递目录名称而不是模型名称来加载此模型。

 🤗 Transformers 可以轻松地在 PyTorch 和 TensorFlow 之间切换：。 TensorFlow 模型中加载已保存的 PyTorch 模型，请像这样使用 from_pretrained()：

```python
from transformers import TFAutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = TFAutoModel.from_pretrained(save_directory, from_pt=True)
```
 PyTorch 模型中加载保存的 TensorFlow 模型：

```python
from transformers import AutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModel.from_pretrained(save_directory, from_tf=True)
```
最后，如果需要，您还可以要求模型返回所有隐藏状态和所有注意力权重：

```python
pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states  = pt_outputs.hidden_states
all_attentions = pt_outputs.attentions
```
### 两种模型加载方式（Accessing the code）
AutoModel 和 AutoTokenizer 类只是可以自动与任何预训练模型一起使用的快捷方式，在背后，该库为每个架构加类的组合提供一个model class。
当模型是“distilbert-base-uncased-finetuned-sst-2-english”时，[AutoModelForSequenceClassification](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModelForSequenceClassification)所自动创建的模型就是 [DistilBertForSequenceClassification](https://huggingface.co/transformers/model_doc/distilbert.html#transformers.DistilBertForSequenceClassification)。所以也有另一种加载模型的方式：

```python
## PYTORCH CODE
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

## TENSORFLOW CODE
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```
### 自定义模型
如果要更改模型本身的构建方式，可以自定义配置。 每个架构都有自己的相关配置。 例如，DistilBertConfig 允许您指定 DistilBERT 的隐藏维度hidden dimension、 dropout rate等参数。 如果您进行核心修改，例如更改hidden size，您将无法再使用预训练模型，而需要从头开始训练。再将直接从此配置实例化模型。

下面，我们使用 from_pretrained() 方法为分词器加载预定义的词汇表。 然后从头开始初始化模型。 因此，不是使用 from_pretrained() 方法，而是从配置中实例化模型：

```python
## PYTORCH CODE
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)

## TENSORFLOW CODE
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification(config)
```
==在from_pretrained() 方法中就可以修改model head的参数（例如，标签的数量）==，而不是创建model 的新配置来改变label数量。 例如，定义一个10个标签的分类器：

```python
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```
## 安装
transformer的安装见[官方文档](https://huggingface.co/transformers/installation.html)
### 缓存模型
transformers库给预训练模型提供了本地下载和缓存的功能。 使用 from_pretrained 等方法时，模型将自动下载到 shell 环境变量 TRANSFORMERS_CACHE 给出的文件夹中。 它的默认值将是 Hugging Face 缓存主页，后跟 /transformers/。 如下（按优先顺序）：
- shell 环境变量 HF_HOME
- shell 环境变量 XDG_CACHE_HOME + /huggingface/
- 默认值：~/.cache/huggingface/

所以如果你没有设置任何特定的环境变量，缓存目录将在 ~/.cache/huggingface/transformers/。

此外可以使用cache_dir=... 指定模型存储位置。
### 离线模式
可以在有防火墙或无网络的环境中运行 🤗 Transformers。

- 设置环境变量 TRANSFORMERS_OFFLINE=1 ：🤗 Transformers 只使用本地文件，不会尝试查找。
- 可同时使用HF_DATASETS_OFFLINE=1 （它对 🤗 数据集执行和上面相同的操作）

Here is an example of how this can be used on a filesystem that is shared between a normally networked and a firewalled to the external world instances.
下面是一个示例，说明如何在正常联网和防火墙到外部世界实例之间共享的文件系统上使用它。

在具有正常网络的实例上运行您的程序，该程序将下载和缓存模型（如果您使用 🤗 数据集，还可以选择数据集）。 例如：

```python
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```
然后使用相同的文件系统filesystem，可以在防火墙实例firewalled instance上运行相同的程序：

```python
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path t5-small --dataset_name wmt16 --dataset_config ro-en ...
```
并且它应该成功而不会hanging waiting to timeout。

### 下载模型和tokenizer，离线使用
之前的代码都是在线下载（模型、tokenizer），文件缓存以备稍后使用。 但也可以下载到本地使用。

下载文件的两种方式：
- 在 Web 界面单击“Download”按钮下载
- 输入代码从Huggingface_hub 库下载

下载的内容可以选择为：
- 使用 snapshot_download 下载整个存储库
- 使用 hf_hub_download 下载特定文件

详细的下载方法请参考[文档](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub)。

### 您想在移动设备上运行 Transformer 模型吗？
swift-coreml-transformers 存储库的工具，可将 PyTorch 或 TensorFlow 2.0 训练的 Transformer 模型（目前包含 GPT-2、DistilGPT-2、BERT 和 DistilBERT）转换为在 iOS 设备上运行的 CoreML 模型。未来三者可以无缝连接使用。

## Philosophy
🤗 Transformers 是一个opinionated库，专为：

- 寻求使用/研究/扩展大型变压器模型的 NLP 研究人员和教育工作者
- 希望微调这些模型和/或在生产中为它们服务的实践从业者
- 只想下载预训练模型并使用它来解决给定 NLP 任务的工程师。

该库的设计考虑了两个重要目标：

- 尽可能简单快速地使用：
	- 几乎没有任何abstractions，每个模型只需要三个标准类：配置、模型和标记器。
	- 所有这些类可用通用的 from_pretrained() 实例化方法初始化，from_pretrained() 下载、缓存和加载相关类实例和相关数据（超参数、词汇表和模型的权重）。这些来自 Hugging Face Hub 上的预训练检查点或您自己保存的检查点。
	- 在这三个基类之上，该库提供了两个 API：
		- pipeline() 用于在给定任务上快速使用模型（加上其关联的标记器和配置）
		- Trainer()/TFT​​rainer() 用于快速训练或微调给定的模型。
	- 因此，该库不是神经网络构建块的模块化工具箱。如果您想扩展/构建库，只需使用常规 Python/PyTorch/TensorFlow/Keras 模块并从库的基类继承以重用模型加载/保存等功能。
- 提供最先进的模型，其性能尽可能接近原始模型：
	- 我们为每个架构提供至少一个示例，该示例重现了所述架构的官方作者提供的结果。
	- 代码通常尽可能接近原始代码库，这意味着某些 PyTorch 代码可能不像 TensorFlow 代码转换后那样具有 pytorchic，反之亦然。

其他几个目标：

- 尽可能一致地公开模型的内部结构：
- 我们使用单个 API 访问完整的隐藏状态和注意力权重。
- Tokenizer 和基本模型的 API 已标准化，以便在模型之间轻松切换。
- 结合主观选择的有前途的工具来微调/调查这些模型：
- 一种向词汇表和嵌入中添加新标记以进行微调的简单/一致的方法。
- 屏蔽和修剪变压器头的简单方法。
- 在 PyTorch 和 TensorFlow 2.0 之间轻松切换，允许使用一种框架进行训练并使用另一种框架进行推理。
### 主要概念
transformers库围绕每个模型的三种类构建：

- Model classes ，例如 BertModel，包含30 多个 PyTorch 模型 (torch.nn.Module) 或 Keras 模型 (tf.keras.Model)，它们使用库中提供的预训练权重。

- Configuration classes 配置类，如 BertConfig存储构建模型所需的所有参数，可自动实例化模型配置（这是模型的一部分）。
- Tokenizer 类，例如 BertTokenizer。它存储每个模型的词汇表，并提供用于编码/解码要馈送到模型的token embedding索引列表中的字符串的方法。

所有这些类都可用预训练实例来实例化，本地保存方法有两种：

- from_pretrained() ：从库本身提供的预训练版本，或用户本地存储（或在服务器上）实例化模型/配置/标记器，
- save_pretrained() 允许您在本地保存模型/配置/标记器，以便可以使用 from_pretrained() 重新加载它。
## 词汇表
### General terms：
- 自编码模型：参见 MLM
- 自回归模型：参见 CLM
- CLM：因果语言建模，一种预训练任务，模型按顺序读取文本并预测下一个单词。它通常是通过阅读整个句子，但在模型内部使用掩码来隐藏特定时间步长的未来标记来完成的。
- 深度学习：使用多层神经网络的机器学习算法。
- MLM：屏蔽语言建模，一种预训练任务，其中模型看到文本的损坏版本，通常通过随机屏蔽一些标记来完成，并且必须预测原始文本。
- 多任务模式：将文本与另一种输入（例如图像）相结合的任务。
- NLG：自然语言生成，所有与生成文本相关的任务（例如与转换器对话、翻译）。
- NLP：自然语言处理，“处理文本”的通用方式。
- NLU：自然语言理解，所有与理解文本内容相关的任务（例如对整个文本、单个单词进行分类）。
- 预训练模型：已对某些数据（例如维基百科的所有数据）进行预训练的模型。预训练方法涉及一个自我监督的目标，它可以是阅读文本并尝试预测下一个单词（参见 CLM）或屏蔽一些单词并尝试预测它们（参见 MLM）。
- RNN：循环神经网络，一种使用层上的循环来处理文本的模型。
- self-attention：输入的每个元素找出他们应该关注输入的哪些其他元素。
- seq2seq 或序列到序列：从输入生成新序列的模型，如翻译模型或摘要模型（如 Bart 或 T5）。
- token：句子的一部分，通常是一个词，但也可以是一个子词（非常用词通常被拆分成子词）或标点符号。
- Transformer：基于自注意力的深度学习模型架构。
### 模型输入：
大多数模型使用相同的输入，此处将与使用示例一起详细说明。
#### Input IDs
输入 id 通常是作为输入传递给模型的唯一必需参数。 它们是标记索引token indices，标记的数字表示构建将用作模型输入的序列。（numerical representations of tokens building the sequences that will be used as input by the model.）
