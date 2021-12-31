@[toc]
## 一些说明
&#8195;&#8195;比赛官方链接为：[《零基础入门NLP - 新闻文本分类》](https://tianchi.aliyun.com/competition/entrance/531810/introduction)。
&#8195;&#8195;讨论区有大佬张帆、惊鹊和张贤等人的代码，值得大家仔细阅读。
&#8195;&#8195;最后我的模型参考了这些代码的一些config，比如bert.config，lr等等。然后大佬们的代码对我来说还是太复杂，pytorch功力不够，看的吃力。所以自己用huggingface实现了。
&#8195;&#8195;第一步分词我就考虑了很久，没有像张帆他们那样用pytorch具体一步步写，而是参考HF主页的教程。所以一开始我是翻译了构建tokenizer的教程，如果对比赛代码中分词有疑问的可以参考。


## 三、最终代码及解析
主要思路：
1. 构建分词器。参考HF教程[《How to train and use your very own tokenizer》](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)。3750、648、900这三个应该是标点符号（详见张帆task02的分析），直接把这三个替换成‘，’、‘.’和‘！’。主要是为了断句。在预分词器pre_tokenizers.BertPreTokenizer中，有根据标点进行断句的方法，直接将文本换成带标点的格式就行，预分词器会自动断句。
	- 和BERT 有关的 Tokenizer 主要写在[models/bert/tokenization_bert.py](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py)中。这部分内容其实在[nlp教程3.1](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A03-%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AATransformer%E6%A8%A1%E5%9E%8B%EF%BC%9ABERT/3.1-%E5%A6%82%E4%BD%95%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AABERT)里面有写。
	- BasicTokenizer负责处理的第一步——按标点、空格等分割句子，并处理是否统一小写，以及清理非法字符。对于中文字符，通过预处理（加空格）来按字分割；同时可以通过never_split指定对某些词不进行分割；
	- 分词器参考bert-baseChinese的分词器配置[tokenizer.json](https://huggingface.co/bert-base-chinese/blob/main/tokenizer.json)。具体的：
		- "normalizer":"BertNormalizer"。
		- "pre_tokenizer":{"type":"BertPreTokenizer"}
		- "post_processor":{"type":"TemplateProcessing"}
		- "decoder":{"type":"WordPiece","prefix":"##","cleanup":true}
		- "model":{"type":"WordPiece","unk_token":"[UNK]"...}
	- 词表大小选的7000，我是看讨论区是6900+，这里还有点没想清楚。中文的wordpiece是也可以吧高频率的汉字拼成词语吧，用‘##’连接。如果这样，采用wordpiece，vocab size大一点，最后整词掩码感觉效果会更好。但是整词掩码我不知道怎么写，所以最后没有用。没有整词掩码，就没有wordpiece的必要了。所以我做的有点矛盾，最后是懒得改了，就这么写。
2. 预训练bert模型。参考nlp教程4.5 [《微调语言模型》](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.5-%E7%94%9F%E6%88%90%E4%BB%BB%E5%8A%A1-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)。用BertConfig配置模型参数，设置了一个小型的初始化bert进行mlm任务预训练。
	- 训练集选择train_set和test两个csv，因为test预训练时不需要分类标签，只作为掩码任务，不存在数据泄露问题。注意训练数据要把三个token换成标点。
	- 最终我训练了8个epoch（第一次5个epoch，lr=4e-4，loss=1.695，batch_size=128。第二次3个epoch，lr=2e-4，结果一开始steps=3000时，loss=1.78，应该是第二次训练的lr还是太大，震荡了。第二个epoch快训练完才降到1.69，浪费了两个小时。最终loss是1.63）
	- 我选择的是colab的tpu进行训练，每个epoch是13903steps，大概50-60分钟左右。如果是colab-GPU，大概31-35小时。colab tpu使用可以参考我的代码。如果选择tpu时提示无法分配，不用管，继续连接，第二次连接我都成功了。
3. 分类微调，加一个首尾截断。我是之前看文章说文章分类首尾截断效果更好（[论文解读】文本分类上分利器:Bert微调trick大全](https://mp.weixin.qq.com/s/WBK-XYzP-vIf6Ni6GO-diQ)）。trainer没有首尾截断的机制，在前面数据处理时用pandas实现。最终训练了6个epoch，用tpu大概88分钟（我也是跑了两次。中间colab断了...）
4. 读取测试集，跟训练集一样处理，保存结果并提交。最终得分0.961。
```python
class BertTokenizer(PreTrainedTokenizer):
...
...
if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
...
...
class BasicTokenizer(object):
...
...
#BasicTokenizer中定义了标点分割的方法，不需要再去另外处理
def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]
```

### 3.1 构建分词器
参考本文第二节，并查看了[bert-base-chinese,josn](https://huggingface.co/bert-base-chinese/tree/main)文件配置分词器。训练语言模型参考[此教程](https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)及[中文翻译](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.5-%E7%94%9F%E6%88%90%E4%BB%BB%E5%8A%A1-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)。
感受：最坑的是训练分词器，从头到尾选择decoders, models, pre_tokenizers, processors, trainers, Tokenizer有点麻烦。最后装进PreTrainedTokenizerFast之后还有些东西需要设置，看了好多次文档才试出来。
```python
#从google云盘上加载数据
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/transformers/天池-入门NLP - 新闻文本分类')
```

```python
#安装transformers=4.11.2
!pip install transformers datasets

# 文件读取
import pandas as pd
from datasets import load_dataset
from datasets import Dataset

train_df=pd.read_csv('./train_set.csv',sep='\t')
test_df=pd.read_csv('./test_a.csv', sep ='\t')
df=pd.concat((train_df,test_df))
```

```python
#将3750/648/900改成标点符号，删除原text列，新增words列重名为text列
import re
def replacepunc(x):
  x=re.sub('3750',",",x)
  x=re.sub('900',".",x)
  x=re.sub('648',"!",x)
  return x

df['words']=df['text'].map(lambda x: replacepunc(x))
df.drop('text',axis=1,inplace=True)
df.columns=['label','text']

#数据载入dataset，去除多余的列，只保留text列
data=Dataset.from_pandas(df).remove_columns(['label', '__index_level_0__'])
```

```python
#构建数据批处理迭代器，这部分代码是参考HF主页教程
batch_size = 1000

def batch_iterator():
  for i in range(0, len(data), batch_size):
    yield data['text'][i : i + batch_size]
```

```python
#设置分词器并进行训练
#初始化分词器、预分词器
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=7000,min_frequency=2,special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.decoders = decoders.WordPiece(prefix="##")

#开始训练
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
```

```python
#进行分词后处理
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
mask_token_id = tokenizer.token_to_id("[MASK]")
pad_token_id = tokenizer.token_to_id("[PAD]")

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]",cls_token_id),("[SEP]",sep_token_id),("[MASK]",mask_token_id)],
    )

tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding(pad_token='[PAD]')
```

```python
#测试分词结果
encoding = tokenizer.encode('2491 4109 1757 7539 648 3695 3038 4490 23 7019 3731 4109 3792 2465',' 2893 7212 5296 1667 3618 7044 1519 5413 1283 6122 4893 7495 2435 5510')
encoding.tokens
```
```python
"""保存模型并重新加载
tokenizer已经完成，我们必须将它放在与我们要使用的模型相对应的标记器 fast 类。
正在构建的分词器与 Transformers 中的任何类都不匹配(分词器非常特殊)，
您可以将它包装在 PreTrainedTokenizerFast 中"""
tokenizer.save("tokenizer.json")

from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast
(tokenizer_file="tokenizer.json",
model_max_length=512,mask_token='[MASK]',pad_token='[PAD]',
unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]',
padding_side='right',return_special_tokens_mask=True)

#PreTrainedTokenizerFast中一定要设置mask_token，pad_token等，
#不然mlm报错没有设定mask_token以及分词器无法padding
```
### 3.2 预训练bert模型
参考nlp教程4.5
```python
#data_collator是一个函数，负责获取样本并将它们批处理成张量
#在data_collator中可以确保每次以新的方式完成随机掩蔽。
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=fast_tokenizer,mlm=True,mlm_probability=0.15)
```

```python
#初始化bert模型，参数参考讨论区代码
from transformers import BertConfig
config = BertConfig(
    vocab_size=7000,
    hidden_size=512,
    intermediate_size=4*512,
    max_position_embeddings=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    type_vocab_size=2
)

from transformers import BertForMaskedLM
model = BertForMaskedLM(config=config)

#（掉线后）加载训练到一半的模型：
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained('/content/drive/MyDrive/transformers/天池-入门NLP - 新闻文本分类/test-clm/checkpoint-56000')
```

```python
#数据进行分词预处理，删除‘text'列，否则后面拼接的时候会报错。
tokenized_datasets=data.map(lambda examples:fast_tokenizer(examples['text']),batched=True).remove_columns("text")
```

```python
# 拼接所有文本，这一块解释可以看nlp 4.5教程
block_size = 128
def group_texts(examples):

  concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
  total_length = len(concatenated_examples[list(examples.keys())[0]])
  # 我们将余数对应的部分去掉。但如果模型支持的话，可以添加padding，您可以根据需要定制此部件。
  total_length = (total_length // block_size) * block_size
  # 通过max_len进行分割。
  result = {
      k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
      for k, t in concatenated_examples.items()
  }
  result["labels"] = result["input_ids"].copy()
  return result
 
 lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)
```

```python
#加载和保存拼接后的文本，掉线的时候这么做
lm_datasets.save_to_disk('./lm')
import pandas as pd
from datasets import load_from_disk
lm_datasets=load_from_disk('./lm')
```

```python
#解码分词器预处理的lm_datasets数据，里面有标点符号
la=fast_tokenizer.decode(lm_datasets[0]['input_ids'])
la

[CLS] 2967 6758 339 2021 1854 3731 4109 3792 4149 1519 2058 3912 2465 2410 1219 6654 7539 264 2456 4811 1292 2109 6905 5520 7058 6045 3634 6591 3530 6508 2465 7044 1519 3659 2073, 3731 4109 3792 6831 2614 3370 4269 3370 486 5770 4109 4125, 5445 2466 6831 6758 3743 3630 1726 2313 5906 826 4516 657. 1871 7044, 2967 3731 1757 1939! 2828 4704 7039 3706, 965 2490 7399 3743 2145 2407 7451 3775 6017 5998 1641 299 4704 2621 7029 3056 6333 433! 1667 1099. 2289 1099! 5780 220 7044 1279 7426 4269, 2967 6758 6631 3099 2205 7305 2620 5977, 3329 1793 6666 2042 3193 4149 1519 7039 3706 2446 5399
```

```python
#使用GPU训练，运行这段代码
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```
==GPU memory开始占用1GB，但是还没开始使用计算。==

```python
#安装TPU依赖
import os
assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

#将模型复制到TPU进行训练
import torch_xla.core.xla_model as xm
device = xm.xla_device()
model.to(device)
```

```python
#设定args和trainer准备训练.3000步看一次loss，9000步保存一次模型（怕掉线）
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    "Test-Clm",
    logging_strategy="steps",
    logging_steps=3000,
    save_strategy="steps",
    save_steps=9000,
    num_train_epochs=2,
    learning_rate=3e-4,
    per_device_train_batch_size=96,
    weight_decay=0.01)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    data_collator=data_collator)
```

```python
#训练并保存模型
trainer.train()

trainer.save_model("./pre_Bert")
```
这段是当时batch_size太高，显存爆了，我找一下原因。可以忽略。
1%的数据试验
1. 第一次训练涨到5.9GB，5个epoch，540steps，batch_size=128。训练完后是显存1.2GB。logging_steps=100,可以选择多久看一次loss。
2. 再次训练没有指定batch_size，显存是1.8GB。训练完1.5GB。算了一下默认batch_size=8。

### 3.3 分类任务微调：
1. 加载预训练好的模型，GPU或TPU训练
```python

from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained("./pre_Bert",num_labels=14)

#使用GPU训练
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#将模型复制到TPU进行训练
import torch_xla.core.xla_model as xm
device = xm.xla_device()
model.to(device)
```
2. 读取数据集，准备进行预处理

```python
from datasets import Dataset
import pandas as pd
#读取数据并shuffle
train_df=pd.read_csv('./train_set.csv',sep='\t').sample(frac=1)

#将训练数据中三个token换成标点
train_df['texts']=train_df['text'].map(lambda x:replacepunc(x))


#准备将text文本首尾截断，各取255tokens
def slipt2(x):
	ls=x.split(' ')
	le=len(ls)
	if le<511:
	    return x
    else:
	    return ' '.join(ls[:255]+ls[-255:])
```
3. 划分训练集和测试集，比例0.1
```python
val_df=train_df.iloc[:20000, ]
trains_df=train_df.iloc[20000:, ]

#首尾截断
val_df['summary']=val_df['texts'].apply(lambda x:slipt2(x))
trains_df['summary']=trains_df['texts'].apply(lambda x:slipt2(x))

#加载到dataset并预处理
trains_ds=Dataset.from_pandas(trains_df).remove_columns(["texts","text"])
val_ds=Dataset.from_pandas(val_df).remove_columns(["texts","text"])

tokenized_trains_ds=trains_ds.map(lambda examples:fast_tokenizer(examples['summary'],truncation=True,padding=True),batched=True)
tokenized_val_ds=val_ds.map(lambda examples:fast_tokenizer(examples['summary'],truncation=True,padding=True),batched=True)
```


4. 设置TrainingArguments和Trainer
```python
#设置acc评估方式
from datasets import load_metric
metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
	return metric.compute(predictions=predictions, references=labels)
 
#进行任务微调
from transformers import TrainingArguments,Trainer
args=TrainingArguments(
  output_dir='news-classification-2',
  evaluation_strategy="epoch",
  save_strategy="epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=96,
  per_device_eval_batch_size=96,
  num_train_epochs=6,
  weight_decay=0.01,
  load_best_model_at_end=True,
  metric_for_best_model="accuracy")

trainer=Trainer(
  model,
  args,
  train_dataset=tokenized_trains_ds,
  eval_dataset=tokenized_val_ds,
  tokenizer=fast_tokenizer,
  compute_metrics=compute_metrics)
```

==训练完GPU memory还是1.5GB==
```python
trainer.train()
trainer.save_model("./finally_bert")
```
==一开始训练,GPU memory跳到15.8GB（batch_size=128）。爆了之后选择分类微调模型的batch_size=16，GPU memory为3.4GB==

5. 最后读取测试集，预测结果
```python
#读取测试集并预处理
#读取测试集
import pandas as pd
from datasets import load_dataset
test_df=pd.read_csv('./test_a.csv',sep='\t')

#将训练数据中三个token换成标点
test_df['texts']=test_df['text'].map(lambda x:replacepunc(x))

#首尾截断
from datasets import Dataset
test_df['summary']=test_df['texts'].apply(lambda x:slipt2(x))

#加载到dataset并预处理
test_ds=Dataset.from_pandas(test_df).remove_columns(["texts","text"])

tokenized_test_ds=test_ds.map(lambda examples:fast_tokenizer(examples['summary'],truncation=True,padding=True),batched=True)
```

```python
#用trainer预测结果并保存
predictions,metrics,Loss=trainer.predict(tokenized_test_ds,metric_key_prefix="test")
pred=np.argmax(predictions,axis=1)
pd.DataFrame({'label':pred}).to_csv('submit1022.csv',index=None)
```

3.4 赛事总结：
1. 一开始要搞懂baseline的基本框架和config作为参考，如果比较难读不懂，可以直接跑一遍或者debug（还没有跑，有人说内存爆了。。。）
2. 最开始用少量数据跑，batch_size、学习率、数据集、epoch和时长确定好再跑一遍。之前就是嫌时间太长加了batch_size结果跑到模型微调时显存崩了，所有缓存数据都没了。
3. 中间数据和训练中的模型要记得保存，一旦掉线或者崩了或者想修改参数可以继续加载再跑。因为一开始不知道如何保存和加载datasets数据、kaggle的notebook老是无法保存，结果总是白跑了模型，浪费时间。
4. 想到再补
 
## 零、分词tokenization
比赛数据脱敏，需要从头开始预训练。第一步就是建立词表，训练自己的分词器

参考资料：[《Summary of the tokenizers》](https://huggingface.co/transformers/tokenizer_summary.html)
[《\[NLP\]——BPE、WordPiece、Unigram and SentencePiece》](https://blog.csdn.net/jokerxsy/article/details/116998827)

==wordpiece和BPE的差异在于合并时对token对的选择:BPE是选择出现次数最大的，wordpiece衡量的是token对和单独的两个token之间的概率差，选择概率差最大的进行合并。==

考虑token a和b，以及合并之后的token ab，概率差的公式如下:
$$p(a,b)/(p(a)∗p(b))$$

==这可以近似理解为合并前后，整个语料的互信息。即，当前选择合并的token对能够让语料的熵最小化->确定性最大化->信息量最小化->在计算机中存储所需要的编码长度最短化。==

所以如果词表中字符a和b本身次数就很高，如果合并ab的概率就算不高（比如0.1）：
- 对于bpe来说ab次数多，需要合并
- 对于wordpiece，ab合并概率低，不合并

tokenizer可以将文本拆分为词或子词（即标记文本）。 🤗 Transformers 中使用的三种主要类型的分词器： Byte-Pair Encoding字节对编码 (BPE)、WordPiece 和 SentencePiece，下面展示哪个模型使用哪种分词器类型的示例。

>在每个模型页面上，您可以查看相关分词器的文档以了解预训练模型使用的分词器类型。 例如，如果我们查看 BertTokenizer，我们可以看到该模型使用 WordPiece

### 1.2 分词规则
分词有多种方式，对于一个句子：
"Don't you love 🤗 Transformers? We sure do."

- 可以按空格分词：
["Don't", "you", "love", "🤗", "Transformers?", "We", "sure", "do."]
- 区分标点：tokens和标点的各种组合会导致模型必须学习的表示数量激增，所以应该予以清理。标点处理后得到：
["Don", "'", "t", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
- 区分缩写：“Don't”代表“do not”，因此最好将其标记为 ["Do", "n't"]。这就是事情开始变得复杂的地方，也是每个模型都有自己的标记器类型的部分原因

根据我们应用于标记文本的规则，为相同的文本生成不同的标记输出。 预训练模型输入必须是，用于标记其训练数据的相同规则的标记输入，这样才能正常执行。

spaCy 和 Moses 是两种流行的基于规则的标记器。 将它们应用到我们的示例中，spaCy 和 Moses 将输出如下内容：
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]

这里使用了空格和标点符号化以及基于规则的标记化，其松散定义为将句子拆分为单词。这种标记化方法非常简单，但是可能会导致大量文本语料库出现问题，生成一个非常大的词汇表（使用的所有唯一单词和标记的集合）。例如，Transformer XL 使用空格和标点符号化，导致词汇量大小为 267,735！

如此大的词汇量迫使模型有一个巨大的嵌入矩阵作为输入和输出层，这会导致内存和时间复杂度的增加。一般来说，==transformers 模型的词汇量很少超过 50,000，尤其是当它们仅在一种语言上进行预训练时==。

那么如果简单的空格和标点符号化不能令人满意，为什么不简单地对字符char进行标记化呢？

### 1.3 character-based-tokenizer
字符标记化往往伴随着性能的损失，使模型学习有意义的输入表示变得更加困难。例如。 学习字母“t”的有意义的上下文比学习单词“today”的上下文无关表示要困难得多。 因此，为了两全其美，transformers 模型使用了词级和字符级标记化之间的混合，称为子词标记化。
### 1.4 Subword tokenization
原则：不应将常用词拆分为更小的子词，而应将稀有词分解为有意义的子词。以通过将子词串在一起来形成（几乎）任意长的复杂词。
例如，“annoyingly”可能被认为是一个罕见的词，可以分解为“annoying”和“ly”。 “annoying”和“ly”作为独立的子词出现的频率会更高，同时“annoyingly”的意思被“annoying”和“ly”的复合词所保持。
- 子词标记化允许模型具有合理的词汇量
- 同时能够学习有意义的上下文无关表示
- 使模型能够通过将它们分解为已知的子词来处理它以前从未见过的词

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("I have a new GPU!")
["i", "have", "a", "new", "gp", "##u", "!"]
```
我们可以看到单词 ["i", "have", "a", "new"] 出现在分词器的词汇表中，但单词“gpu”却没有。 因此，分词器将“gpu”拆分为已知的子词：[“gp”和“##u”]。 “##”表示令牌的其余部分应附加到前一个，没有空格（用于解码或逆转令牌化）。

再举一个例子，XLNetTokenizer 将我们之前的示例文本分词如下：

```python
from transformers import XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do.")
["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]
```
SentencePiece：将罕见词"Transformers" 拆分成更常见的子词 "Transform" 和 "ers".

现在让我们看看不同的子词标记化算法是如何工作的。

### 1.5 Byte-Pair Encoding字节对编码 (BPE)
- Byte-Pair Encoding (BPE) 是Neural Machine Translation引入的（具有罕见词的子词units）。依赖于将训练数据拆分为word的pre-tokenizer。预标记化可以像空格化（space tokenization）一样简单，就像GPT-2, Roberta。
- 更高级的预标记化包括基于规则的标记化，例如XLM、FlauBERT（在大多数语言中使用 Moses）或 GPT（使用 Spacy 和 ftfy）来计算训练语料库中每个单词的频率。
- 在预标记化之后：
	- 根据训练数据形成一系列唯一token及其出现频率
	- BPE 创建一个由唯一单词集中的所有符号组成的基本词汇表
	- 学习合并规则以从基本词汇表的两个符号形成一个新符号。直到词汇量达到所需的词汇量大小。请注意，==所需的词汇量是在训练分词器之前定义的超参数==。

举个例子，让我们假设在预标记化之后，已经确定了以下一组单词，包括它们的频率：

```python
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```
1. 基本词汇是 ["b", "g", "h", "n", "p", "s", "u"]。 将所有单词拆分为基本词汇表的符号，我们得到：

```python
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```
2. BPE 计算每个可能的符号对的频率并选择出现频率最高的符号对。最频繁的符号对是“u”后跟“g”，总共出现 10 + 5 + 5 = 20 次，因此，分词器学习的第一个合并规则是将所有“u”符号和后跟“g”符号组合在一起。 接下来，将“ug”添加到词汇表中。 这组词然后变成：

```python
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```
3. BPE 识别下一个最常见的符号对：“u”后跟“n”16 次。 "u", "n" 合并到 "un" 并添加到词汇表中。 下一个最频繁的符号对是“h”后跟“ug”，出现 15 次。 这对再次合并，并且可以将“hug”添加到词汇表中。

在这个阶段，词汇是 ["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"] 和我们的 一组独特的词表示为：

```python
("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)
```
4. 假设字节对编码训练将在此时停止，然后将学习到的合并规则应用于新单词。例如"bug"分词成 ["b", "ug"]，但是"mug" 分词成 ["<unk>", "ug"]。因为词汇表不包含：“m”。

如前所述，词汇量大小，即基本词汇量大小 + 合并次数（base vocabulary size + the number of merges），是一个可供选择的超参数。 例如，GPT 的词汇量是 40,478，因为它们有 478 个基本字符，并且在 40,000 次合并后选择停止训练。

### 1.6 字节级 BPE（Byte-level BPE）
GPT-2 使用字节作为基础词汇，这是一个巧妙的技巧，可以强制基础词汇的大小为 256，同时确保每个基础字符都包含在词汇中。再加上一些额外的标点符号处理规则，GPT-2的分词器不需要\<unk>符号。
GPT-2 的词汇量大小为 50,257，对应于 256 字节的基本标记、一个特殊的文本结束标记和通过 50,000 次合并学习的符号。

### 1.7 WordPiece
WordPiece 是用于 BERT、DistilBERT 和 Electra 的子词标记化算法，与 BPE 非常相似。 WordPiece 首先初始化词汇表以包含训练数据中存在的每个字符，并逐步学习给定数量的合并规则。==与 BPE 相比，WordPiece 不选择最频繁的符号对，而是选择将训练数据添加到词汇表中的可能性最大化的符号对==。

那么这到底是什么意思呢？参考前面的例子，最大化训练数据的似然性相当于找到符号对，其概率除以其第一个符号后跟第二个符号的概率在所有符号对中最大。例如。只有当“ug”除以“u”、“g”的概率大于任何其他符号对时，“u”和“g”才会被合并。直观地说，WordPiece 与 BPE 略有不同，它通过合并两个符号来评估它的损失，以确保it’s worth it。

### 1.8 Unigram
- 基本词汇表可以对应于所有预先标记的单词和最常见的子串
- 删除了损失增加最低的符号，重复这个过程，直到词汇量达到所需的大小。

Unigram 是在 Subword 正则化：与 BPE 或 WordPiece 相比，==Unigram 将其基本词汇表初始化为大量符号，并逐步缩减每个符号以获得较小的词汇表。例如，基本词汇表可以对应于所有预先标记的单词和最常见的子串==。 Unigram 不直接用于transformers中的任何模型，但它与 SentencePiece 结合使用。

在每个训练步骤中，Unigram 算法在给定当前词汇和 unigram 语言模型的情况下定义训练数据的损失（通常定义为对数似然）。然后，==对于词汇表中的每个符号，算法计算如果要从词汇表中删除该符号，总体损失会增加多少。然后 Unigram 删除了损失增加最低的符号的 概率p（通常为 10% 或 20%），即那些对训练数据的整体损失影响最小的符号。重复这个过程，直到词汇量达到所需的大小。== Unigram 算法始终保留基本字符，以便可以对任何单词进行标记。

由于 Unigram 不基于合并规则（与 BPE 和 WordPiece 不同），因此该算法有多种方法可以在训练后对新文本进行标记。例如，如果经过训练的 Unigram 分词器展示词汇表：

```python
["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"],
```
“hug”可以标记为 ["hug", "s"], ["h", "ug", "s"] 或 ["h", "u", "g", "s"]。 那么该选择哪一个呢？ Unigram 在保存词汇的基础上还保存了训练语料库中每个标记的概率，以便在训练后计算每个可能的标记化的概率。 该算法在实践中只是简单地选择最可能的标记化，但也提供了根据概率对可能的标记化进行采样的可能性。

这些概率由分词器训练的损失定义。 假设训练数据由单词 x1,…,xN 组成，并且单词 xi 的所有可能标记的集合被定义为 S(xi)，那么总损失定义为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/93a7af17c6b14b92ba23d3f6e59ceccc.png)
### 1.9 SentencePiece
到目前为止描述的所有标记化算法都有相同的问题：==假设输入文本使用空格来分隔单词==。
但是，并非所有语言都使用空格来分隔单词，例如中文、日文和泰文。
一种可能的解决方案是使用特定于语言的预分词器，例如XLM 使用特定的预分词器。[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al., 2018)](https://arxiv.org/pdf/1808.06226.pdf) 将输入视为原始输入流，thus including the space in the set of characters to use.然后它使用 BPE 或 unigram 算法来构建适当的词汇表。

例如，XLNetTokenizer 中使用的 SentencePiece，解码非常容易，因为所有标记都可以连接起来，并且“-” 被空格替换。 SentencePiece 和unigram 结合使用，包括 ALBERT、XLNet、Marian 和 T5。



本文参考：[how_to_train.ipynb](https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)

## 一、训练分词器
### 1.1 Using tokenizers from 🤗 Tokenizers
>参考文档：[HF文档](https://huggingface.co/transformers/fast_tokenizers.html)

PreTrainedTokenizerFast 依赖于 tokenizers 库。 从 🤗 Tokenizers 库中获得的分词器可以非常简单地加载到 🤗 Transformers 中。
在详细介绍之前，让我们首先在几行中创建一个虚拟标记器：
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.pre_tokenizer = Whitespace()
files = [...]
tokenizer.train(files, trainer)
```
现在有了一个我们定义的标记器，可以继续使用，或者将它保存到一个 JSON 文件中以备将来重用。
- 直接以tokenizer object使用

```python
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```
- json文件加载使用

```python
tokenizer.save("tokenizer.json")
#我们保存此文件的路径可以使用 tokenizer_file 参数传递给 PreTrainedTokenizerFast 初始化方法：
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```
### 1.2 Train your tokenizer
Transformers Notebooks——[How to train and use your very own tokenizer](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)
#### 1.2.1 从头训练分词器
给定语料库上训练分词器，进而从头训练transformer模型。 在[tokenizers summary](https://huggingface.co/transformers/tokenizer_summary.html) 中可以查看子词分词算法之间的差异（也就是上一节内容）。

下面举例使用wikitext数据集（包含 4.5MB 的文本，所以我们的例子训练速度很快）训练分词器：

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

dataset
Dataset({
    features: ['text'],
    num_rows: 36718
})

dataset[:5]
{'text': ['',
  ' = Valkyria Chronicles III = \n',
  '',
  ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . \n',
  " The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . \n"]}
```
训练我们的分词器的 API 将需要一批文本的迭代器，例如文本列表：
```python
batch_size = 1000
all_texts = [dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)]
```
为了避免将所有内容加载到内存中（因为 Datasets 库将元素保存在磁盘上并且仅在请求时将它们加载到内存中），我们定义了一个 Python 迭代器来进行批处理：

```python
def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
```
接下来有两种方法训练分词器：
1. 使用现有的分词器，一行代码就可以在给定数据集上训练新的分词器
2. 逐块构建分词器，因此可以自定义每一步 ！

#### 1.2.2 使用已有的分词器训练
如果您想使用与现有算法完全相同的算法和参数来训练一个分词器，您可以只使用 train_new_from_iterator API。 例如，让我们使用相同的标记化算法在 Wikitext-2 上训练新版本的 GPT-2 tokenzier。

首先，我们需要加载我们想要用作模型的tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

>确保您选择的标记器是快速版本（由 🤗 Tokenizers 库支持），否则 notebook 的其余部分将无法运行：

```python
tokenizer.is_fast
True
```
然后我们将训练语料库（list of list或我们之前定义的迭代器）提供给 train_new_from_iterator 方法。 我们还必须指定要使用的词汇量大小：

```python
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=25000)
```
到此就完成了分词器的训练。由于使用了Rust 支持的 🤗 Tokenizers 库，训练进行得非常快。
您现在有一个新的标记器可以预处理您的数据并训练语言模型。 您可以像往常一样输入输入文本：

```python
new_tokenizer(dataset[:5]["text"])
{'input_ids': [[], [238, 8576, 9441, 2987, 238, 252], [], [4657, 74, 4762, 826, 8576, 428, 466, 609, 6881, 412, 204, 9441, 311, 2746, 466, 10816, 168, 99, 150, 192, 112, 14328, 3983, 112, 4446, 94, 18288, 4446, 193, 3983, 98, 3983, 22171, 95, 19, 201, 6374, 209, 8576, 218, 198, 3455, 1972, 428, 310, 201, 5099, 3242, 227, 281, 8576, 9441, 2987, 2553, 1759, 201, 301, 196, 13996, 1496, 277, 2330, 1464, 674, 1898, 307, 742, 3541, 225, 7514, 14, 54, 719, 274, 198, 4777, 15522, 209, 19895, 221, 1341, 1633, 221, 1759, 201, 322, 301, 198, 1368, 674, 221, 198, 8576, 843, 209, 2468, 1795, 223, 198, 1049, 9595, 218, 13996, 225, 1563, 277, 582, 6493, 281, 457, 14371, 201, 198, 1422, 3373, 7452, 227, 198, 455, 674, 225, 4687, 198, 239, 21976, 239, 201, 196, 21657, 1680, 3773, 5591, 198, 4196, 218, 4679, 427, 661, 198, 3518, 1288, 220, 1051, 516, 889, 3947, 1922, 2500, 225, 390, 2065, 744, 872, 198, 7592, 3773, 239, 1975, 251, 208, 89, 22351, 239, 209, 252], [261, 674, 959, 1921, 221, 1462, 201, 7600, 547, 196, 1178, 4753, 218, 198, 630, 3591, 263, 8576, 9441, 1180, 209, 1831, 322, 7568, 198, 3621, 2240, 218, 198, 843, 201, 322, 471, 9575, 5291, 16591, 967, 201, 781, 281, 1815, 198, 674, 604, 10344, 1252, 274, 843, 664, 3147, 320, 209, 13290, 8751, 8124, 2528, 6023, 74, 235, 225, 7445, 10040, 17384, 241, 11487, 8950, 857, 1835, 340, 1382, 22582, 201, 1008, 296, 8576, 9441, 1180, 2436, 21134, 5337, 19463, 5161, 209, 240, 1178, 927, 218, 3776, 8650, 198, 3355, 209, 261, 674, 268, 83, 2511, 3472, 258, 8288, 307, 1010, 268, 78, 209, 252]], 'attention_mask': [[], [1, 1, 1, 1, 1, 1], [], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```
保存模型：

```python
new_tokenizer.save_pretrained("my-new-tokenizer")

('my-new-tokenizer/tokenizer_config.json',
 'my-new-tokenizer/special_tokens_map.json',
 'my-new-tokenizer/vocab.json',
 'my-new-tokenizer/merges.txt',
 'my-new-tokenizer/added_tokens.json',
 'my-new-tokenizer/tokenizer.json')
```
之后可以加载此分词器：

```python
tok = new_tokenizer.from_pretrained("my-new-tokenizer")
```

或者推送到 Hugging Face Hub 以从任何地方使用这个新的 tokenzier，具体操作参考[此处](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)。

#### 1.2.3 从头构建分词器
如果你想创建和训练一个新的标记器，它看起来不像现有的任何东西，你需要使用 🤗 Tokenizers 库从头开始构建它.

要了解如何从头开始构建标记器，我们必须深入了解 🤗 Tokenizers 库和标记化管道。 此管道需要几个步骤：
- Normalization：对初始输入字符串执行所有初始转换。 例如，当您需要小写某些文本时，可能会将其剥离，甚至应用一种常见的 unicode 规范化过程，您将添加一个 Normalizer。
- Pre-tokenization：负责分割初始输入字符串。 这是决定在何处以及如何对原始字符串进行预分段的组件。 最简单的例子是使用空格进行分割。

```python
pre_tokenizers??

BertPreTokenizer = pre_tokenizers.BertPreTokenizer
ByteLevel = pre_tokenizers.ByteLevel
CharDelimiterSplit = pre_tokenizers.CharDelimiterSplit
Digits = pre_tokenizers.Digits
Metaspace = pre_tokenizers.Metaspace
Punctuation = pre_tokenizers.Punctuation
Sequence = pre_tokenizers.Sequence
Split = pre_tokenizers.Split
UnicodeScripts = pre_tokenizers.UnicodeScripts
Whitespace = pre_tokenizers.Whitespace
WhitespaceSplit = pre_tokenizers.WhitespaceSplit
```

- model：处理所有sub-token的发现和生成，这是可训练且真正依赖于您的输入数据的部分。

```python
models??

BPE = models.BPE
Unigram = models.Unigram
WordLevel = models.WordLevel
WordPiece = models.WordPiece
```

- 后处理Post-Processing：提供与一些基于 Transformers 的 SoTA 模型兼容的高级构建功能。 例如，对于 BERT，它会将标记化的句子包裹在 [CLS] 和 [SEP] 标记周围。

```python
processors??

BertProcessing = processors.BertProcessing
ByteLevel = processors.ByteLevel
RobertaProcessing = processors.RobertaProcessing
TemplateProcessing = processors.TemplateProcessing
```

- 解码Decoding：负责将标记化的输入映射回原始字符串。 通常根据我们之前使用的 PreTokenizer 来选择解码器。

```python
decoders??

ByteLevel = decoders.ByteLevel
WordPiece = decoders.WordPiece
Metaspace = decoders.Metaspace
BPEDecoder = decoders.BPEDecoder
```

对于模型的训练，🤗 Tokenizers 库提供了一个我们将使用的 Trainer 类。
trainers??

BPE = trainers.BPE
Unigram = trainers.Unigram
WordLevel = trainers.WordLevel
WordPiece = trainers.WordPiece



所有这些构建块都可以组合起来创建tokenization pipelines。 下面将展示三个完整的管道：GPT-2、BERT 和 T5（它将为你提供 BPE、WordPiece 和 Unigram 标记器的示例）。

##### 1.2.3.2 WordPiece model like BERT
创建一个 WordPiece 标记器（like BERT）：
1. 创建一个带有空 WordPiece 模型的 Tokenizer：

```python
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))
```
2. 添加normalization（可选）

```python
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
#如果你想自定义它，你可以使用现有的块并按顺序组合它们：例如，我们小写，应用 NFD 规范化并去除重音：
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
```
3. 添加pre-tokenizer（分词）

```python
#直接使用 BertPreTokenizer，它使用空格和标点符号预先标记：
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
```

与 normalizer 一样，我们可以在一个 Sequence 中组合多个 pre-tokenizer。 如果我们想快速了解它如何预处理输入，我们可以调用 pre_tokenize_str 方法：

```python
tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!")
[('This', (0, 4)),
 ('is', (5, 7)),
 ('an', (8, 10)),
 ('example', (11, 18)),
 ('!', (18, 19))]
```
请注意，==pre-tokenizer 不仅将文本拆分为单词，还保留了偏移量==，即原始文本中每个单词的开头和开头。 这将使最终的分词器==能够将每个标记与它来自的文本部分进行匹配（我们用于问答或标记分类任务的功能）==。

4. 构建 post-processor，传递special tokens给trainer。
#直接使用WordPieceTrainer

```python
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
```
5. 构建数据集（text files）或批处理工具（batches of texts）：

```python
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
```
6. 分词器已经训练完毕，定义后处理器：开头添加 CLS 标记并在末尾添加 SEP 标记（对于单个句子）或几个 SEP 标记（对于句子对）。 可以使用 [TemplateProcessing](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.processors.TemplateProcessing) 来做到这一点。

```python
#获取CLS 和SEP 的token id

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

#使用TemplateProcessing构建后处理器
#在模板中指明如何用一个句子（$A）或两个句子（$A 和 $B）组织特殊标记。
#后跟一个数字表示要赋予每个部分的token type ID，也就是哪部分是第一句，哪部分是第二句。
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),],
)
```
下面编码一个句子看看结果：

```python
encoding = tokenizer.encode("This is one sentence.", "With this one we have a pair.")
encoding.tokens

['[CLS]',
 'this',
 'is',
 'one',
 'sentence',
 '.',
 '[SEP]',
 'with',
 'this',
 'one',
 'we',
 'have',
 'a',
 'pair',
 '.',
 '[SEP]']

encoding.type_ids

[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```
7. 解码器：我们使用 WordPiece 解码器并指示特殊前缀 ##：

```python
tokenizer.decoder = decoders.WordPiece(prefix="##")
```
现在我们的tokenizer已经完成，我们必须将它放在与我们要使用的模型相对应的标记器 fast 类中，这里是一个 BertTokenizerFast：

```python
from transformers import BertTokenizerFast

new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
```
- 和以前一样，我们可以将此分词器用作普通的 Transformers 分词器，并使用 save_pretrained 或 push_to_hub 方法。

- 如果您正在构建的分词器与 Transformers 中的任何类都不匹配(分词器非常特殊)，您可以将它包装在 PreTrainedTokenizerFast 中。

##### 1.2.3.3 BPE model like GPT-2
下面看看如何创建一个 BPE 标记器（like GPT-2 tokenizer）：
1. 创建一个带有初始 BPE model的 Tokenizer：

```python
tokenizer = Tokenizer(models.BPE())
```
2. 添加可选normalization（GPT2不使用）
3. 指定pre-tokenizer（GPT2使用byte level pre-tokenizer）

```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```
调用 pre_tokenize_str 方法，快速了解它如何预处理输入：

```python
tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!")
[('This', (0, 4)),
 ('Ġis', (4, 7)),
 ('Ġan', (7, 10)),
 ('Ġexample', (10, 18)),
 ('!', (18, 19))]
```
我们对前缀空格使用 GPT-2的默认值，所以除了第一个单词之外，每个单词的开头都添加了一个首字母“Ġ”。

4. 使用 BpeTrainer训练分词器：

```python
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
```
5. 添加后处理和解码器：

```python
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()
```
6. 将此分词器包装在 Transformers tokenizer object中：

```python
from transformers import GPT2TokenizerFast

new_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
```
##### 1.2.3.4 Unigram model like Albert
现在让我们看看如何创建一个 Unigram 分词器(类似 T5 的分词器）：
1. 创建 初始Unigram 模型的 Tokenizer：

```python
tokenizer = Tokenizer(models.Unigram())
```
2. 添加normalization 和pre-tokenizer（Metaspace pre-tokenizer：它用一个特殊字符（默认为“▁” ）替换所有空格，然后在该字符上拆分。）

```python
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.Replace("``", '"'), normalizers.Replace("''", '"'), normalizers.Lowercase()]
)
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
```
调用 pre_tokenize_str 方法，快速了解它如何预处理输入：

```python
tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!")
[('▁This', (0, 4)), ('▁is', (4, 7)), ('▁an', (7, 10)), ('▁example!', (10, 19))]
```
每个单词都在开头添加了一个首字母“ ▁”，这是由 sentencepiece完成的。

3. 使用 UnigramTrainer训练分词器，并设置unknown token。

```python
trainer = trainers.UnigramTrainer(vocab_size=25000, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"], unk_token="<unk>")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
```
4. 添加后处理和解码器（Metaspace，类似pre-tokenizer）

```python
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)
tokenizer.decoder = decoders.Metaspace()
```
5. 将此分词器包装在 Transformers tokenizer object中：

```python
from transformers import AlbertTokenizerFast

new_tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)
```

现在可以用新的tokenizer训练模型了。

- 使用新的分析器在notebook上从头训练模型
- 在l[anguage modeling scripts](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling) 上使用tokenizer_name参数来从头训练模型。
## 二、HF模型预训练方式
使用HF主页的tokenizer和MLM包，进行trainer训练
### 1.加载数据集：
选择多语言多语料数据集[OSCAR corpus](https://traces1.inria.fr/oscar/)
```python
# in this notebook we'll only get one of the files (the Oscar one) for the sake of simplicity and performance
!wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt
```
### 2.训练tokenizer
选择字节级别byte-level BPE分词器（类似GPT2使用的），比BERT的WordPiece（字符级别BPE分词器，切分成子词）好处是几乎不会有未登录词"\<unk> tokens"。

```python
# 安装transformers和tokenizers
!pip install git+https://github.com/huggingface/transformers
!pip list | grep -E 'transformers|tokenizers'
# transformers version at notebook update --- 2.11.0
# tokenizers version at notebook update --- 0.8.0rc1
```




```python
%%time 
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
paths = [str(x) for x in Path(".").glob("**/*.txt")]

# tokenizer初始化
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
```
#### 2.2 分词器的训练参数如下：
```python
#BPE的分词器
classtokenizers.trainers.BpeTrainer(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[], limit_alphabet=None, initial_alphabet=[], continuing_subword_prefix=None, end_of_word_suffix=None)
```
- vocab_size (int, optional) – 最终词汇的大小，包括所有标记和字母表。
- min_frequency (int, optional) – 为了合并，一对应该具有的最小频率。
- show_progress (bool, optional) – 训练时是否显示进度条。
- special_tokens (List[Union[str,AddedToken]], optional) – 模型应该知道的特殊标记列表。
- limit_alphabet (int, optional) – 字母表中保留的最大不同字符数。
- initial_alphabet (List[str], optional) – 包含在初始字母表中的字符列表，即使在训练数据集中没有出现。 如果字符串包含多个字符，则仅保留第一个字符。
- continue_subword_prefix (str, optional) -- 用于每个不是词开头的子词的前缀。
- end_of_word_suffix (str, optional) – 用于每个词尾的子词的后缀。

```python
#WordPiece分词器，参数和上一个相同
classtokenizers.trainers.WordPieceTrainer(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[], limit_alphabet=None, initial_alphabet=[], continuing_subword_prefix='##', end_of_word_suffix=None)
```
- vocab_size (int, optional) – 最终词汇的大小，包括所有标记和字母表。
- min_frequency (int, optional) – 为了合并，一对应该具有的最小频率。
- show_progress (bool, optional) – 训练时是否显示进度条。
- special_tokens (List[Union[str,AddedToken]], optional) – 模型应该知道的特殊标记列表。
- limit_alphabet (int, optional) – 字母表中保留的最大不同字符数。
- initial_alphabet (List[str], optional) – 包含在初始字母表中的字符列表，即使在训练数据集中没有出现。 如果字符串包含多个字符，则仅保留第一个字符。
- continue_subword_prefix (str, optional) -- 用于每个不是词开头的子词的前缀。
- end_of_word_suffix (str, optional) – 用于每个词尾的子词的后缀。

#### 2.3 分词器保存和加载
将训练好的分词器保存在EsperBERTo文件夹：

```python
!mkdir EsperBERTo
tokenizer.save_model("EsperBERTo")
```
最终得到两个分词器文件：
- EsperBERTo/vocab.json：vocab.json，按频率排列的常见token的列表
- EsperBERTo/merges.txt'： merges.txt，merges列表

```python
{ "<s>": 0,"<pad>": 1,"</s>": 2,"<unk>": 3, "<mask>": 4,"!": 5,"\"": 6,"#": 7,
    "$": 8,"%": 9,"&": 10,"'": 11,"(": 12,")": 13, # ...}

# merges.txt
l a
Ġ k
o n
Ġ la
t a
Ġ e
Ġ d
Ġ p
# ...
```
tokenizer针对Esperanto进行了优化，更多单词是a single, unsplit token表示。==我们还以更有效的方式表示序列。 在这个语料库中，编码序列的平均长度比使用预训练的 GPT-2 标记器时小约 30%。==

加载分词器，处理 RoBERTa 特殊标记：
```python
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./EsperBERTo/vocab.json",
    "./EsperBERTo/merges.txt",
)
```

```python
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
```
token_to_id：将给定的token转换为其对应的 id
BertProcessing参数：

```python
classtokenizers.processors.BertProcessing(self, sep, cls)
```

这个后处理器负责添加 Bert 模型所需的特殊标记：
- sep (Tuple[str, int]) – 带有 SEP 令牌的字符串表示及其 id 的元组
- cls (Tuple[str, int]) – 一个带有 CLS 标记的字符串表示的元组，以及它的 id



测试效果：

```python
tokenizer.encode("Mi estas Julien.")
Encoding(num_tokens=7, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
```

```python
tokenizer.encode("Mi estas Julien.").tokens
['<s>', 'Mi', 'Ġestas', 'ĠJuli', 'en', '.', '</s>']
```
### 3.从头开始训练语言模型
参考[run_language_modeling.py](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) 文件。直接设置 [Trainer](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) 选择训练方法。下面以训练类似 [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) 的模型来举例：（相比bert采用动态掩码、舍弃NSP任务，以及更大的训练）

```python
import torch
#定义模型参数
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

#重新创建tokenizer
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)
```
#### 3.2 初始化模型
由于我们是从头开始训练，因此我们仅从配置进行初始化，而不是从现有的预训练模型或检查点进行初始化。

```python
from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)

model.num_parameters()

84095008# => 84 million parameters
```



#### 3.3 创建训练集

由于只有一个text文件，不需要自定义数据集。直接使用LineByLineDataset加载之后用tokenizer预处理。

```python
%%time
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)


CPU times: user 4min 54s, sys: 2.98 s, total: 4min 57s
Wall time: 1min 37s
```
定义data_collator：帮助我们将数据集样本进行批处理的数据整理器。 如果输入的长度不同，则输入会动态填充到批次的最大长度。

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
```

```python
class transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase, mlm: bool = True, mlm_probability: float = 0.15, pad_to_multiple_of: Optional[int] = None, tf_experimental_compile: bool = False, return_tensors: str = 'pt')
```
- tokenizer（PreTrainedTokenizer 或 PreTrainedTokenizerFast）——用于编码数据的标记器。
- mlm (bool, optional, defaults to True) – 是否使用掩码语言建模。 如果设置为 False，则标签与忽略填充标记的输入相同（通过将它们设置为 -100）。 否则，non-masked tokens 的label和masked token的预测值为 -100。（If set to False, the labels are the same as the inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked tokens and the value to predict for the masked token.）
- mlm_probability（浮点数，可选，默认为 0.15）– 当 mlm 设置为 True 时（随机）屏蔽输入中的标记的概率。
- pad_to_multiple_of (int, optional) – 如果设置，则将序列填充为所提供值的倍数。

### 3.4 初始化 Trainer并训练

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
```
开始训练
```python
%%time
trainer.train()

CPU times: user 1h 43min 36s, sys: 1h 3min 28s, total: 2h 47min 4s
Wall time: 2h 46min 46s
TrainOutput(global_step=15228, training_loss=5.762423221226405)
```
保存模型

```python
trainer.save_model("./EsperBERTo")
```
### 5. 检查训练好的模型
除了查看训练和评估损失下降之外，可以通过FillMaskPipeline加载模型进行预测

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask",model="./EsperBERTo",tokenizer="./EsperBERTo")
```

```python
# The sun <mask>.
# =>

fill_mask("La suno <mask>.")
```

```python
[{'score': 0.02119220793247223,
  'sequence': '<s> La suno estas.</s>',
  'token': 316},
 {'score': 0.012403824366629124,
  'sequence': '<s> La suno situas.</s>',
  'token': 2340},
 {'score': 0.011061107739806175,
  'sequence': '<s> La suno estis.</s>',
  'token': 394},
 {'score': 0.008284995332360268,
  'sequence': '<s> La suno de.</s>',
  'token': 274},
 {'score': 0.006471084896475077,
  'sequence': '<s> La suno akvo.</s>',
  'token': 1833}]
```
最后，当你有一个不错的模型时，请考虑与社区分享：

使用 CLI 上传您的模型：transformers-cli upload
写一个 README.md 模型卡并将其添加到 model_cards/ 下的存储库中。 理想情况下，您的模型卡应包括：
- 模型描述
- 训练参数（数据集、预处理、超参数）
- 评估结果
- 预期用途和限制
- 其它有用信息 🤓

