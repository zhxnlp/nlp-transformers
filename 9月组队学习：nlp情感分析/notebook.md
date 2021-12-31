本文涉及的jupter notebook在[篇章4代码库中](https://github.com/datawhalechina/learn-nlp-with-transformers/tree/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1)。

也直接使用google colab notebook打开本教程，下载相关数据集和模型。
如果您正在google的colab中打开这个notebook，您可能需要安装Transformers和🤗Datasets库。将以下命令取消注释即可安装。


```python
!pip install transformers datasets
```

    Requirement already satisfied: transformers in /usr/local/lib/python3.7/dist-packages (4.10.2)
    Requirement already satisfied: datasets in /usr/local/lib/python3.7/dist-packages (1.12.1)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers) (4.8.1)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (1.19.5)
    Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.10.3)
    Requirement already satisfied: sacremoses in /usr/local/lib/python3.7/dist-packages (from transformers) (0.0.45)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (5.4.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from transformers) (21.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2019.12.20)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers) (4.62.2)
    Requirement already satisfied: huggingface-hub>=0.0.12 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.0.17)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers) (3.0.12)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from huggingface-hub>=0.0.12->transformers) (3.7.4.3)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->transformers) (2.4.7)
    Requirement already satisfied: dill in /usr/local/lib/python3.7/dist-packages (from datasets) (0.3.4)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.7/dist-packages (from datasets) (3.7.4.post0)
    Requirement already satisfied: fsspec[http]>=2021.05.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (2021.8.1)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.7/dist-packages (from datasets) (2.0.2)
    Requirement already satisfied: multiprocess in /usr/local/lib/python3.7/dist-packages (from datasets) (0.70.12.2)
    Requirement already satisfied: pyarrow!=4.0.0,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (3.0.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from datasets) (1.1.5)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2021.5.30)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (21.2.0)
    Requirement already satisfied: async-timeout<4.0,>=3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (3.0.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (5.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.6.3)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers) (3.5.0)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2018.9)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.15.0)
    Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (7.1.2)
    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (1.0.1)
    

如果您正在本地打开这个notebook，请确保您已经进行上述依赖包的安装。
您也可以在[这里](https://github.com/huggingface/transformers/tree/master/examples/text-classification)找到本notebook的多GPU分布式训练版本。


# 微调预训练模型进行文本分类

我们将展示如何使用 [🤗 Transformers](https://github.com/huggingface/transformers)代码库中的模型来解决文本分类任务，任务来源于[GLUE Benchmark](https://gluebenchmark.com/).

![Widget inference on a text classification task](https://github.com/huggingface/notebooks/blob/master/examples/images/text_classification.png?raw=1)

GLUE榜单包含了9个句子级别的分类任务，分别是：
- [CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability) 鉴别一个句子是否语法正确.
- [MNLI](https://arxiv.org/abs/1704.05426) (Multi-Genre Natural Language Inference) 给定一个假设，判断另一个句子与该假设的关系：entails, contradicts 或者 unrelated。
- [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (Microsoft Research Paraphrase Corpus) 判断两个句子是否互为paraphrases.
- [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) (Question-answering Natural Language Inference) 判断第2句是否包含第1句问题的答案。
- [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) (Quora Question Pairs2) 判断两个问句是否语义相同。
- [RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment) (Recognizing Textual Entailment)判断一个句子是否与假设成entail关系。
- [SST-2](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank) 判断一个句子的情感正负向.
- [STS-B](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) (Semantic Textual Similarity Benchmark) 判断两个句子的相似性（分数为1-5分）。
- [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) (Winograd Natural Language Inference) Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not. 

对于以上任务，我们将展示如何使用简单的Dataset库加载数据集，同时使用transformer中的`Trainer`接口对预训练模型进行微调。


```python
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
```

本notebook理论上可以使用各种各样的transformer模型（[模型面板](https://huggingface.co/models)），解决任何文本分类分类任务。

如果您所处理的任务有所不同，大概率只需要很小的改动便可以使用本notebook进行处理。同时，您应该根据您的GPU显存来调整微调训练所需要的btach size大小，避免显存溢出。


```python
task = "sst2"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
```

## 加载数据

我们将会使用[🤗 Datasets](https://github.com/huggingface/datasets)库来加载数据和对应的评测方式。数据加载和评测方式加载只需要简单使用`load_dataset`和`load_metric`即可。


```python
from datasets import load_dataset, load_metric
```

除了`mnli-mm`以外，其他任务都可以直接通过任务名字进行加载。数据加载之后会自动缓存。


```python


metric = load_metric('glue',task)
```

这个`datasets`对象本身是一种[`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict)数据结构. 对于训练集、验证集和测试集，只需要使用对应的key（train，validation，test）即可得到相应的数据。

为了能够进一步理解数据长什么样子，下面的函数将从数据集里随机选择几个例子进行展示。

每一个文本分类任务所对应的metic有所不同，具体如下:

- for CoLA: [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
- for MNLI (matched or mismatched): Accuracy
- for MRPC: Accuracy and [F1 score](https://en.wikipedia.org/wiki/F1_score)
- for QNLI: Accuracy
- for QQP: Accuracy and [F1 score](https://en.wikipedia.org/wiki/F1_score)
- for RTE: Accuracy
- for SST-2: Accuracy
- for STS-B: [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) and [Spearman's_Rank_Correlation_Coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
- for WNLI: Accuracy

所以一定要将metric和任务对齐

## 数据预处理


```python
import re
import nltk
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

import os
for dirname, _, filenames in os.walk('/transformers'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

os.listdir("./transformers/nlp-getting-started/")
data_dir = "./transformers/"

data = pd.read_csv(data_dir + "nlp-getting-started/train.csv")
test_data = pd.read_csv(data_dir + "nlp-getting-started/test.csv")

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data = data.drop("location", 1)
train_data.target.value_counts() 
```




    0    4342
    1    3271
    Name: target, dtype: int64




```python
def clean_text(text):
    
    new_text = text.lower() # lowercase the text
    new_text = re.sub(r"\w+\:\/\/([a-z]+)\.co\/\w+(\n)?", "", new_text) #remove urls
    new_text = re.sub(r"@[a-zA-Z0-9]+(?:;)*", "", new_text) # remove @s
    new_text = re.sub(r"#", "", new_text) # remove #s
    new_text = re.sub(r"[^a-z0-9A-Z]", " ", new_text) # remove non alphanumerics
    new_text = re.sub(r"[0-9]+[^\w+]", "", new_text) # remove words made wholy of digits
    new_text = re.sub(r"\b\w{1,2}\b", "", new_text) # remove words w/ 1 char
    new_text = re.sub(" +", " ", new_text) # remove multiple consecutive spaces
    
    new_text = new_text.strip() # remove leading/trailing whitespaces
    
    return new_text

train_data['keyword'].fillna('', inplace=True)
test_data['keyword'].fillna('', inplace=True)

train_data['text'] = train_data['text'].map(clean_text)
train_data['keyword'] = train_data['keyword'].map(clean_text)
test_data['text'] = test_data['text'].map(clean_text)
test_data['keyword'] = test_data['keyword'].map(clean_text)
```


```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lm=nltk.stem.WordNetLemmatizer()
all_tokens=[item for _, value in tweets.items() for item in word_tokenize(value)]
#all_tokens_lm=[lm.lemmatize(t) for t in all_tokens]
all_tokens_lm=[lm.lemmatize(t) for t in all_tokens if t not in stopwords.words('english')]

N = len(all_tokens_lm)
V = len(set(all_tokens_lm))
         
print(f"There are {N} tokens after processing")
print(f"There are {V} unique tokens after processing")
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-41c9ff8a385f> in <module>()
          4 nltk.download('wordnet')
          5 lm=nltk.stem.WordNetLemmatizer()
    ----> 6 all_tokens=[item for _, value in tweets.items() for item in word_tokenize(value)]
          7 #all_tokens_lm=[lm.lemmatize(t) for t in all_tokens]
          8 all_tokens_lm=[lm.lemmatize(t) for t in all_tokens if t not in stopwords.words('english')]
    

    NameError: name 'tweets' is not defined



```python
train_data.iloc[:6090]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td></td>
      <td>our deeds are the reason this earthquake may a...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td></td>
      <td>forest fire near ronge sask canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td></td>
      <td>all residents asked shelter place are being no...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td></td>
      <td>people receive wildfires evacuation orders cal...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td></td>
      <td>just got sent this photo from ruby alaska smok...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6085</th>
      <td>8692</td>
      <td>sinking</td>
      <td>you feel like you are sinking low self image t...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6086</th>
      <td>8693</td>
      <td>sinking</td>
      <td>after few years afloat pension plans start sin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6087</th>
      <td>8694</td>
      <td>sinking</td>
      <td>you feel like you are sinking unhappiness take...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6088</th>
      <td>8695</td>
      <td>sinking</td>
      <td>with sinking music video career brooke hogan s...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6089</th>
      <td>8696</td>
      <td>sinking</td>
      <td>feel bad for them can literally feel that feel...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6090 rows × 4 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
train_texts,val_texts,train_keyword,val_keyword,train_labels,val_labels = train_test_split(train_data['text'].tolist(),train_data['keyword'].tolist(),train_data['target'].tolist(),test_size=0.25)



```

在将数据喂入模型之前，我们需要对数据进行预处理。预处理的工具叫`Tokenizer`。`Tokenizer`首先对输入进行tokenize，然后将tokens转化为预模型中需要对应的token ID，再转化为模型需要的输入格式。

为了达到数据预处理的目的，我们使用`AutoTokenizer.from_pretrained`方法实例化我们的tokenizer，这样可以确保：

- 我们得到一个与预训练模型一一对应的tokenizer。
- 使用指定的模型checkpoint对应的tokenizer的时候，我们也下载了模型需要的词表库vocabulary，准确来说是tokens vocabulary。

这个被下载的tokens vocabulary会被缓存起来，从而再次使用的时候不会重新下载。


```python
print(train_data['text'].str.split().str.len().describe())
print(train_data['keyword'].str.split().str.len().describe())
print(test_data['text'].str.split().str.len().describe())
print(test_data['keyword'].str.split().str.len().describe())
```


```python
from transformers import AutoTokenizer
    
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)
```

注意：`use_fast=True`要求tokenizer必须是transformers.PreTrainedTokenizerFast类型，因为我们在预处理的时候需要用到fast tokenizer的一些特殊特性（比如多线程快速tokenizer）。如果对应的模型没有fast tokenizer，去掉这个选项即可。

几乎所有模型对应的tokenizer都有对应的fast tokenizer。我们可以在[模型tokenizer对应表](https://huggingface.co/transformers/index.html#bigtable)里查看所有预训练模型对应的tokenizer所拥有的特点。


```python
training_text_encodings=tokenizer(train_texts,padding=True,truncation=True)
val_text_encodings=tokenizer(val_texts,padding=True,truncation=True)

training_keyword_encodings=tokenizer(train_keyword,padding=True,truncation=True)
val_keyword_encodings=tokenizer(val_keyword,padding=True,truncation=True)
```


```python
class DatasetCls():
    def __init__(self, text_encodings, keyword_encodings, labels):
        self.text_encodings = text_encodings
        self.keyword_encodings = keyword_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {f"text_{key}" : torch.tensor(val[idx]) for key, val in self.text_encodings.items()}
        item.update({f"keyword_{key}" : torch.tensor(val[idx]) for key, val in self.text_encodings.items()})
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```


```python
from torch.utils.data import DataLoader
train_dataset = DatasetCls(training_text_encodings,training_keyword_encodings,train_labels)
val_dataset = DatasetCls(val_text_encodings,val_keyword_encodings,val_labels)

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
eval_loader = DataLoader(val_dataset,batch_size=16)
```

随后将预处理的代码放到一个函数中：


```python
def preprocess_function(examples):    
  return tokenizer(examples['text'],examples['keyword'],truncation=True)
    
```

接下来对数据集datasets里面的所有样本进行预处理，处理的方式是使用map函数，将预处理函数prepare_train_features应用到（map)所有样本上。


```python
train_datasets=preprocess_function(train_set)
#eval_datasets=validation_set.map(preprocess_function,batched=True)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-75-8e57f1a3e6d3> in <module>()
    ----> 1 train_datasets=preprocess_function(train_set)
          2 #eval_datasets=validation_set.map(preprocess_function,batched=True)
    

    <ipython-input-51-0f8c637e0fb9> in preprocess_function(examples)
          1 def preprocess_function(examples):
    ----> 2   return tokenizer(examples['text'],examples['keyword'],truncation=True)
          3 
    

    /usr/local/lib/python3.7/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
       2355         if not _is_valid_text_input(text):
       2356             raise ValueError(
    -> 2357                 "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
       2358                 "or `List[List[str]]` (batch of pretokenized examples)."
       2359             )
    

    ValueError: text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).



更好的是，返回的结果会自动被缓存，避免下次处理的时候重新计算（但是也要注意，如果输入有改动，可能会被缓存影响！）。datasets库函数会对输入的参数进行检测，判断是否有变化，如果没有变化就使用缓存数据，如果有变化就重新处理。但如果输入参数不变，想改变输入的时候，最好清理调这个缓存。清理的方式是使用`load_from_cache_file=False`参数。另外，上面使用到的`batched=True`这个参数是tokenizer的特点，以为这会使用多线程同时并行对输入进行处理。

## 微调预训练模型

既然数据已经准备好了，现在我们需要下载并加载我们的预训练模型，然后微调预训练模型。既然我们是做seq2seq任务，那么我们需要一个能解决这个任务的模型类。我们使用`AutoModelForSequenceClassification` 这个类。和tokenizer相似，`from_pretrained`方法同样可以帮助我们下载并加载模型，同时也会对模型进行缓存，就不会重复下载模型啦。

需要注意的是：STS-B是一个回归问题，MNLI是一个3分类问题：



```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=2)
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_transform.weight', 'vocab_layer_norm.weight', 'vocab_layer_norm.bias', 'vocab_transform.bias', 'vocab_projector.bias', 'vocab_projector.weight']
    - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'pre_classifier.weight', 'pre_classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    


```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 5
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
```


```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


```


      0%|          | 0/1785 [00:00<?, ?it/s]



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-44f4409bc57e> in <module>()
         12     for batch in train_loader:
         13         batch = {k: v.to(device) for k, v in batch.items()}
    ---> 14         outputs = model(**batch)
         15         loss = outputs.loss
         16         loss.backward()
    

    /usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
       1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
       1050                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1051             return forward_call(*input, **kwargs)
       1052         # Do not call functions when jit is used
       1053         full_backward_hooks, non_full_backward_hooks = [], []
    

    TypeError: forward() got an unexpected keyword argument 'text_input_ids'



```python
metric_name="accuracy"

args = TrainingArguments(
    "nlp-getting-started",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)
```

上面evaluation_strategy = "epoch"参数告诉训练代码：我们每个epcoh会做一次验证评估。

上面batch_size在这个notebook之前定义好了。

最后，由于不同的任务需要不同的评测指标，我们定一个函数来根据任务名字得到评价方法:


```python
def compute_metrics(eval_pred):
    predictions, labels=eval_pred
    if task != "stsb":
        predictions=np.argmax(predictions, axis=1)
    else:
        predictions=predictions[:,0]
    return metric.compute(predictions=predictions, references=labels)
```

全部传给 `Trainer`:


```python
#validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=train_loader,
    eval_dataset=eval_loader,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```


```python
import numpy as np
import pandas as pd
```

开始训练:


```python
trainer.train()
```

    ***** Running training *****
      Num examples = 357
      Num Epochs = 5
      Instantaneous batch size per device = 16
      Total train batch size (w. parallel, distributed & accumulation) = 16
      Gradient Accumulation steps = 1
      Total optimization steps = 115
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-21-3435b262f1ae> in <module>()
    ----> 1 trainer.train()
    

    /usr/local/lib/python3.7/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
       1256             self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
       1257 
    -> 1258             for step, inputs in enumerate(epoch_iterator):
       1259 
       1260                 # Skip past any already trained steps if resuming training
    

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py in __next__(self)
        519             if self._sampler_iter is None:
        520                 self._reset()
    --> 521             data = self._next_data()
        522             self._num_yielded += 1
        523             if self._dataset_kind == _DatasetKind.Iterable and \
    

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
        559     def _next_data(self):
        560         index = self._next_index()  # may raise StopIteration
    --> 561         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        562         if self._pin_memory:
        563             data = _utils.pin_memory.pin_memory(data)
    

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]
    

    /usr/local/lib/python3.7/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
         42     def fetch(self, possibly_batched_index):
         43         if self.auto_collation:
    ---> 44             data = [self.dataset[idx] for idx in possibly_batched_index]
         45         else:
         46             data = self.dataset[possibly_batched_index]
    

    TypeError: 'DataLoader' object is not subscriptable


训练完成后进行评估:


```python
trainer.evaluate()
```

    The following columns in the evaluation set  don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: sentence, idx.
    ***** Running Evaluation *****
      Num examples = 1043
      Batch size = 16
    



<div>

  <progress value='66' max='66' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [66/66 00:02]
</div>






    {'epoch': 5.0,
     'eval_loss': 0.7508974671363831,
     'eval_matthews_correlation': 0.5347185537666785,
     'eval_runtime': 2.1314,
     'eval_samples_per_second': 489.347,
     'eval_steps_per_second': 30.965}



To see how your model fared you can compare it to the [GLUE Benchmark leaderboard](https://gluebenchmark.com/leaderboard).


```python
tokenizer.save_pretrained("C/sentences_classfication")
model.save_pretrained("C/sentences_classfication")
```

    tokenizer config file saved in C/sentences_classfication/tokenizer_config.json
    Special tokens file saved in C/sentences_classfication/special_tokens_map.json
    Configuration saved in C/sentences_classfication/config.json
    Model weights saved in C/sentences_classfication/pytorch_model.bin
    


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    

## 超参数搜索

`Trainer`同样支持超参搜索，使用[optuna](https://optuna.org/) or [Ray Tune](https://docs.ray.io/en/latest/tune/)代码库。

反注释下面两行安装依赖：


```python
! pip install optuna
! pip install ray[tune]
```

超参搜索时，`Trainer`将会返回多个训练好的模型，所以需要传入一个定义好的模型从而让`Trainer`可以不断重新初始化该传入的模型：


```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
```

和之前调用 `Trainer`类似:


```python
trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

    loading configuration file https://huggingface.co/distilbert-base-uncased/resolve/main/config.json from cache at /root/.cache/huggingface/transformers/23454919702d26495337f3da04d1655c7ee010d5ec9d77bdb9e399e00302c0a1.d423bdf2f58dc8b77d5f5d18028d7ae4a72dcfd8f468e81fe979ada957a8c361
    Model config DistilBertConfig {
      "activation": "gelu",
      "architectures": [
        "DistilBertForMaskedLM"
      ],
      "attention_dropout": 0.1,
      "dim": 768,
      "dropout": 0.1,
      "hidden_dim": 3072,
      "initializer_range": 0.02,
      "max_position_embeddings": 512,
      "model_type": "distilbert",
      "n_heads": 12,
      "n_layers": 6,
      "pad_token_id": 0,
      "qa_dropout": 0.1,
      "seq_classif_dropout": 0.2,
      "sinusoidal_pos_embds": false,
      "tie_weights_": true,
      "transformers_version": "4.9.1",
      "vocab_size": 30522
    }
    
    loading weights file https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin from cache at /root/.cache/huggingface/transformers/9c169103d7e5a73936dd2b627e42851bec0831212b677c637033ee4bce9ab5ee.126183e36667471617ae2f0835fab707baa54b731f991507ebbb55ea85adb12a
    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_projector.weight', 'vocab_transform.weight', 'vocab_projector.bias', 'vocab_layer_norm.bias', 'vocab_transform.bias', 'vocab_layer_norm.weight']
    - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.weight', 'classifier.weight', 'pre_classifier.bias', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

调用方法`hyperparameter_search`。注意，这个过程可能很久，我们可以先用部分数据集进行超参搜索，再进行全量训练。
比如使用1/10的数据进行搜索：


```python
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
```

`hyperparameter_search`会返回效果最好的模型相关的参数：


```python
best_run
```

将`Trainner`设置为搜索到的最好参数，进行训练：


```python
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
```

最后别忘了，查看如何上传模型 ，上传模型到](https://huggingface.co/transformers/model_sharing.html) 到[🤗 Model Hub](https://huggingface.co/models)。随后您就可以像这个notebook一开始一样，直接用模型名字就能使用您自己上传的模型啦。


```python
zip -q -r checkpoint-2675.zip /content/test-glue/checkpoint-2675
```


      File "<ipython-input-28-bcb0775fb004>", line 1
        zip -q -r checkpoint-2675.zip /content/test-glue/checkpoint-2675
                           ^
    SyntaxError: invalid syntax
    



```python
from google.colab import drive
drive.mount('/content/drive')

```

    Mounted at /content/drive
    


```python
import os
os.chdir("/content/drive/MyDrive")
```


```python

```
