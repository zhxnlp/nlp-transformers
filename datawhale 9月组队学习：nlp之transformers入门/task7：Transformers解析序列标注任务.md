# Transformers解析序列标注任务
本文主要来自datawhale的[transformer教程4.2](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.2-%E5%BA%8F%E5%88%97%E6%A0%87%E6%B3%A8)和[天国之影学习笔记](https://relph1119.github.io/my-team-learning/#/transformers_nlp28/task07)。
## 1 序列标注任务简介

- 序列标注可以看作时token级别的分类问题，为文本中的每一个token预测一个标签
- token级别的分类任务：
  1. NER（Named-entity recognition 名词-实体识别）分辨出文本中的名词和实体（person人名, organization组织机构名, location地点名...）
  2. POS（Part-of-speech tagging词性标注）根据语法对token进行词性标注（noun名词、verb动词、adjective形容词...）
  3. Chunk（Chunking短语组块）将同一个短语的tokens组块放在一起
  
&#8195;&#8195;只要预训练的transformer模型最顶层有一个token分类的神经网络层（比如上一篇章提到的BertForTokenClassification,需要对应的预训练模型有fast tokenizer这个功能，参考[这个表](https://huggingface.co/transformers/index.html#bigtable)），那么本notebook理论上可以使用各种各样的transformer模型（[模型面板](https://huggingface.co/models)），解决任何token级别的分类任务。

&#8195;&#8195;如果您所处理的任务有所不同，大概率只需要很小的改动便可以使用本notebook进行处理。同时，您应该根据您的GPU显存来调整微调训练所需要的btach size大小，避免显存溢出。

```python
# 设置分类任务
task = "ner" 
# 设置BERT模型
model_checkpoint = "distilbert-base-uncased"
# 根据GPU调整batch_size大小，避免显存溢出
batch_size = 16
```
## 2 加载数据
```python
#加载数据和评测方式
from datasets import load_dataset, load_metric
```
&#8195;&#8195;本文使用的是[CONLL 2003 dataset](https://aclanthology.org/W03-0419.pdf)数据集。来处理Datasets库中的任何token分类任务。如果要加载自定义的json/csv文件数据集，可以参考[数据集文档](https://huggingface.co/docs/datasets/loading_datasets.html#from-local-files)来学习如何加载。自定义数据集可能需要在加载属性名字上做一些调整
```python
# 加载conll2003数据集
datasets = load_dataset("conll2003")
```
    Reusing dataset conll2003 (C:\Users\hurui\.cache\huggingface\datasets\conll2003\conll2003\1.0.0\40e7cb6bcc374f7c349c83acd1e9352a4f09474eb691f64f364ee62eb65d0ca6)
    
```python
datasets
```
datasets对象本身是一种DatasetDict数据结构。可以使用对应的key得到相应的数据
```python
    DatasetDict({
        train: Dataset({
            features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
            num_rows: 14041
        })
        validation: Dataset({
            features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
            num_rows: 3250
        })
        test: Dataset({
            features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
            num_rows: 3453
        })
    })
#label列对应tokens的标注
```

```python
# 查看训练集第一条数据
datasets["train"][0]
```
```python
    {'id': '0',
     'tokens': ['EU',
      'rejects',
      'German',
      'call',
      'to',
      'boycott',
      'British',
      'lamb',
      '.'],
     'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7],
     'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0],
     'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]}
```
&#8195;&#8195;所有的数据标签labels都已经被编码成了整数，可以直接被预训练transformer模型使用。这些整数的编码所对应的实际类别储存在features中。
```python
# 查看features属性
datasets["train"].features[f"ner_tags"]
```
    Sequence(feature=ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)

&#8195;&#8195;以NER为例，0对应的标签类别是”O“， 1对应的是”B-PER“等等。具体标签含义对应如下：
- PER：person
- ORG：organization
- LOC：location
- MISC：miscellaneous
- O：没有特别实体（no special entity）
- B-\*：实体开始的token
- I-\*：实体中间的token

```python
label_list = datasets["train"].features[f"{task}_tags"].feature.names
label_list
```
    ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
定义下面的函数，从数据集里随机选择几个例子进行展示
```python
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    """从数据集中随机选择几条数据"""
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))
```
```python
show_random_elements(datasets["train"])
```
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>tokens</th>
      <th>pos_tags</th>
      <th>chunk_tags</th>
      <th>ner_tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4143</td>
      <td>[The, 85-year-old, nun, said, in, the, past, that, she, was, praying, for, the, couple, ,, whose, divorce, is, expected, to, become, final, next, week, .]</td>
      <td>[DT, JJ, NN, VBD, IN, DT, NN, IN, PRP, VBD, VBG, IN, DT, NN, ,, WP\$, NN, VBZ, VBN, TO, VB, JJ, JJ, NN, .]</td>
      <td>[B-NP, I-NP, I-NP, B-VP, B-PP, B-NP, I-NP, B-SBAR, B-NP, B-VP, I-VP, B-PP, B-NP, I-NP, O, B-NP, I-NP, B-VP, I-VP, I-VP, I-VP, B-NP, I-NP, I-NP, O]</td>
      <td>[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2442</td>
      <td>[2., Marie-Jose, Perec, (, France, ), 49.72]</td>
      <td>[CD, NNP, NNP, (, NNP, ), CD]</td>
      <td>[B-NP, I-NP, I-NP, O, B-NP, O, B-NP]</td>
      <td>[O, B-PER, I-PER, O, B-LOC, O, O]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1090</td>
      <td>[There, were, no, significant, differences, between, the, groups, receiving, garlic, and, placebo, ,, ", they, wrote, in, the, Journal, of, the, Royal, College, of, Physicians, .]</td>
      <td>[EX, VBD, DT, JJ, NNS, IN, DT, NNS, VBG, NN, CC, NN, ,, ", PRP, VBD, IN, DT, NNP, IN, DT, NNP, NNP, IN, NNPS, .]</td>
      <td>[B-NP, B-VP, B-NP, I-NP, I-NP, B-PP, B-NP, I-NP, B-VP, B-NP, I-NP, I-NP, O, O, B-NP, B-VP, B-PP, B-NP, I-NP, B-PP, B-NP, I-NP, I-NP, B-PP, B-NP, O]</td>
      <td>[O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, B-ORG, I-ORG, I-ORG, I-ORG, I-ORG, I-ORG, I-ORG, O]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1972</td>
      <td>[Pakistan, first, innings]</td>
      <td>[NNP, RB, NN]</td>
      <td>[B-NP, B-ADVP, B-NP]</td>
      <td>[B-LOC, O, O]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13714</td>
      <td>[The, Taiwan, dollar, closed, slightly, firmer, on, Thursday, amid, tight, Taiwan, dollar, liquidity, in, the, banking, system, ,, and, dealers, said, the, rate, was, likely, to, move, narrowly, in, the, near, term, .]</td>
      <td>[DT, NNP, NN, VBD, RB, JJR, IN, NNP, IN, JJ, NNP, NN, NN, IN, DT, NN, NN, ,, CC, NNS, VBD, DT, NN, VBD, JJ, TO, VB, RB, IN, DT, JJ, NN, .]</td>
      <td>[B-NP, I-NP, I-NP, B-VP, B-ADVP, B-ADJP, B-PP, B-NP, B-PP, B-NP, I-NP, I-NP, I-NP, B-PP, B-NP, I-NP, I-NP, O, O, B-NP, B-VP, B-NP, I-NP, B-VP, B-ADJP, B-VP, I-VP, I-VP, B-PP, B-NP, I-NP, I-NP, O]</td>
      <td>[O, B-LOC, O, O, O, O, O, O, O, O, B-LOC, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4806</td>
      <td>[nine, of, the, superbike, world, championship, on, Sunday, :]</td>
      <td>[CD, IN, DT, JJ, NN, NN, IN, NNP, :]</td>
      <td>[B-NP, B-PP, B-NP, I-NP, I-NP, I-NP, B-PP, B-NP, O]</td>
      <td>[O, O, O, O, O, O, O, O, O]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7452</td>
      <td>[The, accident, happened, when, the, Sanchez, Zarraga, family, took, their, boat, out, for, a, nighttime, spin, ,, Civil, Defence, and, Coast, Guard, officials, said, .]</td>
      <td>[DT, NN, VBD, WRB, DT, NNP, NNP, NN, VBD, PRP\$, NN, RP, IN, DT, NN, NN, ,, NNP, NN, CC, NNP, NNP, NNS, VBD, .]</td>
      <td>[B-NP, I-NP, B-VP, B-ADVP, B-NP, I-NP, I-NP, I-NP, B-VP, B-NP, I-NP, B-ADVP, B-PP, B-NP, I-NP, I-NP, O, B-NP, I-NP, O, B-NP, I-NP, I-NP, B-VP, O]</td>
      <td>[O, O, O, O, O, B-PER, I-PER, O, O, O, O, O, O, O, O, O, O, B-ORG, I-ORG, I-ORG, I-ORG, I-ORG, O, O, O]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2332</td>
      <td>[7., Julie, Baumann, (, Switzerland, ), 13.36]</td>
      <td>[NNP, NNP, NNP, (, NNP, ), CD]</td>
      <td>[B-NP, I-NP, I-NP, O, B-NP, O, B-NP]</td>
      <td>[O, B-PER, I-PER, O, B-LOC, O, O]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9786</td>
      <td>[The, pilot, said, several, hijackers, appeared, to, be, placed, around, the, plane, .]</td>
      <td>[DT, NN, VBD, JJ, NNS, VBD, TO, VB, VBN, IN, DT, NN, .]</td>
      <td>[B-NP, I-NP, B-VP, B-NP, I-NP, B-VP, I-VP, I-VP, I-VP, B-PP, B-NP, I-NP, O]</td>
      <td>[O, O, O, O, O, O, O, O, O, O, O, O, O]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3451</td>
      <td>[(, 7-4, ), 6-2]</td>
      <td>[(, CD, ), CD]</td>
      <td>[B-LST, B-NP, O, B-NP]</td>
      <td>[O, O, O, O]</td>
    </tr>
  </tbody>
</table>

## 3 预处理数据
### 3.1 数据预处理流程
- 数据预处理工具：Tokenizer
- 流程：
  1. 对输入数据进行tokenize，得到tokens
  2. 将tokens转化为预训练模型中需要对应的token ID
  3. 将token ID转化为模型需要的输入格式
&#8195;&#8195;为了达到数据预处理的目的，我们使用AutoTokenizer.from_pretrained方法实例化我们的tokenizer，这样可以确保：
- 我们得到一个与预训练模型一一对应的tokenizer。
- 使用指定的模型checkpoint对应的tokenizer的时候，我们也下载了模型需要的词表库vocabulary，准确来说是tokens vocabulary。
- 这个被下载的tokens vocabulary会被缓存起来，从而再次使用的时候不会重新下载
### 3.2 构建模型对应的tokenizer
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```
&#8195;&#8195;以下代码要求tokenizer必须是transformers.PreTrainedTokenizerFast类型，因为我们在预处理的时候需要用到fast tokenizer的一些特殊特性（比如多线程快速tokenizer）。在这里[big table of models](https://huggingface.co/transformers/index.html#bigtable)查看模型是否有fast tokenizer。
&#8195;&#8195;tokenizer既可以对单个文本进行预处理，也可以对一对文本进行预处理，tokenizer预处理后得到的数据满足预训练模型输入格式
```python
import transformers
# 模型使用的时fast tokenizer
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
```
```python
tokenizer("Hello, this is one sentence!")
```
    {'input_ids': [101, 7592, 1010, 2023, 2003, 2028, 6251, 999, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

```python
tokenizer(["Hello", ",", "this", "is", "one", "sentence", "split",
          "into", "words", "."], is_split_into_words=True)
```

    {'input_ids': [101, 7592, 1010, 2023, 2003, 2028, 6251, 3975, 2046, 2616, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

**补充：**
- transformer预训练模型会将切分后的word，继续用tokenizer分词器切分为subword
```python
example = datasets["train"][4]
print(example["tokens"])
```
    ['Germany', "'s", 'representative', 'to', 'the', 'European', 'Union', "'s", 'veterinary', 'committee', 'Werner', 'Zwingmann', 'said', 'on', 'Wednesday', 'consumers', 'should', 'buy', 'sheepmeat', 'from', 'countries', 'other', 'than', 'Britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.']
    
```python
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)
```
    ['[CLS]', 'germany', "'", 's', 'representative', 'to', 'the', 'european', 'union', "'", 's', 'veterinary', 'committee', 'werner', 'z', '##wing', '##mann', 'said', 'on', 'wednesday', 'consumers', 'should', 'buy', 'sheep', '##me', '##at', 'from', 'countries', 'other', 'than', 'britain', 'until', 'the', 'scientific', 'advice', 'was', 'clearer', '.', '[SEP]']
    
单词"Zwingmann" 和 "sheepmeat"继续被切分成了3个subtokens

### 3.3 解决subtokens对齐问题
&#8195;&#8195;由于标注数据通常是在word级别进行标注的，既然word还会被切分成subtokens，那么意味着我们还需要对标注数据进行subtokens的对齐。同时，由于预训练模型输入格式的要求，往往还需要加上一些特殊符号比如： [CLS] 和  [SEP]。

```python
# 使用word_ids方法解决subtokens对齐问题
print(tokenized_input.word_ids())
```
    [None, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, None]
&#8195;&#8195;word_ids将每一个subtokens位置都对应了一个word的下标。比如第1个位置对应第0个word，然后第2、3个位置对应第1个word。特殊字符对应了None。有了这个list，我们就能将subtokens和words还有标注的labels对齐啦。  
```python
# 获取subtokens位置
word_ids = tokenized_input.word_ids()

# 将subtokens、words和标注的labels对齐
aligned_labels = [
    -100 if i is None else example[f"{task}_tags"][i] for i in word_ids]

print(len(aligned_labels), len(tokenized_input["input_ids"]))

39 39#输出结果
```
&#8195;&#8195;我们通常将特殊字符的label设置为-100，在模型中-100通常会被忽略掉不计算loss    

两种对齐label的方式：
- 多个subtokens对齐一个word，对齐一个label
- 多个subtokens的第一个subtoken对齐word，对齐一个label，其他subtokens直接赋予-100.
以上两种方式通过label_all_tokens = True切换
### 3.4 整合预处理函数
将以上所有内容合起来变成我们的预处理函数，is_split_into_words=True在上面已经结束啦（？）
```python
label_all_tokens = True
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        # 获取subtokens位置
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        # 遍历subtokens位置索引
        for word_idx in word_ids:
            if word_idx is None:
                # 将特殊字符的label设置为-100
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        
        # 对齐word
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```
以上的预处理函数可以处理一个样本，也可以处理多个样本exapmles（返回多个样本被预处理之后的结果list）
```python
tokenize_and_align_labels(datasets['train'][:5])
```
    {'input_ids': [[101, 7327, 19164, 2446, 2655, 2000, 17757, 2329, 12559, 1012, 102], [101, 2848, 13934, 102], [101, 9371, 2727, 1011, 5511, 1011, 2570, 102], [101, 1996, 2647, 3222, 2056, 2006, 9432, 2009, 18335, 2007, 2446, 6040, 2000, 10390, 2000, 18454, 2078, 2329, 12559, 2127, 6529, 5646, 3251, 5506, 11190, 4295, 2064, 2022, 11860, 2000, 8351, 1012, 102], [101, 2762, 1005, 1055, 4387, 2000, 1996, 2647, 2586, 1005, 1055, 15651, 2837, 14121, 1062, 9328, 5804, 2056, 2006, 9317, 10390, 2323, 4965, 8351, 4168, 4017, 2013, 3032, 2060, 2084, 3725, 2127, 1996, 4045, 6040, 2001, 24509, 1012, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'labels': [[-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, -100], [-100, 1, 2, -100], [-100, 5, 0, 0, 0, 0, 0, -100], [-100, 0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100], [-100, 5, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, -100]]}

### 3.5 对数据集datasets所有样本进行预处理
&#8195;&#8195;使用map函数将预处理函数应用到（map)所有样本上。
```python
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
```

    Loading cached processed dataset at C:\Users\hurui\.cache\huggingface\datasets\conll2003\conll2003\1.0.0\40e7cb6bcc374f7c349c83acd1e9352a4f09474eb691f64f364ee62eb65d0ca6\cache-fa2382f441f8d16d.arrow
    Loading cached processed dataset at C:\Users\hurui\.cache\huggingface\datasets\conll2003\conll2003\1.0.0\40e7cb6bcc374f7c349c83acd1e9352a4f09474eb691f64f364ee62eb65d0ca6\cache-8057d57320e0ee7a.arrow
    Loading cached processed dataset at C:\Users\hurui\.cache\huggingface\datasets\conll2003\conll2003\1.0.0\40e7cb6bcc374f7c349c83acd1e9352a4f09474eb691f64f364ee62eb65d0ca6\cache-ea32e2b3f93b1edb.arrow
    
    
&#8195;&#8195;返回的结果会自动被缓存，避免下次处理的时候重新计算（但是也要注意，如果输入有改动，可能会被缓存影响！）。datasets库函数会对输入的参数进行检测，判断是否有变化，如果没有变化就使用缓存数据，如果有变化就重新处理。但如果输入参数不变，想改变输入的时候，最好清理调这个缓存。清理的方式是使用load_from_cache_file=False参数。另外，上面使用到的batched=True这个参数是tokenizer的特点，以为这会使用多线程同时并行对输入进行处理。
## 4 微调预训练模型
&#8195;&#8195;既然我们是做seq2seq任务，那么我们需要使用AutoModelForSequenceClassification 这个类。和tokenizer相似，from_pretrained方法同样可以帮助我们下载并加载模型，同时也会对模型进行缓存，就不会重复下载模型啦。
### 4.1 加载分类模型


```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list))
```

    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForTokenClassification: ['vocab_layer_norm.weight', 'vocab_projector.weight', 'vocab_projector.bias', 'vocab_layer_norm.bias', 'vocab_transform.bias', 'vocab_transform.weight']
    - This IS expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForTokenClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    
&#8195;&#8195;由于我们微调的任务是文本分类任务，而我们加载的是预训练的语言模型，所以会提示我们加载模型的时候扔掉了一些不匹配的神经网络参数（比如：预训练语言模型的神经网络head被扔掉了，同时随机初始化了文本分类的神经网络head）

### 4.2 设定训练参数
&#8195;&#8195;==Trainer是一个简单但功能完整的 PyTorch 训练和评估循环，针对 🤗 Transformers 进行了优化==。Trainer训练工具需要3个要素模型、数据集和训练参数。
```python
Trainer(
    model,#如果使用transformer模型，它将是一个transformers.PreTrainedModel类的子类
    args,#训练参数
    data_collator,
    train_dataset,#训练集
    eval_dataset,#测试集
    tokenizer,#分词器
    compute_metrics,#评测方式，评估时计算方式的函数
    model_init: Callable[[], transformers.modeling_utils.PreTrainedModel] = None,
    callbacks: Union[List[transformers.trainer_callback.TrainerCallback], NoneType] = None,#回调函数，用于保存最优模型参数
    optimizers: Tuple[torch.optim.optimizer.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None,None) )#优化器
```
Trainer最重要的是训练参数 TrainingArguments。这个训练设定包含了能够定义训练过程的所有属性。

```python
args = TrainingArguments(
    f"test-{task}",
    # 每个epcoh会做一次验证评估
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    log_level='error',
    logging_strategy="no",
    report_to="none"
)
```
&#8195;&#8195;最后我们需要一个数据收集器data collator，将我们处理好的输入喂给模型。
```python
from transformers import DataCollatorForTokenClassification
# 数据收集器，用于将处理好的数据输入给模型
data_collator = DataCollatorForTokenClassification(tokenizer)
```
### 4.3 设定评估方法
&#8195;&#8195;我们使用seqeval metric来完成评估。将模型预测送入评估之前，我们也会做一些数据后处理：
```python
metric = load_metric("seqeval")
```

```python
#评估的输入是预测和label的list
labels = [label_list[i] for i in example[f"{task}_tags"]]
metric.compute(predictions=[labels], references=[labels])
```
```python
 {'LOC': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 2},
  'ORG': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
  'PER': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1},
  'overall_precision': 1.0,
  'overall_recall': 1.0,
  'overall_f1': 1.0,
  'overall_accuracy': 1.0}
```
对模型预测结果做一些后处理：
- 选择预测分类最大概率的下标
- 将下标转化为label
- 忽略-100所在地方
下面的函数将上面的步骤合并了起来。

```python
import numpy as np

def compute_metrics(p):
    """模型预测"""
    predictions, labels = p
    # 选择预测分类最大概率的下标
    predictions = np.argmax(predictions, axis=2)

    # 将下标转化为label，并忽略-100的位置
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```
&#8195;&#8195;我们计算所有类别总的precision/recall/f1，所以会扔掉单个类别的precision/recall/f1

构造训练器Trainer

```python
# 构造训练器Trainer，将数据/模型/参数传入Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```
### 4.4 训练模型

```python
trainer.train()
```
  <progress value='2634' max='2634' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [2634/2634 02:13, Epoch 3/3]
</div>

<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.062855</td>
      <td>0.925795</td>
      <td>0.937913</td>
      <td>0.931814</td>
      <td>0.983844</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.062855</td>
      <td>0.925795</td>
      <td>0.937913</td>
      <td>0.931814</td>
      <td>0.983844</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>0.062855</td>
      <td>0.925795</td>
      <td>0.937913</td>
      <td>0.931814</td>
      <td>0.983844</td>
    </tr>
  </tbody>
</table><p>


    TrainOutput(global_step=2634, training_loss=0.02493840813546264, metrics={'train_runtime': 133.2372, 'train_samples_per_second': 316.151, 'train_steps_per_second': 19.769, 'total_flos': 511610930296956.0, 'train_loss': 0.02493840813546264, 'epoch': 3.0})

### 4.5 模型评估

```python
trainer.evaluate()
```
<div>

  <progress value='408' max='204' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [204/204 00:07]
</div>

    {'eval_loss': 0.06285537779331207,
     'eval_precision': 0.9257950530035336,
     'eval_recall': 0.9379125181787672,
     'eval_f1': 0.931814392886913,
     'eval_accuracy': 0.983843550923793,
     'eval_runtime': 3.8895,
     'eval_samples_per_second': 835.586,
     'eval_steps_per_second': 52.449,
     'epoch': 3.0}

### 4.6 输出单个类别的precision/recall/f1
&emsp;&emsp;如果想要得到单个类别的precision/recall/f1，我们直接将结果输入相同的评估函数即可：
```python
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results
```
```python
{'LOC': {'precision': 0.9513574660633484,
   'recall': 0.9637127578304049,
   'f1': 0.9574952561669828,
   'number': 2618},
  'MISC': {'precision': 0.8107255520504731,
   'recall': 0.8350934199837531,
   'f1': 0.8227290916366548,
   'number': 1231},
  'ORG': {'precision': 0.8882575757575758,
   'recall': 0.9124513618677043,
   'f1': 0.9001919385796545,
   'number': 2056},
  'PER': {'precision': 0.9778439153439153,
   'recall': 0.9746209624258405,
   'f1': 0.976229778804886,
   'number': 3034},
  'overall_precision': 0.9257950530035336,
  'overall_recall': 0.9379125181787672,
  'overall_f1': 0.931814392886913,
  'overall_accuracy': 0.983843550923793}
```
## 5 总结
&emsp;&emsp;本次任务，主要介绍了用BERT模型解决序列标注任务的方法及步骤，步骤主要分为加载数据、数据预处理、微调预训练模型。在加载数据阶段中，使用CONLL 2003 dataset数据集，并观察实体类别及表示形式；在数据预处理阶段中，对tokenizer分词器的建模，将subtokens、words和标注的labels对齐，并完成数据集中所有样本的预处理；在微调预训练模型阶段，通过对模型参数进行设置，设置seqeval评估方法，并构建Trainner训练器，进行模型训练，对precision、recall和f1值进行评估比较。  
&emsp;&emsp;其中在数据集下载时，需要使用外网方式建立代理。

