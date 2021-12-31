# task0：自然语言处理之PyTorch情感分析简介

> 注：本期组队学习仅适用于需要PyTorch 1.8或以上版本的torchtext 0.9或以上版本。如果你使用的是torchtext 0.8，请点击[这里](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/torchtext08)。

本期学习使用的软件和版本为：[Pytorch1.8](https://github.com/pytorch/pytorch)，[torchtext0.9](https://github.com/pytorch/text) 和Python3.7

本期组队学习主要从一下几个方面进行学习：

- 利用RNN进行情感二分类
- 利用RNN的各种变体，如LSTM, BiLSTM等进行情感二分类
- 利用更快的模型FastText进行情感二分类
- 利用CNN进行情感二分类
- 情感多分类
- 利用BERT进行情感分类
    
前两个Text将介绍情感分析的常用方法：递归神经网络（RNN）；第三个Text介绍了[FastText](https://arxiv.org/abs/1607.01759)模型；最后一个task的学习覆盖一个[卷积神经网络](https://arxiv.org/abs/1408.5882)（CNN）模型。

还有两个额外的“附录”。第一部分介绍如何使用torchtext加载自己的数据集，第二部分简要介绍torchtext提供的经过预训练的单词嵌入。这部分自由学习，在组队学习中不做要求。

## 环境配置

①要安装Pytorch，请参阅[Pytorch网站](https://pytorch.org/get-started/locally)上的安装说明。

②要安装torchtext，请执行以下操作：

```bas
pip install torchtext
```

③若安装速度较慢，可改为以下命令：

```ba
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchtext
```

④此外，我们还将使用spaCy来标记数据。要安装spaCy，可以按照[spaCy官网](https://spacy.io/usage)的指令来安装，或者执行以下命令：

```猛击
python -m venv .env
.env\Scripts\activate
pip install -U pip setuptools wheel
pip install -U spacy[transformers,lookups]
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

⑤对于Taxt6，我们将使用transformers库，可以通过以下方式安装（更改为清华源）：

```猛击
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers
```

这些教程是使用的transformers版本为4.3。

## 组队学习基本内容

* task1- [情感分析baseline](https://github.com/datawhalechina/team-learning-nlp/blob/master/Emotional_Analysis/task1%20%EF%BC%9A%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90baseline.ipynb) 

这一章主要介绍PyTorch with torchtext项目的工作流。我们将学习如何：加载数据、创建训练/测试/验证拆分、构建词汇表、创建数据迭代器、定义模型以及实现训练/评估/测试循环。该模型将简单但是性能较差，可以将其看作一个Baseline，可以用于学习整个情感分析的处理过程，在后续教程中我们将对此模型进行改进。

* task2-[Updated情感分析 ](https://github.com/datawhalechina/team-learning-nlp/blob/master/Emotional_Analysis/task2%EF%BC%9AUpdated%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90%20.ipynb) 

现在我们已经学习了情感分析的基本工作流程，下面我们将学习如何改进模型：使用压缩填充序列、加载和使用预先训练word embedding、不同的优化器、不同的RNN体系结构、双向RNN、多层（又称深层）RNN和正则化。

* task3-[Faster情感分析](https://github.com/datawhalechina/team-learning-nlp/blob/master/Emotional_Analysis/task3%EF%BC%9AFaster%20%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90.ipynb) 

在我们介绍了使用RNN的升级版本的情感分析之后，我们将研究一种不使用RNN的不同方法：我们将实现论文 [《Bag of Tricks for Efficient Text Classification》](https://arxiv.org/abs/1607.01759)中的模型，该论文已经放在了教程中，感兴趣的小伙伴可以参考一下。这个简单的模型实现了与第二节中的升级的情感分析相当的性能，但训练速度要快得多。

* task4-[卷积情感分析](https://github.com/datawhalechina/team-learning-nlp/blob/master/Emotional_Analysis/task4%EF%BC%9A%E5%8D%B7%E7%A7%AF%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90%20.ipynb) 

接下来，我们将介绍用于情绪分析的卷积神经网络（CNN）。该模型将是[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)的实现。

* task5-[多模型融合情绪分析](https://github.com/datawhalechina/team-learning-nlp/blob/master/Emotional_Analysis/task5%EF%BC%9A%E5%A4%9A%E7%B1%BB%E5%88%AB%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90.ipynb) 

这一章，我们将使用包含以上两种模型的处理形式，这在NLP中很常见。我们将使用Text4中的CNN模型和一个包含6个分类的新数据集。

* task6-[使用Transformers进行情感分析](https://github.com/datawhalechina/team-learning-nlp/blob/master/Emotional_Analysis/task6：Transformers情感分析.ipynb) n
这一章，我们将学习如何使用transformers库加载预训练的transformer模型，实现论文[BERT：Pre-training of Deep Bidirectional Transfoemers for Language Understanding](https://arxiv.org/abs/1810.04805)中的BERT模型（该论文也以放入教程中），并使用它完成文本的embeddings。这些embeddings可以输入到任何模型中来预测情绪，在这里，我们使用了一个门循环单元（GRU）。

## 拓展（待更新……）

* A-[在自己的数据集上使用torchtext]() 

因为本教程使用的数据集为TorchText的内置数据集，附录A说明了如何使用TorchText加载自己的数据集。

* B-[再看word embedding]() 

通过使用TorchText提供的预训练word embedding来查看类似单词，以及实现一个简单的，完全基于word embedding的拼写错误校正器。

* C-[加载、保存和固定Word embedding]() 

我们知道，在NLP领域，预训练语言模型已经发挥着越来越大的作用，在本附录中，我们将介绍：如何加载自定义单词嵌入，如何在训练我们的模型时固定和解除word embedding，以及如何保存我们学到的embedding，以便它们可以在其他模型中使用。

## 参考资料

* http://anie.me/On-Torchtext/
* http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
* https://github.com/spro/practical-pytorch
* https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
* https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
* https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py
* https://github.com/Shawn1993/cnn-text-classification-pytorch
* https://github.com/bentrevett/pytorch-sentiment-analysis




```python

```
