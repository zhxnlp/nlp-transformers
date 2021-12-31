@[toc]
```python
from IPython.display import Image
Image(filename='images/aiayn.png')
```
![Attention is All You Need](https://img-blog.csdnimg.cn/20738c10d0fe48f7854387d840d3946f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

> &#8195;&#8195;本文翻译自[《The Annotated Transformer》](https://nlp.seas.harvard.edu/2018/04/03/attention.html)。 本文主要由Harvard NLP的学者在2018年初撰写，以逐行实现的形式呈现了论文的“注释”版本,对原始论文进行了重排，并在整个过程中添加了评论和注释。本文的note book可以在[篇章2](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.2.1-Pytorch%E7%BC%96%E5%86%99Transformer)下载。
 
&#8195;&#8195;["Attention is All You Need"](https://arxiv.org/abs/1706.03762) 的 Transformer 在过去的一年里一直在很多人的脑海中出现。 Transformer 在机器翻译质量上有重大改进,它还为许多其它NLP 任务提供了一种新的体系结构。论文本身写得很清楚,但传统的看法是论文很难准确的去实现。  

&#8195;&#8195;接下来你首先需要安装[PyTorch](http://pytorch.org/). 完整的 notebook 也可以在[github](https://github.com/harvardnlp/annotated-transformer) 或者 Google [Colab](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sharing)上找到。

&#8195;&#8195;这里的代码主要基于 Harvard NLP的[OpenNMT](http://opennmt.net) 包。 (If helpful feel free to [cite](#conclusion).) 对于模型的其他完整服务实现,请查看 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) (tensorflow) 和 [Sockeye](https://github.com/awslabs/sockeye) (mxnet).

- Alexander Rush ([@harvardnlp](https://twitter.com/harvardnlp) or srush@seas.harvard.edu)


# 预备工作


```python
# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 
```


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
```

# 背景

&#8195;&#8195;减少顺序计算的目的也形成了扩展的神经GPU、ByteNet和VISS2S的基础，所有这些都使用卷积神经网络作为基本构建块，并行计算所有输入和输出位置的隐藏表示。在这些模型中，将来自两个任意输入或输出位置的信号关联起来所需的操作数量随着位置之间的距离而增加，对于ConvS2S是线性增长，对于ByteNet是对数增长。这使得学习远距离位置之间的依赖关系变得更加困难。在transformer中，这被减少到一个恒定的操作数，**尽管由于用attention权重化的位置取平均降低了效果，但是我们可以用多头注意来抵消这种影响。**

&#8195;&#8195;Self-attention，有时称为intra-attention内部注意，是一种将单个序列的不同位置关联起来以计算序列表示的注意机制。Self-attention已经成功地应用于各种任务中，包括阅读理解、摘要生成、文本蕴含以及学习和任务无关的句子表征。端到端记忆网络基于一种循环注意机制，而不是序列对齐的循环，并且已被证明在简单语言问答和语言建模任务上表现良好。

&#8195;&#8195;然而，据我们所知，Transformer 是第一个完全依赖自注意力来计算其输入和输出表示的转换模型，**而不是使用序列对齐RNN或卷积**。

# 模型架构

&#8195;&#8195;大部分神经序列转换模型都有一个编码器-解码器结构 [(引用)](https://arxiv.org/abs/1409.0473)。编码器把一个输入序列$(x_{1},...x_{n})$映射到一个连续的表示$z=(z_{1},...z_{n})$中。解码器对z中的每个元素，生成输出序列$(y_{1},...y_{m})$，一个时间步生成一个元素。在每一步中，模型都是自回归的[(引用)](https://arxiv.org/abs/1308.0850)，在生成下一个结果时，会将先前生成的结果加入输入序列来一起预测。（自回归模型的特点）


```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```


```python
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

Transformer的编码器和解码器都使用self-attention堆叠和point-wise、全连接层。如图1的左、右两边所示。


```python
Image(filename='images/ModalNet-21.png')
```
![Transformer](https://img-blog.csdnimg.cn/e70713a881e74e35b504d9dd3964db20.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


## Encoder and Decoder 堆栈

### Encoder

编码器由N = 6 个完全相同的层组成。


```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```


```python
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

编码器的每个子层（Self Attention 层和 FFNN）都再接一个残差连接[(cite)](https://arxiv.org/abs/1512.03385)。然后是层标准化（layer-normalization） [(cite)](https://arxiv.org/abs/1607.06450)。


```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

&#8195;&#8195;每个子层的输出是$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, 其中 $\mathrm{Sublayer}(x)$ 是子层本身实现的函数。 我们将dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) 应用于每个子层的输出，然后再将其添加到子层输入中并进行归一化。

&#8195;&#8195;为了便于进行残差连接，模型中的所有子层以及embedding层产生的输出的维度都为 $d_{\text{model}}=512$。


```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```

每一层都有两个子层。 第一层是一个multi-head self-attention机制（的层），第二层是一个简单的、全连接的前馈网络。


```python
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### Decoder

解码器也是由N = 6 个完全相同的层组成。  


```python
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

&#8195;&#8195;除了每个decoder层中的两个子层之外，decoder还有第三个子层，该层对encoder的输出执行multi-head attention。（即encoder-decoder-attention层，q向量来自上一层的输入，k和v向量是encoder最后层的输出向量memory）与encoder类似，我们在每个子层再采用残差连接，然后进行层标准化。


```python
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

&#8195;&#8195;我们还修改decoder层中的self-attention子层，以防止在当前位置关注到后面的位置。这种掩码结合将输出embedding偏移一个位置，确保对位置i的预测只依赖位置i之前的已知输出。


```python
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

> 下面的attention mask显示了每个tgt单词（行）允许查看（列）的位置。在训练时将当前单词的未来信息屏蔽掉，阻止此单词关注到后面的单词。


```python

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
None
```
![attention mask](https://img-blog.csdnimg.cn/68291c370c2647cf809898140e39041c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)

### Attention                                                                                                                                                                                                                                                                             
&#8195;&#8195;Attention功能可以描述为将query和一组key-value对映射到输出，其中query、key、value和输出都是向量。输出为value的加权和，其中每个value的权重通过query与相应key的兼容函数来计算。                                                                                                                                                                                                                                                                                  

&#8195;&#8195;我们将particular attention称之为“缩放的点积Attention”(Scaled Dot-Product Attention")。其输入为query、key(维度是$d_k$)以及values(维度是$d_v$)。我们计算query和所有key的点积，然后对每个除以 $\sqrt{d_k}$, 最后用softmax函数获得value的权重。                                                                                                                                                                                                                                 
                                                                                                                                                                     


```python
Image(filename='images/ModalNet-19.png')
```
![ModalNet](https://img-blog.csdnimg.cn/be3311ebcc3c405f9833840e8077550b.png#pic_center)

 
在实践中，我们同时计算一组query的attention函数，并将它们组合成一个矩阵$Q$。key和value也一起组成矩阵$K$和$V$。 我们计算的输出矩阵为：
                                                                 
$$                                                                         
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V               
$$   


```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

&#8195;&#8195;两个最常用的attention函数是加法attention[(cite)](https://arxiv.org/abs/1409.0473)和点积（乘法）attention。除了缩放因子$\frac{1}{\sqrt{d_k}}$ ，点积Attention跟我们的平时的算法一样。加法attention使用具有单个隐层的前馈网络计算兼容函数。虽然理论上点积attention和加法attention复杂度相似，但在实践中，点积attention可以使用高度优化的矩阵乘法来实现，因此点积attention计算更快、更节省空间。                                                                                           

                                                                        
&#8195;&#8195;当$d_k$ 的值比较小的时候，这两个机制的性能相近。当$d_k$比较大时，加法attention比不带缩放的点积attention性能好  [(cite)](https://arxiv.org/abs/1703.03906)。我们怀疑，对于很大的$d_k$值, 点积大幅度增长，将softmax函数推向具有极小梯度的区域。(为了说明为什么点积变大，假设q和k是独立的随机变量，均值为0，方差为1。那么它们的点积$q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, 均值为0方差为$d_k$)。为了抵消这种影响，我们将点积缩小 $\frac{1}{\sqrt{d_k}}$倍。     

&#8195;&#8195;在此引用苏剑林文章[《浅谈Transformer的初始化、参数化与标准化》](https://zhuanlan.zhihu.com/p/400925524?utm_source=wechat_session&utm_medium=social&utm_oi=1400823417357139968&utm_campaign=shareopn)中谈到的，为什么Attention中除以$\sqrt{d}$这么重要？
&#8195;&#8195;Attention的计算是在内积之后进行softmax，主要涉及的运算是$e^{q⋅k}$，我们可以大致认为内积之后、softmax之前的数值在$-3\sqrt{d}$到$3\sqrt{d}$这个范围内，由于d通常都至少是64，所以$e^{3\sqrt{d}}$比较大而$e^{-3\sqrt{d}}$比较小，因此经过softmax之后，Attention的分布非常接近一个one hot分布了，这带来严重的梯度消失问题，导致训练效果差。（例如y=softmax(x)在|x|较大时进入了饱和区，x继续变化y值也几乎不变，即饱和区梯度消失）

&#8195;&#8195;相应地，解决方法就有两个:
- 像NTK参数化那样，在内积之后除以$\sqrt{d}$，使q⋅k的方差变为1，对应$e^3$,$e^{−3}$都不至于过大过小，这样softmax之后也不至于变成one hot而梯度消失了，这也是常规的Transformer如BERT里边的Self Attention的做法
- 另外就是不除以$\sqrt{d}$，但是初始化q,k的全连接层的时候，其初始化方差要多除以一个d，这同样能使得使q⋅k的初始方差变为1，T5采用了这样的做法。

 


```python
Image(filename='images/ModalNet-20.png')
```
![ModalNet-20](https://img-blog.csdnimg.cn/e8dedd89ddd54ca69f00af472970146e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_16,color_FFFFFF,t_70,g_se,x_16#pic_center)


&#8195;&#8195;Multi-head attention允许模型共同关注来自不同位置的不同表示子空间的信息，如果只有一个attention head，它的平均值会削弱这个信息。                 $$    
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O    \\                                           
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)                                
$$                                                                                                                 

&#8195;&#8195;其中映射由权重矩阵完成：$W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。<br>                                                                                                                                             &#8195;&#8195;在这项工作中，我们采用$h=8$个平行attention层或者叫head。对于这些head中的每一个，我们使用$d_k=d_v=d_{\text{model}}/h=64$。由于每个head的维度减小，总计算成本与具有全部维度的单个head attention相似。 


```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

### 模型中Attention的应用                                                                                                                                                     
multi-head attention在Transformer中有三种不同的使用方式：                                                        
- 在encoder-decoder attention层中，queries来自前面的decoder层，而keys和values来自encoder的输出。这使得decoder中的每个位置都能关注到输入序列中的所有位置。这是模仿序列到序列模型中典型的编码器—解码器的attention机制，例如 [(cite)](https://arxiv.org/abs/1609.08144).    


- encoder包含self-attention层。在self-attention层中，所有key，value和query来自同一个地方，即encoder中前一层的输出。在这种情况下，encoder中的每个位置都可以关注到encoder上一层的所有位置。


- 类似地，decoder中的self-attention层允许decoder中的每个位置都关注decoder层中当前位置之前的所有位置（包括当前位置）。 为了保持解码器的自回归特性，需要防止解码器中的信息向左流动。我们在缩放点积attention的内部，通过屏蔽softmax输入中所有的非法连接值（设置为$-\infty$）实现了这一点。                                                                                                                                                                                                                                                     

## 基于位置的前馈网络                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
&#8195;&#8195;除了attention子层之外，我们的编码器和解码器中的每个层都包含一个全连接的前馈网络，该网络在每个层的位置相同（都在每个encoder-layer或者decoder-layer的最后）。该前馈网络包括两个线性变换，并在两个线性变换中间有一个ReLU激活函数。

$$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                        
&#8195;&#8195;尽管两层都是线性变换，但它们在层与层之间使用不同的参数。另一种描述方式是两个内核大小为1的卷积。 输入和输出的维度都是 $d_{\text{model}}=512$, 内层维度是$d_{ff}=2048$。（也就是第一层输入512维,输出2048维；第二层输入2048维，输出512维）


```python
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

## Embeddings and Softmax                                                                                                                                                                                                                                                                                           
&#8195;&#8195;与其他序列转换模型类似，我们使用学习到的embedding将输入token和输出token转换为$d_{\text{model}}$维的向量。我们还使用普通的线性变换和softmax函数将解码器输出转换为预测的下一个token的概率 在我们的模型中，两个嵌入层之间和pre-softmax线性变换共享相同的权重矩阵，类似于[(cite)](https://arxiv.org/abs/1608.05859)。在embedding层中，我们将这些权重乘以$\sqrt{d_{\text{model}}}$。                                                                                                                               


```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

## 位置编码                                                                                                                            
&#8195;&#8195;由于我们的模型不包含循环和卷积，为了让模型利用序列的顺序，我们必须加入一些序列中token的相对或者绝对位置的信息。为此，我们将“位置编码”添加到编码器和解码器堆栈底部的输入embeddinng中。位置编码和embedding的维度相同，也是$d_{\text{model}}$ , 所以这两个向量可以相加。有多种位置编码可以选择，例如通过学习得到的位置编码和固定的位置编码 [(cite)](https://arxiv.org/pdf/1705.03122.pdf)。

&#8195;&#8195;在这项工作中，我们使用不同频率的正弦和余弦函数：                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             $$PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})$$                                                                                                                                                                                                                                                        
&#8195;&#8195;其中$pos$ 是位置，$i$ 是维度。也就是说，位置编码的每个维度对应于一个正弦曲线。 这些波长形成一个从$2\pi$ 到 $10000 \cdot 2\pi$的集合级数。我们选择这个函数是因为我们假设它会让模型很容易学习对相对位置的关注，因为对任意确定的偏移$k$, $PE_{pos+k}$ 可以表示为 $PE_{pos}$的线性函数。

&#8195;&#8195;此外，我们会将编码器和解码器堆栈中的embedding和位置编码的和再加一个dropout。对于基本模型，我们使用的dropout比例是$P_{drop}=0.1$。
                                                                                                                                                                                                                                                    



```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
```

> 如下图，位置编码将根据位置添加正弦波。波的频率和偏移对于每个维度都是不同的。


```python
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
None
```
![Positional Encoding](https://img-blog.csdnimg.cn/6ba09324272b45b39f40b246ff841ae3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


&#8195;&#8195;我们还尝试使用学习的位置embeddings[(cite)](https://arxiv.org/pdf/1705.03122.pdf)来代替固定的位置编码，结果发现两种方法产生了几乎相同的效果。于是我们选择了正弦版本，因为它可能允许模型外推到，比训练时遇到的序列更长的序列。

## 完整模型

> 在这里，我们定义了一个从超参数到完整模型的函数。


```python
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```


```python
# Small example model.
tmp_model = make_model(10, 10, 2)
None
```

# 训练

本节描述了我们模型的训练机制。

> 我们在这快速地介绍一些工具，这些工具用于训练一个标准的encoder-decoder模型。首先，我们定义一个批处理对象，其中包含用于训练的 src 和目标句子，以及构建掩码。

## 批处理和掩码


```python
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
```

> 接下来我们创建一个通用的训练和评估函数来跟踪损失。我们传入一个通用的损失函数，也用它来进行参数更新。

## Training Loop


```python
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
```

## 训练数据和批处理
&#8195;&#8195;我们在包含约450万个句子对的标准WMT 2014英语-德语数据集上进行了训练。这些句子使用字节对编码进行编码，源语句和目标语句共享大约37000个token的词汇表。对于英语-法语翻译，我们使用了明显更大的WMT 2014英语-法语数据集，该数据集由 3600 万个句子组成，并将token拆分为32000个word-piece词表。<br>
每个训练批次包含一组句子对，句子对按相近序列长度来分批处理。每个训练批次的句子对包含大约25000个源语言的tokens和25000个目标语言的tokens。

> 我们将使用torch text进行批处理（后文会进行更详细地讨论）。在这里，我们在torchtext函数中创建批处理，以确保我们填充到最大值的批处理大小不会超过阈值（如果我们有8个gpu，则为25000）。


```python
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
```
## 硬件和训练时间                                                                                                                                                                                                  
我们在一台配备8个 NVIDIA P100 GPU 的机器上训练我们的模型。使用论文中描述的超参数的base models，每个训练step大约需要0.4秒。我们对base models进行了总共10万steps或12小时的训练。而对于big models，每个step训练时间为1.0秒，big models训练了30万steps（3.5 天）。
## Optimizer

我们使用Adam优化器[(cite)](https://arxiv.org/abs/1412.6980)，其中 $\beta_1=0.9$, $\beta_2=0.98$并且$\epsilon=10^{-9}$。我们根据以下公式在训练过程中改变学习率：                                         
$$                                                                                                                                                                                                                                                                                         
lrate = d_{\text{model}}^{-0.5} \cdot                                                                                                                                                                                                                                                                                                
  \min({step\_num}^{-0.5},                                                                                                                                                                                                                                                                                                  
    {step\_num} \cdot {warmup\_steps}^{-1.5})                                                                                                                                                                                                                                                                               
$$                                                                                                                                                                                             
这对应于在第一次$warmup\_steps$步中线性地增加学习速率，并且随后将其与步数的平方根成比例地减小。我们使用$warmup\_steps=4000$。                            

> 注意：这部分非常重要。需要使用此模型设置进行训练。


```python

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```


> 以下是此模型针对不同模型大小和优化超参数的曲线示例。


```python
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None), 
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
```
![不同模型大小和优化超参数的曲线示例](https://img-blog.csdnimg.cn/e874f28ed920446098fc8426ac460578.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_16,color_FFFFFF,t_70,g_se,x_16#pic_center)


##正则化                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                      
### 标签平滑

在训练过程中，我们使用的label平滑的值为$\epsilon_{ls}=0.1$ [(cite)](https://arxiv.org/abs/1512.00567)。这让模型不易理解，因为模型学得更加不确定，但提高了准确性和BLEU得分。

> 我们使用KL div损失实现标签平滑。我们没有使用one-hot独热分布，而是创建了一个分布，we create a distribution that has confidence of the correct word and the rest of the smoothing mass distributed throughout the vocabulary。该分布具有对正确单词的“置信度”和分布在整个词汇表中的“平滑”质量的其余部分。（这句后半段不会啊）


```python
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
```

> Here we can see an example of how the mass is distributed to the words based on confidence. 在这里，我们可以看到一个示例，说明质量如何根据置信度分配给单词。

![Example of label smoothing](https://img-blog.csdnimg.cn/8c47df3df8404f0d8d243f5ab3d93996.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)

```python
#Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
         Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
None
```

> Label smoothing actually starts to penalize the model if it gets very confident about a given choice. 如果模型对给定的选择非常有信心，标签平滑实际上会开始惩罚模型。


```python
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    #print(predict)
    return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
None
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/88ef479dd2194c95b077dfc506b8b620.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 首个实例

> 我们可以从尝试一个简单的复制任务开始。给定来自小词汇表的一组随机输入符号symbols，目标是生成这些相同的符号。

## Synthetic Data合成数据


```python
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
```

## Loss Computation损失计算


```python
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
```

## Greedy Decoding贪婪解码


```python
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))
```

    Epoch Step: 1 Loss: 3.023465 Tokens per Sec: 403.074173
    Epoch Step: 1 Loss: 1.920030 Tokens per Sec: 641.689380
    1.9274832487106324
    Epoch Step: 1 Loss: 1.940011 Tokens per Sec: 432.003378
    Epoch Step: 1 Loss: 1.699767 Tokens per Sec: 641.979665
    1.657595729827881
    Epoch Step: 1 Loss: 1.860276 Tokens per Sec: 433.320240
    Epoch Step: 1 Loss: 1.546011 Tokens per Sec: 640.537198
    1.4888023376464843
    Epoch Step: 1 Loss: 1.682198 Tokens per Sec: 432.092305
    Epoch Step: 1 Loss: 1.313169 Tokens per Sec: 639.441857
    1.3485562801361084
    Epoch Step: 1 Loss: 1.278768 Tokens per Sec: 433.568756
    Epoch Step: 1 Loss: 1.062384 Tokens per Sec: 642.542067
    0.9853351473808288
    Epoch Step: 1 Loss: 1.269471 Tokens per Sec: 433.388727
    Epoch Step: 1 Loss: 0.590709 Tokens per Sec: 642.862135
    0.5686767101287842
    Epoch Step: 1 Loss: 0.997076 Tokens per Sec: 433.009746
    Epoch Step: 1 Loss: 0.343118 Tokens per Sec: 642.288427
    0.34273059368133546
    Epoch Step: 1 Loss: 0.459483 Tokens per Sec: 434.594030
    Epoch Step: 1 Loss: 0.290385 Tokens per Sec: 642.519464
    0.2612409472465515
    Epoch Step: 1 Loss: 1.031042 Tokens per Sec: 434.557008
    Epoch Step: 1 Loss: 0.437069 Tokens per Sec: 643.630322
    0.4323212027549744
    Epoch Step: 1 Loss: 0.617165 Tokens per Sec: 436.652626
    Epoch Step: 1 Loss: 0.258793 Tokens per Sec: 644.372296
    0.27331129014492034
    

> 为了简单起见，此代码使用贪婪解码来预测翻译。


```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
```

    
        1     2     3     4     5     6     7     8     9    10
    [torch.LongTensor of size 1x10]
    
    

# 真实场景示例

> 现在我们考虑一个使用IWSLT德语-英语翻译任务的真实示例。这个任务比论文中考虑的WMT任务小得多，但这个任务也能说明整个（翻译）系统。我们还展示了如何使用多GPU处理，使任务能真正快速地训练。


```python
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
```

## 加载数据
> 我们将使用torchtext和spacy加载数据集来进行tokenization。 


```python
# For data loading.
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
```

> Batching matters a ton for speed. We want to have very evenly divided batches, with absolutely minimal padding. To do this we have to hack a bit around the default torchtext batching. This code patches their default batching to make sure we search over enough sentences to find tight batches. 

批处理对训练速度非常重要。我们希望有非常均匀的批次，绝对最小的填充。为此，我们必须对默认的torchtext批处理进行一些修改。此代码修补了torchtext的默认批处理，以确保我们通过搜索足够的句子来找到稳定的批处理。

## Iterators迭代器


```python
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
```

## 多GPU训练（这一段需要检查，中英文都有）

> Finally to really target fast training, we will use multi-gpu. This code implements multi-gpu word generation. It is not specific to transformer so I won't go into too much detail. The idea is to split up word generation at training time into chunks to be processed in parallel across many different gpus. We do this using pytorch parallel primitives:


最后，为了真正达到快速训练的效果，我们将使用多个GPU。此代码实现了多GPU单词生成，即在训练时将单词生成分成多个块，以便在许多不同 GPU上并行处理。由于它不是针对transformer的，所以不会详细介绍。<br>
我们使用pytorch并行来做到这一点：
* replicate - split modules onto different gpus.
* scatter - split batches onto different gpus
* parallel_apply - apply module to batches on different gpus
* gather - pull scattered data back onto one gpu. 
* nn.DataParallel - a special module wrapper that calls these all before evaluating. 

* replicate -将模块拆分到不同的 GPU 上。
* scatter -将批次拆分到不同的 GPU 上。
* parallel_apply - 将模块应用于不同 GPU 上的批次
* gather - 将分散的数据拉回到一个 GPU 上。
* nn.DataParallel - 一个特殊的模块包装器，在评估之前调用以上所有这些参数。


```python
# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
```

> Now we create our model, criterion, optimizer, data iterators, and paralelization。
> 现在我们创建我们的模型、criterion、优化器、数据迭代器和并行化。


```python
# GPUs to use
devices = [0, 1, 2, 3]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
None
```

> Now we train the model. I will play with the warmup steps a bit, but everything else uses the default parameters.  On an AWS p3.8xlarge with 4 Tesla V100s, this runs at ~27,000 tokens per second with a batch size of 12,000 


>现在我们训练模型。我将稍微试一下warmup steps，但其他一切都使用默认参数。在带有4个Tesla V100 的 AWS p3.8xlarge 上，以每秒27,000 个tokens的速度运行，batch size=12,000。

## Training the System


```python
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
```


```python
if False:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model_par, 
                  MultiGPULossCompute(model.generator, criterion, 
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                          MultiGPULossCompute(model.generator, criterion, 
                          devices=devices, opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt")
```

> 一旦训练，我们就可以对模型进行解码以生成一组翻译。这里我们简单地翻译验证集中的第一句话。这个数据集非常小，所以贪婪搜索的翻译结果相当准确。 


```python
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
```

    Translation:	<unk> <unk> . In my language , that means , thank you very much . 
    Target:	<unk> <unk> . It means in my language , thank you very much . 
    

# Additional Components: BPE, Search, Averaging
# 附加组件：BPE、搜索、平均

> So this mostly covers the transformer model itself. There are four aspects that we didn't cover explicitly. We also have all these additional features implemented in [OpenNMT-py](https://github.com/opennmt/opennmt-py).
> 以上内容主要涵盖了transformer模型本身，但其实还有四个附加功能我们没有涉及。不过我们在[OpenNMT-py](https://github.com/opennmt/opennmt-py)中实现了所有这些附加功能。


> 1) BPE/ Word-piece: We can use a library to first preprocess the data into subword units. See Rico Sennrich's [subword-nmt](https://github.com/rsennrich/subword-nmt) implementation. These models will transform the training data to look like this:
> 1) BPE/Word-piece：我们可以使用一个**库**首先将数据预处理为子词单元。可以参见Rico Sennrich的[subword-nmt](https://github.com/rsennrich/subword-nmt) 来实现。这些模型会将训练数据转换为如下所示：

▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .

> 2) Shared Embeddings: When using BPE with shared vocabulary we can share the same weight vectors between the source / target / generator. See the [(cite)](https://arxiv.org/abs/1608.05859) for details. To add this to the model simply do this:
> 2) Embeddings共享：当使用带有共享词汇的BPE时，我们可以在**源/目标/生成器**之间共享相同的权重向量。详细信息可参考[(cite)](https://arxiv.org/abs/1608.05859)。要将其添加到模型中，只需执行以下操作：


```python
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight
```

> 3) Beam Search：这有点太复杂了，无法在这里介绍。有关pytorch实现，请参阅[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Beam.py)。


> 4) Model Averaging:The paper averages the last k checkpoints to create an ensembling effect. We can do this after the fact if we have a bunch of models:
> 4) Model Averaging:论文对最后k个检查点进行平均以得到集成效果。如果我们有一堆模型，我们可以这样做：


```python
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
```

# 结果

&#8195;&#8195;在WMT 2014 英语-德语翻译任务中,big transformer模型(表2中的Transformer (big)) 比之前报道的最佳模型(包括集成模型)高出2.0 BLEU以上, 新的最高BLEU分数为28.4。该模型的配置列于表3的底部，模型在8个P100 GPU上训练3.5天。即使是我们的基础模型，其表现也超过了之前的模型和集成模型，且模型训练成本比之前模型要小的多。<br>

&#8195;&#8195;在WMT 2014英语-法语翻译任务中，我们的big model的BLEU得分达到41.0分，优于之前发布的所有单一模型，训练成本不到之前最先进模型的1/4。英语-法语翻译任务训练的Transformer (big)模型使用的丢弃率为Pdrop = 0.1,而不是0.3。


```python
Image(filename="images/results.png")
```
![results](https://img-blog.csdnimg.cn/8c74622228af45fc8d0083a3deb553db.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

> The code we have written here is a version of the base model. There are fully trained version of this system available here  [(Example Models)](http://opennmt.net/Models-py/).
>
> With the addtional extensions in the last section, the OpenNMT-py replication gets to 26.9 on EN-DE WMT. Here I have loaded in those parameters to our reimplemenation. 

> 我们在这里编写的代码是一个基本模型版。在此也提供该系统的完全训练版[(Example Models)](http://opennmt.net/Models-py/)。
>
> 通过上一节讲的几个附加功能，OpenNMT-py在EN-DE WMT上得分可以达到26.9。下面，我将这些参数加载到我的代码重现中。



```python
!wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt
```

```python
model, SRC, TGT = torch.load("en-de-model.pt")
```

```python

```


```python
model.eval()
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask, 
                    max_len=60, start_symbol=TGT.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TGT.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)
```

    Translation:	<s> ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden . 
    

## Attention 可视化

> 就算使用贪婪解码，翻译效果看起来也不错。我们可以进一步将其可视化，以查看注意力的每一层发生了什么


```python
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
            tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()

```

Encoder Layer 2
    
![Encoder Layer 2](https://img-blog.csdnimg.cn/220598d1e865445ca44f93e4113c395f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

 Encoder Layer 4
 
![Encoder Layer 4](https://img-blog.csdnimg.cn/be73258f2eb640cc9bcfafa96c19fccb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

Encoder Layer 6
 ![Encoder Layer 6](https://img-blog.csdnimg.cn/acbee41b87d44caebf2586b209fabc88.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

Decoder Self Layer 2
 
![Decoder Self Layer 2](https://img-blog.csdnimg.cn/63f8e33cdb0a44e3b2fa9d4319d5c3fe.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


![Decoder Src Layer 2](https://img-blog.csdnimg.cn/a8e6c7968a234288afe70143b64cf155.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



![Decoder Self Layer 4](https://img-blog.csdnimg.cn/9a455d6363ec47b482b470c1e73adeca.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



  Decoder Src Layer 4
  ![Decoder Src Layer 4](https://img-blog.csdnimg.cn/35c5c54653e34344931b3302cbc6ce98.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

  
  Decoder Self Layer 6

![Decoder Self Layer 6](https://img-blog.csdnimg.cn/63674004887642eb84c09d39953a468d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


  Decoder Src Layer 6
   
![Decoder Src Layer 6](https://img-blog.csdnimg.cn/c5c0049a88694ed8a0315e099ede89a2.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 结论

> 希望这段代码对未来的研究有用。如果您发现此代码有帮助，还可以查看其他OpenNMT工具。如果您有任何问题，请联系：

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

> Cheers,
> srush

{::options parse_block_html="true" /}
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://harvard-nlp.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
                            

<div id="disqus_thread"></div>
<script>
    /**
     *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
     *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
     */
    /*
    var disqus_config = function () {
        this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
        this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };
    */
    (function() {  // REQUIRED CONFIGURATION VARIABLE: EDIT THE SHORTNAME BELOW
        var d = document, s = d.createElement('script');
        
        s.src = 'https://EXAMPLE.disqus.com/embed.js';  // IMPORTANT: Replace EXAMPLE with your forum shortname!
        
        s.setAttribute('data-timestamp', +new Date());
        (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>

