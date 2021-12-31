## 图解GPT
@[toc]
引言：
&#8195;&#8195;本文是datawhale教程的读书笔记，由datawhale翻译自Jay Alammar的文章[《The Illustrated GPT-2 (Visualizing Transformer Language Models)》](http://jalammar.github.io/illustrated-gpt2/)。中文翻译版原文在datawhale课程[《图解GPT》](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.4-%E5%9B%BE%E8%A7%A3GPT?id=gpt-2-%E5%85%A8%E8%BF%9E%E6%8E%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
###  1.GPT简介
&#8195;&#8195;GPT是OpenAI公司2018年提出的生成式预训练模型（Generative Pre-Trainning，GPT），用来提升自然语言理解任务的效果。GPT的出现打破了NLP各个任务之间的壁垒，不需要根据特定任务了解太多任务背景。根据预训练模型就能得到不错的任务效果。
&#8195;&#8195;GPT提出了“生成式预训练+判别式任务精调”的范式来处理NLP任务。

- 生成式预训练：在大规模无监督语料上进行预训练一个高容量的语言模型，学习丰富的上下文信息，掌握文本的通用语义。
- 判别式任务精调：在通用语义基础上根据下游任务进行领域适配。具体的在预训练好的模型上增加一个与任务相关的神经网络层，比如一个全连接层，预测最终的标签。并在该任务的监督数据上进行微调训练（微调的一种理解：学习率较小，训练epoch数量较少，对模型整体参数进行轻微调整）

&#8195;&#8195;GPT是一个自回归语言模型，它可以根据句子的一部分预测下一个词。只能是单向的的建模文本序列（只能顺序或逆序）。如果双向建模会导致顺序建模时未来信息泄露。
&#8195;&#8195; GPT-2 比你手机上的键盘 app 更大更复杂。GPT-2 是在一个 40 GB 的名为 WebText 的数据集上训练的，最小的 GPT-2 变种，需要 500 MB 的空间来存储它的所有参数。最大的 GPT-2 模型变种是其大小的 13 倍，因此占用的空间可能超过 6.5 GB。
&#8195;&#8195;原始的 Transformer 模型是由 Encoder 和 Decoder 组成的，它们都是由 Transformer 堆叠而成的。这种架构是适用于处理机器翻译。而在随后的许多研究工作中，只使用 Transformer 中的一部分，要么去掉 Encoder，要么去掉 Decoder，并且将它们堆得尽可能高，使用大量的文本进行训练。这些模块堆叠的高度，是区分不同的 GPT-2 的主要因素之一。
![不同级别的GPT2](https://img-blog.csdnimg.cn/8f2cfb40410a4ec0a60881313ff5161f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)


###  2.与 BERT 的一个不同之处
&#8195;&#8195;<font color='red'>GPT-2 是使用 Transformer 的 Decoder 模块构建的。BERT 是使用 Encoder 模块。它们之间的一个重要差异是，GPT-2 和传统的语言模型一样，一次输出一个 token。</font>

*（对英文而言subway是token，sub也是toekn，对中文来说每个字就是token。bert预测和训练是一致的，都是字粒度）*
&#8195;&#8195;“自回归（auto-regression）”：这类模型的实际工作方式是，在产生每个 token 之后，将这个 token 添加到输入的序列中，形成一个新序列。然后这个新序列成为模型在下一个时间步的输入，这种做法可以使得 RNN 非常有效。
&#8195;&#8195;GPT-2，和后来的一些模型如 TransformerXL 和 XLNet，本质上都是自回归的模型。但 BERT 不是自回归模型。这是一种权衡。去掉了自回归后，BERT 能够整合左右两边的上下文，从而获得更好的结果。XLNet 重新使用了 自回归，同时也找到一种方法能够结合两边的上下文。
### 3.GPT2结构详解
#### 3.1Transformer - Decoder 
&#8195;&#8195;原始的Transformer - Decoder 与 Encoder 相比，它有一个层，使得它可以关注来自 Encoder 特定的段——encoder-decoder-attention。但是在GPT中，使用的decoder去掉了这一层。其结构如下：
![transformer-decoder](https://img-blog.csdnimg.cn/c793f3546ea345f48836f3128884fdac.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;虽然舍弃了一层，但还是保留了decoder的masked Self Attention结构，屏蔽未来的信息。
&#8195;&#8195;为啥这个结构不叫encoder（只是没有masked）主要还是从功能上决定。encoder是为了编码信息到一个堆vecotr，decoder是为了把一堆信息解码出来。gpt2的训练方式是生成文本，类似解码。bert是用masker-ML训练，是提取特征建立语言模型。
#### 3.2 GPT2训练过程
&#8195;&#8195;GPT-2 能够处理 1024 个 token。每个 token 沿着自己的路径经过所有的 Decoder 模块。运行一个训练好的 GPT-2 模型的最简单的方法是简单地给它输入初始 token让它自己生成文本。
&#8195;&#8195;初始token为 $<s>$，模型依次顺序地生成单词。具体地在每个时刻，token 流过所有decoder层，输出一个向量。输出向量可以根据模型的词汇表计算出一个分数（模型知道所有的 单词，在 GPT-2 中是 5000 个词）
![token](https://img-blog.csdnimg.cn/f7261ba52e8746a59307ccb7cff1d014.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)


&#8195;&#8195;但若是一直选择模型建议的单词，它有时会陷入重复的循环之中，唯一的出路就是点击第二个或者第三个建议的单词。GPT-2 有一个 top-k 参数，可以用来选择top-1之外的其他词。之后，我们把这个输出添加到我们的输入序列，然后让模型做下一个预测（自回归）。
&#8195;&#8195;请注意，**选择top1之外的其他词是此计算中唯一活动的路径**。GPT-2 的每一层都保留了它自己对第一个 token 的解释，而且会在预测第二个 token 时使用它，但是不会反过来根据后面的token重新计算前面的token。
#### 3.3 GPT2的输入
&#8195;&#8195;和BERT输入由三部分组成不同，GPT的输入只有两部分：词嵌入和位置编码。
&#8195;&#8195;首先，GPT-2 在嵌入矩阵中查找输入的单词的对应的 **embedding 向量**，如下图：
![token - embeddings](https://img-blog.csdnimg.cn/2c77f6cef60741ed96442021d69196d9.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;每一行都是词的 embedding：这是一个数字列表，可以表示一个词并捕获一些含义。这个列表的大小在不同的 GPT-2 模型中是不同的。最小的small级别模型使用的 embedding size= 768。
&#8195;&#8195;其次，要给 token - embedding**融入位置编码**。位置编码只有1024个（输入长度限制为1024）这样训练好的模型中，有一部分是一个包括了 1024 个位置编码向量的矩阵。
![posiyional embeddings](https://img-blog.csdnimg.cn/e8e72027c5f746239e11b7a8274bad3a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;**以上结合起来就是词向量+位置编码：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/e04c2648c05b4e41aea3cffdda0a7f55.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

#### 3.4 token在decoder层向上流动
&#8195;&#8195;token现在可以输入第一个decoder模块了。首先通过 Self Attention 层，然后通过神经网络层。之后会得到一个结果向量，这个结果向量再依次输入下一个decoder层被处理。每个decoder模块的处理过程都是相同的，但是每个模块都有自己的 Self Attention 和神经网络层。
![decoder](https://img-blog.csdnimg.cn/9d43fb88c295427ca3688b40df1be8a8.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

**回顾 Self-Attention可以看我的上一篇帖子**
#### 3.5 模型输出
&#8195;&#8195;当模型顶部的模块产生输出向量时（这个向量是经过 Self Attention 层和神经网络层得到的），模型会将这个向量乘以嵌入矩阵。嵌入矩阵中的每一行都对应于模型词汇表中的一个词。<font color='red'>这个相乘的结果，被解释为模型词汇表中每个词的分数logits。(token概率)</font>
![decoder输出](https://img-blog.csdnimg.cn/96cebc08d2c34df796b9258e704a08a0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;我们可以选择最高分数的 token（top_k=1），但最好是同时考虑其他词，比如设置top_k =40。此时分数就是每个词被选择的概率。
&#8195;&#8195;这样，模型就完成了一次迭代，输出一个单词。模型会继续迭代，直到所有的上下文都已经生成（1024 个 token），或者直到输出了表示句子末尾的 token。
&#8195;&#8195;总结起来就是**输入token embedding向量序列流过decoder得到的输出向量和嵌入矩阵相乘，得到选词概率logits。**
#### 3.6 简化说明
&#8195;&#8195;为了讲解方便，文中对一些说法进行了简化：
&#8195;&#8195;文中交替使用 token 和 词。但实际上，GPT-2 使用 Byte Pair Encoding 在词汇表中创建 token。这意味着 token 通常是词的一部分。
&#8195;&#8195;我们展示的例子是在推理模式下运行。这就是为什么它一次只处理一个 token。在训练时，模型将会针对更长的文本序列进行训练，并且同时处理多个 token。同样，在训练时，模型会处理更大的 batch size，而不是推理时使用的大小为 1 的 batch size。
&#8195;&#8195;为了更加方便地说明原理，我在本文的图片中一般会使用行向量。但有些向量实际上是列向量。在代码实现中，你需要注意这些向量的形式。
&#8195;&#8195;Transformer 使用了大量的层归一化（layer normalization），这一点是很重要的。在上一篇图解Transformer中已经提及到了一部分这点，本文中没有特意指出。
&#8195;&#8195;有时我需要更多的框来表示一个向量，例如下面这幅图：
![zoom in](https://img-blog.csdnimg.cn/ad8ac8d6b722462dba1e8dad90dbd8b5.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
### 4. GPT2 的 Self-Attention
&#8195;&#8195;我们可以让 GPT-2 像 mask Self Attention 一样工作。但是在评价评价模型时，当我们的模型在每次迭代后只添加一个新词，那么对于已经处理过的 token 来说，沿着之前的路径重新计算 Self Attention 是低效的。
&#8195;&#8195;在这种情况下，我们处理第一个 token（现在暂时忽略 $<s>$）会生成token a的三种向量。之后GPT-2的每个 Self Attention 层（每个 decoder模块）都会保存 token a 的 Key 向量和 Value 向量。并在在以后的计算中被使用，而不需要重新生成。（Query向量已经不需要了，**只有计算token a时的才使用Query，Query向量只使用一次**）（每个decoder模块都有它自己的权重，第一次权重矩阵生成的QKV向量是随机初始化的向量，没有经过计算？）
![计算robort时刻](https://img-blog.csdnimg.cn/31f85c3eefb44db3bcc899f910de1572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;上图表示a的Key 、 Value 被保存，进入后面robort的计算。

&#8195;&#8195;再举个例子，假设模型正在处理单词 it。此时 对应的输入就是 it 的 embedding 加上第 9 个位置的位置编码。
#### 4.2 Self-Attention层详细过程
**(1) 创建 Query、Key 和 Value 矩阵**
&#8195;&#8195;==Transformer 中每个模块都有它自己的权重==（在后文中会拆解展示）。Self-Attention 将它的输入乘以权重矩阵（并添加一个 bias 向量，此处没有画出)，得到一个向量，这个向量基本上是 Query、Key 和 Value 向量的拼接。
![权重矩阵](https://img-blog.csdnimg.cn/65897e6719a7473a87a10b69e16938db.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;在之前的例子中，我们只关注了 Self Attention，现在加入 ==multi-head== 部分。Self-attention 在 Q、K、V 向量的不同部分进行了多次计算。拆分 attention heads 只是把一个长向量变为矩阵，矩阵每一行是一个head。小的 GPT-2 有 12 个 attention heads。
![multi-head](https://img-blog.csdnimg.cn/2d886f5e7ab64a33b7e0cbd6c8a54ac1.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
**(2) 评分**
&#8195;&#8195;为了简化，这里只关注一个 attention head。现在， token it 可以根据其他所有 token 的 Key 向量进行评分（这些 Key 向量在前面一个迭代中的第一个 attention head 计算得到的）：
![评分](https://img-blog.csdnimg.cn/48e868ba45c94642a7200df1bd5cfca1.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
**(3) 求和**
&#8195;&#8195;加权求和，得到第一个 attention head 的 Self Attention 结果$Z_{(9,1)}$,然后合并 所有的attention heads（连接成一个向量）
![合并多头注意力](https://img-blog.csdnimg.cn/b0fc1c9587c94b609e47baa15434e6f2.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;但这个向量还没有准备好发送到下一个子层FFNN（==向量的长度不对==）。我们需要把这个隐层状态的巨大向量转换为同质的表示。（如第一篇3.3讲的乘以矩阵$W^O$)
**(4) 映射（投影）**
&#8195;&#8195;在这里，我们使用第二个巨大的权重矩阵$W^O$，将拼接好的 Self Attention映射为最终的输出 Z。Z包含了所有 attention heads（注意力头） 的信息，将会输入到 FFNN 层。
![映射](https://img-blog.csdnimg.cn/8669a383444b41a3b2cebd124f67c3fc.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

![全部过程](https://img-blog.csdnimg.cn/f3809d2b35d441f89aa8d9c3dd6ce955.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
### 5.GPT-2 全连接神经网络FFNN
#### 5.1 FFNN结构
&#8195;&#8195;全连接神经网络是用于处理 Self Attention 层的输出，这个输出的表示包含了合适的上下文。全连接神经网络由两层组成。
**第 1 层**
&#8195;&#8195;第一层是模型大小的 4 倍（由于 GPT-2 small 是 768，因此这个网络会有 4*768=3072个神经元）。为什么是四倍？这只是因为这是原始 Transformer 的大小（如果模型的维度是 512，那么全连接神经网络中第一个层的维度是 2048）。这似乎给了 Transformer 足够的表达能力，来处理目前的任务。
![FFNN1](https://img-blog.csdnimg.cn/4b0e2be83a8c451b9a036ff957c02f17.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
上图没有展示 bias 向量，下面这张图也是

**第 2 层**
&#8195;&#8195;第 2 层把第一层得到的结果映射回模型的维度（在 GPT-2 small 中是 768）。这个相乘的结果是 Transformer 对这个 token 的输出。
![FFNN2](https://img-blog.csdnimg.cn/1f73a3c9a22e41c9809a2fff41468576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
#### 5.2 代码示例
以上看的比较累，源代码这部分很简洁：（借用的TransformerEncoder说明）
```python
class TransformerEncoderLayer(nn.Module):#多个EncoderLayer堆叠成Encoder
	 def __init__(self, d_model, nhead, dim_feedforward=3072, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
     	super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)#如果词向量768维，全连接第一层就是3072个神经元
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)#第二层映射回768维
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation 
"""
参数：
        d_model: 词嵌入的维度（必备）
        nhead: 多头注意力中平行头的数目（必备）
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率（Default = 0.1）
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0（Default = 1e-5）
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)（Default：False）"""
#前向传播
	def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src = positional_encoding(src, src.shape[-1])
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
        key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)#第一次残差连接，后面接标准化
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)#第二次残差连接+标准化
        src = self.norm2(src)
        return src
```
Transformer layer组成Encoder

```python
class TransformerEncoder(nn.Module):
    r'''
    参数：
        encoder_layer（必备）
        num_layers： encoder_layer的层数（必备）
        norm: 归一化的选择（可选）
        '''
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for _ in range(self.num_layers):
            output = self.layer(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output
```
&#8195;&#8195;总结一下，其实经过位置编码，多头注意力，Encoder Layer和Decoder Layer形状不会变的，而Encoder和Decoder分别与src和tgt形状一致。（src表示encoder输入，tgt是decoder输入）
#### 5.3 总结
&#8195;&#8195;总结一下，我们的输入会遇到下面这些权重矩阵：
![decoder总结](https://img-blog.csdnimg.cn/449d0e474163441980280718b10db729.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;每个模块都有它自己的权重。另一方面，模型只有一个 token embedding 矩阵和一个位置编码矩阵。

![多层decoder总结](https://img-blog.csdnimg.cn/c49d9c4663074c5285a822a1d168fbdc.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;如果你想查看模型的所有参数，我在这里对它们进行了统计：
![参数统计](https://img-blog.csdnimg.cn/ac3e82e0a9714c44ab1eb0809078cc62.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;由于某些原因，它们加起来是 124 M，而不是 117 M
### 6. 灾难性遗忘
&#8195;&#8195;为了进一步提升精调后模型的通用性以及收敛速度,可以在下游任务精调时加入一定权重的预训练任务损失。这样做是为了缓解在下游任务精调的过程中出现灾难性遗忘( Catastrophic Forgetting )问题。
&#8195;&#8195;==因为在下游任务精调过程中, GPT 的训练目标是优化下游任务数据上的效果,更强调特殊性。因此,势必会对预训练阶段学习的通用知识产生部分的覆盖或擦除,丢失一定的通用性==。通过<font color='red'>结合下游任务精调损失和预训练任务损失,可以有效地缓解灾难性遗忘问题</font>,在优化下游任务效果的同时保留一定的通用性。
损失函数=精调任务损失+$\lambda$预训练任务损失
一般设置$\lambda=0.5$，因为在精调下游任务时，主要目的还是优化有标注数据集的效果，即优化精调任务损失。预训练任务损失的加入只是为了提升精调模型的通用性，其重要程度不及精调任务损失。


