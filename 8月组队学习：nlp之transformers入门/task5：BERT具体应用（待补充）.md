## 如何应用 BERT
@[toc]
&#8195;&#8195;尝试 BERT 的最佳方式是通过托管在 Google Colab 上的 BERT FineTuning with Cloud TPUs。 BERT 代码可以运行在 TPU、CPU 和 GPU。

上一章我们查看 了BERT 仓库 中的代码：

### 1.BERT代码总结：
#### 1.1 BertTokenizer（Tokenization分词）
- 组成结构：BasicTokenizer和WordPieceTokenizer
- BasicTokenizer主要作用：
  1. 按标点、空格分割句子，对于中文字符，通过预处理（加空格方式）进行按字分割
  2. 通过never_split指定对某些词不进行分割
  3. 处理是否统一小写
  4. 清理非法字符
- WordPieceTokenizer主要作用：
  1. 进一步将词分解为子词(subword)，例如，tokenizer 这个词就可以拆解为“token”和“##izer”两部分，注意后面一个词的“##”表示接在前一个词后面
  2. subword介于char和word之间，保留了词的含义，又能够解决英文中单复数、时态导致的词表爆炸和未登录词的OOV问题
  3. 将词根和时态词缀分割，减小词表，降低训练难度  

- BertTokenizer常用方法：
  1. from_pretrained：从包含词表文件（vocab.txt）的目录中初始化一个分词器；
  2. tokenize：将文本（词或者句子）分解为子词列表；
  3. convert_tokens_to_ids：将子词列表转化为子词对应的下标列表；
  4. convert_ids_to_tokens ：与上一个相反；
  5. convert_tokens_to_string：将subword列表按“##”拼接回词或者句子；
  6. encode：
      - 对于单个句子输入，分解词，同时加入特殊词形成“[CLS], x, [SEP]”的结构，并转换为词表对应的下标列表；
      - 对于两个句子输入（多个句子只取前两个），分解词并加入特殊词形成“[CLS], x1, [SEP], x2, [SEP]”的结构并转换为下标列表；
  7. decode：可以将encode方法的输出变为完整句子。
  
#### 1.2 BertModel
BERT 模型有关的代码主要写在/models/bert/modeling_bert.py中，包含 BERT 模型的基本结构和基于它的微调模型等。

BertModel 主要为 transformer encoder 结构，包含三个部分：
- embeddings，即BertEmbeddings类的实体，根据单词符号获取对应的向量表示；
- encoder，即BertEncoder类的实体；
- pooler，即BertPooler类的实体，这一部分是可选的。

BertModel可以作为编码器（只有自我注意）也可以作为解码器，作为解码器的时候，只需要在自注意力层之间添加了交叉注意力（应该还要加masked机制，屏蔽未来信息）
- BertModel常用方法：
  1. get_input_embeddings：提取 embedding 中的 word_embeddings，即词向量部分；
  2. set_input_embeddings：为 embedding 中的 word_embeddings 赋值；
  3. _prune_heads：提供了将注意力头剪枝的函数，输入为{layer_num: list of heads to prune in this layer}的字典，可以将指定层的某些注意力头剪枝。
#### 1.3 BertEmbeddings
- 输出结果：通过word_embeddings、token_type_embeddings、position_embeddings三个部分求和，并通过一层 LayerNorm+Dropout 后输出得到，其大小为(batch_size, sequence_length, hidden_size)
- word_embeddings：子词(subword)对应的embeddings
- token_type_embeddings：用于表示当前词所在的句子，区别句子与 padding、句子对之间的差异
- position_embeddings：表示句子中每个词的位置嵌入，用于区别词的顺序

> 使用 LayerNorm+Dropout 的必要性：  
&emsp;&emsp;通过layer normalization得到的embedding的分布，是以坐标原点为中心，1为标准差，越往外越稀疏的球体空间中

&emsp;&emsp;词嵌入在torch里基于torch.nn.Embedding实现，实例化时需要设置的参数为词表的大小和被映射的向量的维度比如embed = nn.Embedding(10,8)。向量的维度通俗来说就是向量里面有多少个数。注意，第一个参数是词表的大小，如果你目前最多有8个词，通常填写10（多一个位置留给unk和pad），你后面万一进入与这8个词不同的词就映射到unk上，序列padding的部分就映射到pad上。
&emsp;&emsp;假如我们打算映射到8维（num_features或者embed_dim），那么，整个文本的形状变为100 x 128 x 8。接下来举个小例子解释一下：假设我们词表一共有10个词(算上unk和pad)，文本里有2个句子，每个句子有4个词，我们想要把每个词映射到8维的向量。于是2，4，8对应于batch_size, seq_length, embed_dim（如果batch在第一维的话）。
&emsp;&emsp;另外，一般深度学习任务只改变num_features，所以讲维度一般是针对最后特征所在的维度
### 1.4 BertEncoder
- 技术拓展：梯度检查点（gradient checkpointing），通过减少保存的计算图节点压缩模型占用空间

##### 1.4.1 BertAttention
分为BertSelfAttention+BertSelfOutput。后一个是Add+LayerNorm。
- BertSelfAttention
  1. 初始化部分：检查隐藏层和注意力头的参数配置倍率、进行各参数的赋值
  2. 前向传播部分：
      - multi-head self-attention的基本公式：
      $$
       \text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\ 
       \text{head}_i = \text{SDPA}(\text{QW}_i^Q, \text{KW}_i^K, \text{VW}_i^V) \\
       \text{SDPA}(Q, K, V) = \text{softmax}(\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$$
      - transpose_for_scores：用于将 hidden_size 拆成多个头输出的形状，并且将中间两维转置进行矩阵相乘
      - torch.einsum：根据下标表示形式，对矩阵中输入元素的乘积求和
      - positional_embedding_type：  
          - absolute：默认值，不用进行处理
          - relative_key：对key layer处理
          - relative_key_query：对 key 和 value 都进行相乘以作为位置编码
- BertSelfOutput：  
&emsp;&emsp;前向传播部分使用LayerNorm+Dropout组合，残差连接用于降低网络层数过深，带来的训练难度，对原始输入更加敏感。
##### 1.4.2 BertIntermediate
self-attention输出Z全连接3072个神经元（以small是768词向量举例），得到一个扩维4倍的结果。
- 主要结构：全连接和激活操作
- 全连接：将原始维度进行扩展，参数intermediate_size
- 激活：激活函数默认为 gelu，使用一个包含tanh的表达式进行近似求解
##### 1.4.3 BertOutput
全连接768个神经元，映射回768维向量。之后Add+LayerNorm。
主要结构：全连接、dropout+LayerNorm、残差连接（residual connect）
### 1.5BertPooler
&#8195;&#8195;主要作用：取出句子的第一个token，即\[CLS\]对应的向量，然后通过一个全连接层和一个激活函数后输出结果。
#### 1.6 总结
- BERT 不会将单词作为 token。相反，它关注的是 WordPiece。 tokenizer（就是tokenization.py）会将你的单词转换为适合 BERT 的 wordPiece；
- 模型是在 modeling.py（class BertModel）中定义的，和普通的 Transformer encoder 完全相同；
- run_classifier.py 是微调网络的一个示例。它还构建了监督模型分类层。如果你想构建自己的- 分类器，请查看这个文件中的 create_model() 方法；
- 可以下载一些预训练好的模型。这些模型包括 BERT Base、BERT Large，以及英语、中文和包括 102 种语言的多语言模型，这些模型都是在维基百科的数据上进行训练的。

