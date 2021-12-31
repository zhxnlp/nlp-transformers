@[toc]
# transformers29期问题点
## 1. BERT的三个Embedding直接相加会对语义有影响吗？
[原帖子在这](https://www.zhihu.com/question/374835153)

这是一个非常有意思的问题，苏剑林老师也给出了回答，真的很妙啊：

>Embedding的数学本质，就是以one hot为输入的单层全连接，也就是说，世界上本没什么Embedding，有的只是one hot。我们将token,position,segment三者都用one hot表示，然后concat起来，然后才去过一个单层全连接，等价的效果就是三个Embedding相加。[原文链接：词向量与Embedding究竟是怎么回事？](https://kexue.fm/archives/4122)


在这里想用一个例子再尝试解释一下：

假设 token Embedding 矩阵维度是 [4,768]；position Embedding 矩阵维度是 [3,768]；segment Embedding 矩阵维度是 [2,768]。假设它的 token one-hot 是[1,0,0,0]；它的 position one-hot 是[1,0,0]；它的 segment one-hot 是[1,0]。

那这个字最后的 word Embedding，就是上面三种 Embedding 的加和。如此得到的 word Embedding，和concat后的特征：[1,0,0,0,1,0,0,1,0]，再过维度为 [4+3+2,768] = [9, 768] 的全连接层，得到的向量其实就是一样的。

再换一个角度理解：

直接将三个one-hot 特征 concat 起来得到的 [1,0,0,0,1,0,0,1,0] 不再是one-hot了，但可以把它映射到三个one-hot 组成的特征空间，空间维度是 4*3*2=24 ，那在新的特征空间，这个字的one-hot就是[1,0,0,0,0...] (23个0)。

此时，Embedding 矩阵维度就是 [24,768]，最后得到的 word Embedding 依然是和上面的等效，但是三个小Embedding 矩阵的大小会远小于新特征空间对应的Embedding 矩阵大小。

当然，在相同初始化方法前提下，两种方式得到的 word Embedding 可能方差会有差别，但是，BERT还有Layer Norm，会把 Embedding 结果统一到相同的分布。

BERT的三个Embedding相加，本质可以看作一个特征的融合，强大如 BERT 应该可以学到融合后特征的语义信息的。

## 2. 为什么BERT在第一句前会加一个[CLS]标志?或者说为何选[CLS]做整个句子的表征？

BERT在第一句前会加一个[CLS]标志，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。为什么选它呢，因为与文本中已有的其它词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个词的语义信息，从而更好的表示整句话的语义。

具体来说，self-attention是用文本中的其它词来增强目标词的语义表示，但是目标词本身的语义还是会占主要部分的，因此，经过BERT的12层，每次词的embedding融合了所有词的信息，可以去更好的表示自己的语义。

而[CLS]位本身没有语义，经过12层，得到的是attention后所有词的加权平均，相比其他正常词，可以更好的表征句子语义。

当然，也可以通过对最后一层所有词的embedding做pooling去表征句子语义。再补充一下bert的输出，有两种，在BERT TF源码中对应：
- get_pooled_out()，就是上述[CLS]的表示，输出shape是[batch size,hidden size]。
- get_sequence_out()，获取的是整个句子每一个token的向量表示，输出shape是[batch_size, seq_length, hidden_size]，这里也包括[CLS]，因此在做token级别的任务时要注意

## 3.为啥要做点积缩放，即为什么Attention中除以$\sqrt{d}$这么重要？
&#8195;&#8195;Attention的计算是在内积之后进行softmax，主要涉及的运算是$e^{q⋅k}$，我们可以大致认为内积之后、softmax之前的数值在$-3\sqrt{d}$到$3\sqrt{d}$这个范围内，由于d通常都至少是64，所以$e^{3\sqrt{d}}$比较大而$e^{-3\sqrt{d}}$比较小，因此经过softmax之后，Attention的分布非常接近一个one hot分布了，
这带来严重的梯度消失问题，导致训练效果差。（例如y=softmax(x)在|x|较大时进入了饱和区，x继续变化y值也几乎不变，即饱和区梯度消失）

&#8195;&#8195;相应地，解决方法就有两个:
- 像NTK参数化那样，在内积之后除以$\sqrt{d}$，使q⋅k的方差变为1，对应$e^3$,$e^{−3}$都不至于过大过小，这样softmax之后也不至于变成one hot而梯度消失了，这也是常规的Transformer如BERT里边的Self Attention的做法
- 另外就是不除以$\sqrt{d}$，但是初始化q,k的全连接层的时候，其初始化方差要多除以一个d，这同样能使得使q⋅k的初始方差变为1，T5采用了这样的做法。

## 4. 位置代码问题
P[:,:,0::2]=torch.sin(x)中，P[:,:,0::2]是什么意思？

X = X + P[:,:X.shape[1],:].to(X.device)是在干嘛？
输入向量是词向量加位置编码

## 5. 第一章前言图片《A Survey of Transformers》来自哪？
![A Survey of Transformers](https://img-blog.csdnimg.cn/294e8962cff046939e43412b8603025c.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

这张图片出自复旦大学邱锡鹏组最新综述：[《A Survey of Transformers》](http://arxiv.org/abs/2106.04554)！（2021-06）、[论文解读](https://blog.csdn.net/weixin_47728827/article/details/118546250)

在这之前有一篇[《Pre-trained Models for Natural Language Processing: A Survey》](https://arxiv.org/abs/2003.08271)（2020-03）、[中文版翻译](https://blog.csdn.net/weixin_42691585/article/details/105950385)，都是对于transformers模型的综述。
## 6.为什么输入X要经过权重矩阵变换得到QKV向量？为啥不直接用X运算？
&#8195;&#8195;如果直接用输入X进行计算，则X同时承担了三种角色：査询( Query )键( Key )和值( Value )，导致其不容易学习。

&#8195;&#8195;更好的做法是,对不同的角色使用不同的向量。<font color='red'>即使用不同的参数矩阵对原始的输人向量做线性变换,从而让不同的变换结果承担不同的角色。</font>具体地,分别使用三个不同的参数矩阵$W^Q$, $W^K$, $W^V$，将输入向量$x_{i}$映射为三个新的向量 $q_{i}$、$k_{i}$、$v_{i}$，分别表示查询、键和值对应的向量。
## 7. 为啥要使用多头注意力
- 自注意力结果之间是互斥的,无法同时关注多个输入：自注意力结果需要经过归一化,导致即使一个输入和多个其他的输入相关,无法同时为这些输入赋予较大的注意力值。使用多组自注意力模型产生多组不同的注意力结果,则不同组注意力模型可能关注到不同的输入上,从而增强模型的表达能力。
- 标准的 Attention 模型无法处理这种多语义的情况,所以,需要将向量序列 Q 、 K 、 V 多次转换至不同的语义空间（多个语义角度），对标准的 Attention 模型进行多语义匹配改进。
- 
## 8.nn.Linear函数的用法
[参考官方文档](https://pytorch.org/docs/1.7.1/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear)

# transformers28期问题点
## 1. 为什么bert只用了transformer-encoder部分
1.从两种模型的任务场景来说：

&#8195;&#8195;Transformer最初是用来做机器翻译的，所以借用了seq2seq的结构，用encoder来编码提取特征，用decoder来解析特征，得到翻译结果。解析时，解码器不仅要接受编码器的最终输出信息，还要接受上一时刻的翻译结果。而且中英文是两种不同的语义空间，所以中间有一个转换。<br>

&#8195;&#8195;而对于BERT，它作为一个预训练模型，最根本的目的是提取语料特征。即利用巨大的未标记文本数据，从中学习语言的良好的表示形式（包含通用语义、语法规则等等的语言模型）。所以BERT的输入是一种语义空间下的文本信息，只需要训练一种语言，只需要使用一种模块，encoder相比decoder更能做到这一点。

  （再扯一扯就是decoder比encoder多了一个encoder-decoder-attention层，只是用一个模块的话，这个层就是多余的了。而且decoder还采用了掩码机制，屏蔽未来时刻的信息，对于翻译来说有必要，对于学习language modeling就不合适）
  
2.从训练机制来说：

&#8195;&#8195;transformer是seq2seq结结构，以英译汉来举例。是用decoder来预测输入的英文对应的中文是什么。训练时，是中英文一对对来训练的，相当于有了标注。

&#8195;&#8195;而BERT主要任务是从无标注文本中学习language modeling，通过masked机制来学习，所以它是一个用上下文去推测中心词[MASK]的任务，故和Encoder-Decoder架构无关。它的输入输出不是句子，其输入是这句话的上下文单词，输出是[MASK]的softmax后的结果，最终计算Negative Log Likelihood Loss，并在一次次迭代中以此更新参数。

&#8195;&#8195;所以说，**BERT的预训练过程，其实就是将Transformer的Decoder拿掉，仅使用Encoder做特征抽取器，再使用抽取得到的“特征”做Masked language modeling的任务，通过这个任务进行参数的修正**。

&#8195;&#8195;当然了，BERT不仅仅做了MLM任务，还有Next Sequence Prediction，这个由于后序验证对模型的效果提升不明显，所以没有赘述。

&#8195;&#8195;注意：我们常说，xxx使用Transformer作为特征抽取器，这其实在说用Transformer的Encoder(主要是Self-Attention和短路连接等模块)做特征抽取器，和Decoder啥关系也没有

## 2.transformer中为啥要有那么多dropout？dropout函数用法是什么？
transformer中很多地方用到了dropout，例如：
- encoderlayer1（attention层）：输入src加入位置编码，进入 Multi-self-attention层。self.norm1(src + self.dropout1(src2))。即dropout输出src2，然后残差连接+Norm
- encoderlayer2（FFN层）：全连接第一层3072神经元扩维4倍，之后激活并dropout，送入第二个全连接层降维回768维。之后同样的Add+Norm，即src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))、self.norm2(src + self.dropout2(src2))
- decoderlayer中的cross层（另外两层也有）：和encoder比加入第二层，tgt=self.multihead_attn(tgt, memory, memory）。和之前一样，三层之后都是self.norm(src + self.dropout(src2))

[dropout官方文档](https://pytorch.org/docs/1.7.1/_modules/torch/nn/modules/dropout.html#Dropout)，dropout是随机对最后一维的元素，以一定比例替换为0，是一种正则手段，为了防止过拟合。



## 3. Trainer如何构建？
在教程4.0《基于Hugging Face -Transformers的预训练模型微调》中有讲解
## 4. BERT中，multi-head $768*64*12$与直接使用$768*768$矩阵统一计算，有什么区别？
![在这里插入图片描述](https://img-blog.csdnimg.cn/657ebe6178bd4fd084e624c4d409c412.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

假设batchsize=1，输入张量X的维度是[N,embed_dim]，多头的hidden_dim=embed_dim，多头的输出进行拼接，维度为[N,embed_dim*n_heads]，W^O的维度是[embed_dim*n_heads,embed_dim]，于是最后输出[N,embed_dim]，感觉这样也是可以算的？为什么要embed_dim//n_heads的大小呢？

参考苏剑林文章[《BERT中，multi-head 768*64*12与直接使用768*768矩阵统一计算，有什么区别？》](https://www.zhihu.com/question/446385446/answer/1752279087)

## 5.上图中R和X什么区别
x是第一个encoder层输入，就是原始单词需要embedding吧，后面层的输入是上一层的输出，不需要embedding了
## 6.相对位置的方向指的是绝对位置吗？
参考：[面经：什么是Transformer位置编码？](https://mp.weixin.qq.com/s/mZBHjuHJG9Ffd0nSoJ2ISQ)

## 7.self-attention和一般的attention区别是什么？
- self-attention是只有一个输入序列X，X本身做attention操作（当然，X后面转换为QKV向量）。
- 一般的attention的QKV向量可以来自不同的输入（比如图文匹配，输入图像和文字的向量信息？）

资源推荐：
[《神经网络与深度学习》](https://nndl.github.io/)、[github](https://github.com/nndl/nndl.github.io)
[B站视频《transformer从0开始详细解读》](https://www.zhihu.com/question/446385446/answer/1752279087)

## 8.为什么多头注意力机制里面  num_heads 的值需要能够被 embed_dim 整除
多头注意力机制，如果每个头64维度，8个头组成了512维度。于是512维度合并起来做快速矩阵运算。如果不能整除的话，比如513维，那么就有一个65维度的奇怪的头，不好处理的
## 9.permute(0, 2, 1, 3)这个函数的参数如何设置，搜了好久还是没看明白
transpose每次将任意两个指定的dim进行交换，相当于是2D array的转置操作，而permute可以一次将Tensor变换到任意的dimensions sequence的排列。

## 10.如果One hot构建方式最后构成的是一个向量的话，那么word embedding，构成的结果是什么？
也是向量，例如：
- one-hot: [0,0,0,1]
- word-embeding: [0.2,0.1,0.3]
## 11.2.2章Attention代码实例中，为啥KV有10个词，Q有12个词？
```python
class MultiheadAttention(nn.Module):
	#此处省略部分代码
	forward(self, query, key, value, mask=None):
	        # 注意 Q，K，V的在句子长度这一个维度的数值可以一样，可以不一样。
	        # K: [64,10,300], 假设batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
	        # V: [64,10,300], 假设batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
	        # Q: [64,12,300], 假设batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
```
在前面的讲解中，我们的 K、Q、V 矩阵的序列长度都是一样的。但是在实际中，K、V 矩阵的序列长度是一样的（加权求和），而 Q 矩阵的序列长度可以不一样。
  这种情况发生在：在解码器部分的Encoder-Decoder Attention层中，Q 矩阵是来自解码器前一层的输出，而 K、V 矩阵则是来自编码器的输出（也就是encoder层最后的输出memory）。如下图所示，encoder有引出的部分直接输入decoder的第二层。
  ![2层Transformer示意图](https://img-blog.csdnimg.cn/3f11bf0ddff144a693da45c8aa57437e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
再具体解释就是：比如中译英的时候source（源语言）是："这课不错"。然后部分翻译到"this course"的时候q只有2个，但source的 k v有4个(encoder传入的部分）。所以decooder的时候，就好像GPT一样，生成模型不能“看到未来预测未来”。照理来说Q的词个数应该小于等于KV，而教程中是反过来。

## 12.残差连接目的是什么？为什么要加层标准化？
残差连接是为了解决深层网络训练时退化问题，也顺便解决梯度消失问题。参考知乎文章[《重读经典：完全解析特征学习大杀器ResNet》](https://zhuanlan.zhihu.com/p/268308900)

## 13.GPT-2使用的是transformer-decoder部分，只是没有encoder-decoder-attention层。为啥不叫没有masked的encoder层？去掉中间那层不是更像encoder吗 
结构上的确很像，encoder和decoder主要是从语言的意思上来区分吧，一个是为了编码信息到一堆vecotr，另一个是把一堆信息解码出来
## 14.可视化self-attention的这里面，玩具transformer指啥？（这里后面改了）
玩具transformer就是一个输入很少的transformer，小demo。
## 15.bert embedding问题
有些迷糊，可以用embedding喂给bert，也可以用bert生成embedding，这个不是嵌套了吗？或者说模型变来变去，embedding是一直存在的，比如把bert生成的embedding喂给GPT-2

一句话中的词（it is good）比如：it，最开始经过embedding层将其转换成一个embed（it）向量表示为768维，然后这个768维的向量表示和其他词的768向量表示再bert模型里进行交互（比如各种attention和全联接），最后再每个词的位置都得到一个新的768维向量，
此时it依旧对应了一个768维的向量可以表达为BERT（embed（it），embed（is）， embed（good）），虽然都是768维的向量表示，但包含的信息是不一样的。所以狭义理解embedding可以说是：it查embedding table之后直接得到的768维的embedding，广义一点的embedding可以指：it这个词经过一些列模型变换（比如bert gpt lstm）之后得到的768维度向量embedding表示。都是一个向量表示。

## 16.transformer中的encoder的输出是什么来得？能否直接用输出在一个逻辑回归上来做分类任务？
每一个词对应的位置都可以得到一个向量，cls处对应的向量一般认为包含了句子全部信息，可以直接拿来分类。

## 17.提问：为什么调用AutoModelForQuestionAnswering，会找到BertForQuestionAnswering类的调用？
在教程3.2BERT应用，3.5节 BertForQuestionAnswering中，有这样一段代码
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
```
通过设置bert-large-uncased-whole-word-masking-finetuned-squad，会找到对应的json，根据该json文件中的url，可以找到需要下载的config.json，这个配置文件中列出如下的配置信息：
```python
{
  "architectures": [
    "BertForQuestionAnswering"
  ],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```
大家如果阅读到这段代码的时候，可以Debug一下，看看对应的config.json是如何配置的

## 18. tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')直接调用的是BertTokenizer类里的哪个函数呀？
直接调用的是PreTrainedTokenizerBase下面的__call__内置函数：

![PreTrainedTokenizerBase](https://img-blog.csdnimg.cn/156efd1907ca4ae7bbb263b71d813460.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
如果要看最后的调用，是PreTrainedTokenizerBase下面的prepare_for_model方法，返回BatchEncoding实例

调用链：
（1）构建PreTrainedTokenizer
（2）调用PreTrainedTokenizerBase下面的__call__内置方法
（3）调用PreTrainedTokenizerBase下面的prepare_for_model方法
（4）返回BatchEncoding实例

## 19. task6文本分类中超参搜索为啥分数更低？
解决：在配置trainer的args参数时，设置report_to=“none”。即：

```python
args = TrainingArguments(
    "test-glue",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    log_level='error',
    logging_strategy="no",
    report_to="none"
)
```
## 20. task6数据集下载问题
在下载glue数据集时显示无法连接到github连接 但我自己挂代理是能打开glue.py文件，如何解决？
设置代理，如果使用jupyter，需要set环境变量
![在这里插入图片描述](https://img-blog.csdnimg.cn/d3f402eacd10449a822ea88d8b0f2330.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/208f50b6fb7e449eb9a0d847bc8d81db.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
- 或者vpn软件直接全局代理，也不用设置啥，jupyter里就数据集下载好了。

- 没有梯子时：【设置Jupyter Notebook代理】
（1）配置代理，在console输入命令：
set HTTPS_PROXY=http://127.0.0.1:19180
set HTTP_PROXY=http://127.0.0.1:19180
（2）启动Jupyter Notebook：jupyter notebook

## 21【Task06注意】
（1）在数据集下载时，需要使用外网方式建立代理
（2）如果使用conda安装ray[tune]包时，请下载ray-tune依赖包
命令：conda install ray-tune -c conda-forge

【Task09】
https://relph1119.github.io/my-team-learning/#/transformers_nlp28/task09
【本次任务注意】
（1）在数据集下载时，需要使用外网方式建立代理；
（2）sacrebleu需要安装1.5.1版本，不然会报scr.DEFAULT_TOKENIZER找不到的错误；
（3）本次任务中的模型训练，笔者使用的是3080  GPU显卡，需要训练模型长达1小时。

## 22 task6文本分类，单独下载了数据集，如何加载？
【独立数据集加载】
（1）将下载好的数据集存放到{user_dir}\.cache\huggingface\datasets目录
注：Windows用户目录：C:\Users\{用户名}\.cache\huggingface\datasets
（2） 重新执行加载数据集的代码
![本地加载数据集](https://img-blog.csdnimg.cn/08fa742af965471fb6c3b61a457178d8.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 23 部分代码注释
天国之影给部分代码加了注释，地址这[这里](https://github.com/Relph1119/my-team-learning/tree/master/notebook/transformers_nlp28/codes)
这一块是代码部分，已经全部整理成一个python文件，里面写了注释，大家可以参考
![在这里插入图片描述](https://img-blog.csdnimg.cn/e871a2a1993d4d79a5754db8a5521281.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 24.文本分类任务中，数据集label的accptable和unaccptable-指的什么意思？

CoLA(The Corpus of Linguistic Acceptability，语言可接受性语料库)，单句子二分类任务，每个句子被标注为是否合乎语法的单词序列。标签分别是0和1，其中0表示不合乎语法，1表示合乎语法。accptable和un是指的是否合乎语法。

## 25. trainer.evaluate()的时报错AttributeError: 'NoneType' object has no attribute 'log_metric'
解决了，pip install tensorboard

## 26. src_len是啥？为何老在embedding时报错
![在这里插入图片描述](https://img-blog.csdnimg.cn/383d5c6b922445e4854fc8dfaf06f888.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2cb921b5ea674c489a3b18fd62c822e3.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/200e9f25c664448e90bca5ed4a973297.jpg?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
字典索引的问题啊，没有加上特殊字符的个数

## 27 GPT模型处理单词时，生成写一个token的 Query、Value 以及 Key 向量，为何它只需要重新使用第一次迭代中保存的对应向量
教程2.4-图解attention，章节"GPT2中的Self-Attention"中提到：
![GPT-2 attention](https://img-blog.csdnimg.cn/1402430856c2439ab7fd87c560450583.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

我有的奇怪的是，计算第一个token a的时候，不就已经生成所有词的qkv向量了吗，一开始不就可以保存所有词的k和v向量。还是说kv向量在经过decoder处理之后会变？

每个token的embedding是查表得到的，都有，但这个token的embedding还要加上 positioning embedding。position不同位置不一样的。然后就让不同位置的token embedding需要计算一次，综合position和embedding的表示才能得到kvq。

我觉得更好的回答是：输入向量是随机初始化的。当这个词进行attention计算之后，参数才是经过训练的，可以继续使用（还是比较模糊）










​
 
