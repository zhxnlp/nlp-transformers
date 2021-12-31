## GitHub发现开源项目，提高工作效率
&#8195;&#8195;本文是[《learn-github-from-zero》](https://github.com/stormzhang)读书笔记，做了一些简洁化修改。
&#8195;&#8195;主要内容是GitHub页面介绍、Git Bash基础命令、GIT进阶、Git Flow分支管理流程和最后的开源项目参与。不包含GitHub账号注册、Git Bash下载安装、ssh密钥等内容。（这部分可参考[《GitHub 新手详细教程》](https://blog.csdn.net/Hanani_Jia/article/details/77950594)）

@[toc]
### 1.个人主页
&#8195;&#8195;我的 Timeline，这部分你可以理解成微博，就是你关注的一些人的活动会出现在这里，比如如果你们关注我了，那么以后我 star、fork 了某些项目就会出现在你的时间线里。
个人主页your profile
![个人主页](https://img-blog.csdnimg.cn/420944ae9fd441bf8710ab23ff4ead07.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
### 2.基本概念
**Repository**
&#8195;&#8195;仓库的意思，即你的项目，你想在 GitHub 上开源一个项目，那就必须要新建一个 Repository
，如果你开源的项目多了，你就拥有了多个 Repositories 。
**Issue**
&#8195;&#8195;问题的意思，举个例子，就是你开源了一个项目，别人发现你的项目中有bug，或者哪些地方
做的不够好，他就可以给你提个 Issue ，即问题，提的问题多了，也就是 Issues ，然后你看
到了这些问题就可以去逐个修复，修复ok了就可以一个个的 Close 掉。
**Star**
&#8195;&#8195;这个好理解，就是给项目点赞，但是在 GitHub 上的点赞远比微博、知乎点赞难的多，如果你有一个项目获得100个star都算很不容易了！
==Fork==
&#8195;&#8195;可以翻译成分叉。你开源了一个项目，别人想在你这个项目的基础上做些改进，然后应用到自己的项目中，这个时候他就可以 Fork 你的项目。
&#8195;&#8195;这个时候他的 GitHub 主页上就多了一个项目，只不过这个项目是基于你的项目基础（本质上是在原有项目的基础上新建了一个分支，分支的概念后面会在讲解Git的时候说到），他就可以随心所欲的去改进，但是丝毫不会影响原有项目的代码与结构。
==Pull Request==
&#8195;&#8195;发起请求，这个其实是基于 Fork 的，还是上面那个例子，如果别人在你基础上做了改进，后来就想把自己的改进合并到原有项目里，这个时候他就可以发起一个 Pull Request（简称PR） ，原有项目创建人就可以收到这个请求，这个时候他会仔细review你的代码，并且测试觉得OK了，就会接受你的PR，这个时候你做的改进原有项目就会拥有了。
**Watch**
&#8195;&#8195;这个也好理解就是观察，如果你 Watch 了某个项目，那么以后只要这个项目有任何更新，你
都会第一时间收到关于这个项目的通知提醒。
**Gist**
&#8195;&#8195;有时候你没有项目可以开源，只单纯的想分享一些代码片段，这个时候 Gist 就派上用场了！
3.操作起来
&#8195;&#8195;创建一个项目需要填写如上的几部分：项目名、项目描述与简单的介绍，你不付费没法选择私有的，所以接着只能选择 public 的，之后勾选「Initialize this repository with a
README」，这样你就拥有了你的第一个 GitHub 项目：
![创建项目](https://img-blog.csdnimg.cn/4fd2dfbceefa42b3932fccc044b34d89.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;可以看到这个项目只包含了一个 README.md 文件，但是它已经是一个完整的 Git 仓库了，你可以通过对它进行一些操作，如watch、star、fork，还可以 clone 或者下载下来。GitHub 上所有关于项目的详细介绍以及 Wiki 都是基于Markdown 的。
![新项目](https://img-blog.csdnimg.cn/6cac27ad49114212ae703c4368065585.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
### 3.Git 速成
&#8195;&#8195;Git 是当下最方便最流行的==版本控制系统==，**软件开发中源代码的管理变得异常重要：**
&#8195;&#8195;比如为了防止代码的丢失，肯定本地机器与远程服务器都要存放一份，而且还需要有一套机制让本地可以跟远程同步；多人开发互不影响、bug紧急修改、查看历史更改记录等等。
安装好Git-bash之后，Git命令如下：
#### 3.1git命令
![git命令](https://img-blog.csdnimg.cn/a764270960bb4db7b3f2de85a427b648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
#### 3.2 git本地操作
&#8195;&#8195;在进行任何 Git 操作之前，都要先切换到 Git 仓库目录，也就是切换到项目的文件夹目录下。Git 所有的操作命令开头都要以 git 开头。

==首先创建一个新的文件夹并初始化为仓库==

mkdir test （创建文件夹test）
cd test （切换到test目录）
git init（初始化为仓库）
git status(查看状态）

&#8195;&#8195;默认就直接在 master 分支，输入后可以看到a.md 文件Untracked files ，就是说 a.md 这个文件还没有被跟踪，没有提交在 git 仓库里呢，可以使用 `git add` a.md提交的文件。

**提交文件git add**
&#8195;&#8195;再输入`git status`，此时提示以下文件 Changes to be committed ， 意思就是 a.md 文件等待被提交，当然你可以使用 `git rm --cached 文件名` 这个命令去移除这个缓存。或者 `git commit -a -m "first commmt"`移除名为  "first commmt"的提交。（你妹的，本地也给我删了）
**git commit**
&#8195;&#8195;接着我们输入 `git commit -m "first commit"` ，这个命令什么意思呢？ commit 是提交的意思，-m 代表是提交信息，执行了以上命令代表我们已经正式进行了第一次提交。这个时候再输入 git status ，会提示 nothing to commit。

**查看提交记录git log**
&#8195;&#8195;这个时候我们输入 git log 命令，会看到如下：
==git log 命令可以查看所有产生的 commit 记录==，所以可以看到已经产生了一条 commit 记录，而提交时候的附带信息叫 'first commit' 。

**git commit -am**
&#8195;&#8195;看到这里估计很多人会有疑问，我想要提交直接进行 commit 不就行了么，为什么先要再 add一次呢？首先 git add 是先把改动添加到一个「暂存区」，你可以理解成是一个缓存区域，临
时保存你的改动，而 git commit 才是最后真正的提交。这样做的好处就是防止误提交，当然
也有办法把这两步合并成一步，使用==git commit -am==就可以了。

**新建分支git branch a**
&#8195;&#8195;branch 即分支的意思，多人协作中很重要。每个人建立自己的分支，互不影响，最后合并。
&#8195;&#8195;执行 git init 初始化git仓库之后会默认生成一个主分支 master（默认分支），也基本是实际开发正式环境下的分支。一般情况下 ，不要在master 分支上直接操作的。
&#8195;&#8195;执行 git branch a 新建了 a 分支（分支 a 跟分支 master 是一模一样的内容）可以输入 git branch 查看下当前分支情况。

**切换分支git checkout a**

**新建并自动切换分支git checkout -b a**

**删除分支git branch -d **
&#8195;&#8195;分支新建错了，或者a分支的代码已经顺利合并到master 分支来了，那么a分支没用了，需要删除。如果a分支的代码还没有合并到master，你执行 git branch -d a 是删除不了的，它会智能的提示你a分支还有未合并的代码。

**强制删除分支git branch -D**

**合并分支git merge**
&#8195;&#8195;第一步是切换到 master 分支
&#8195;&#8195;第二步执行 git merge a ，合并a分支。但有时候会有冲突合并失败

**添加版本标签git tag**
&#8195;&#8195;git tag v1.0 就代表我在当前代码状态下新建了一个v1.0的标签，输入 git tag 可以查看历史 tag 记录。执行 **git checkout v1.0 **，这样就顺利的切换到 v1.0 tag的代码状态了。
### 3.3向 GitHub 提交代码
#### 3.3.1生成SSH
&#8195;&#8195;SSH是一种网络协议，用于计算机之间的加密登录。大多数 Git 服务器都会选择使用 SSH 公钥来进行授权，所以想要在 GitHub 提交代码的第一步就是要先添加 SSH key 配置。
&#8195;&#8195;输入 ssh-keygen -t rsa ，什么意思呢？就是指定 rsa 算法生成密钥，接着连续三个回车键（不需要输入密码），然后就会生成两个文件 id_rsa 和 id_rsa.pub ，而 id_rsa 是密钥，id_rsa.pub 就是公钥。
&#8195;&#8195;接下来要做的是把 id_rsa.pub 的内容添加到 GitHub 上，这样你本地的 id_rsa 密钥跟 GitHub上的 id_rsa.pub 公钥进行配对，授权成功才可以提交代码。（更多内容可以看其它的介绍）
[GitHub 新手详细教程](https://blog.csdn.net/Hanani_Jia/article/details/77950594)和这篇[Git生成公钥 bash](https://blog.csdn.net/JuncaiLiao/article/details/118176199)。

- 查看公钥：`cat ~/.ssh/id_rsa.pub`
 ####  3.3.2 `fatal: bad boolean config value '“false”' for 'http.sslverify`
从仓库克隆到本地时报错，是因为config --list的最后一行为false。输入`git config --global --edit`将最后一行删掉。
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/bc653fef81d8406ca684b8005228da4c.png)
也可以看到自己的账户信息
![在这里插入图片描述](https://img-blog.csdnimg.cn/235cf3c1e79840bc88aa98077be23c76.png)
#### 3.3.3`error: RPC failed curl 56 OpenSSL SSL read Connection was reset`。
git clone时出现错误原因是curl的postBuffer的默认值较小,需要在终端调整为到合适的大小,这里调整到500M

```python
git config --global http.postBuffer 524288000
#之后可以输入以下命令查看
git config --list
```
#### 3.3.4 `unable to access ‘https:...‘: OpenSSL SSL_read: Connection was reset, errno 10054`。
使用git clone 指令获取github项目时报错。解决方法，将url链接地址中的https改写为git。push时报错可以重启git bash。

```python
git clone https://github.com/....git
Cloning into 'vue'...
fatal: unable to access 'https://github.com/....git/': OpenSSL SSL_read: Connection was reset, errno 10054

git clone git://github.com/....git
```
#### 3.3.5 github push时 登录失败，但明明输入的是正确的账号密码
![在这里插入图片描述](https://img-blog.csdnimg.cn/f805307c71614b18a6f1ad6debd3e0c3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
执行 ：git update-git-for-windows

执行后，会更新git git更新 重新push的界面如下

这样只要你网页等登录github ,就可以登录了
#### 3.3.6 `remote: Support for password authentication was removed on August 13, 2021.（SSH令牌解决法）`
大意：远程：2021 年 8 月 13 日移除了对密码身份验证的支持。请改用个人访问令牌。
即github不支持密码验证的方案了，用`personal access token`代替。

1. 在Github上生成token
GitHub 帐户，转到Settings => Developer Settings => Personal Access Token => Generate New Token (Give your password) => Fill up the form（选择所有选项） => Generate token =>复制generated Token。它将类似于ghp_sFhFsSHhTzMDreGRLjmks4Tzuzgthdvfsrta（刷新就看不到了，要及时保存）

3. 将本地的凭证改为`Personal Access Token`
控制面板 -> 凭据管理器（或者直接控制面板搜索凭证打开） -> Windows凭据 -> 找到`git:https://github.com`，将凭据改为前面生成的token即可。
之前没有github凭证的，单击添加通用凭据=> 互联网地址将是`git:https://github.com`，用户名为gith邮箱。密码是GitHub 个人访问令牌。
![在这里插入图片描述](https://img-blog.csdnimg.cn/4f4b036c8c6d405ab48c64cc4ad0c475.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 3.7 `git commit 提交不了 error: pathspec 'project'' did not match any file(s) known to git.`
在Linux系统中，commit信息使用单引号,而windows系统，commit信息使用双引号。
所以在git bash中git commit -m '提交说明' 这样是可以的，但是在win命令行中就要使用双引号
#### 3.4. Push & Pull
**Push推送代码** ：
&#8195;&#8195;把本地代码推到远程仓库，使本地仓库跟远程仓库就可以保持同步了。
代码示例：
==git push origin master==：本地代码推到远程 master 分支。
**Pull下载远程仓库代码**
==git pull origin master==：把远程最新的代码更新到本地。
&#8195;&#8195;一般我们在 push 之前都会先 pull ，这样不容易冲突。

##### 3.4.2 提交本地文件到github
提交当前目录下所有文件：`git add.` 
此时可能报错：`The file will have its original line endings in your working directory`
 
首先出现这个问题主要原因是：我们从别人github地址上通过git clone下载下来，而又想git push到我们自己的github上，那么就会出现上面提示的错误信息。
这是因为文件中换行符的差别导致的。这个提示的意思是说：会把windows格式（CRLF（也就是回车换行））转换成Unix格式（LF），这些是转换文件格式的警告，不影响使用。
git默认支持LF。windows commit代码时git会把CRLF转LF，update代码时LF换CRLF。

此时需要执行如下代码：
```go
  git rm -r --cached .
  git config core.autocrlf false
  git add .
  git commit -m 'first commit'
  
```

```python
Your branch is ahead of 'origin/main' by 1 commit.
(use "git push" to publish your local commits)
```

意思是本地仓库有一个提交，比远程仓库要先进一个commit.
需要先把这个commit提交到远程仓库

```python
$ git push origin main
```

#### 3.5. 提交代码
&#8195;&#8195;向 GitHub 上我们提交我们自己的项目代码。有两种方法：

**1.克隆GitHub的项目 ，修改后再push**
&#8195;&#8195;以GitHub 上创建的 test 项目为例，执行如下命令：

	git clone git@github.com:stormzhang/test.git
	
&#8195;&#8195;这样就把 test 项目 clone 到了本地（这个时候该项目本身就已经是一个git 仓库了，不需要执行 git init 进行初始化，而且都已经关联好了远程仓库）我们只需要在这个 test 目录下任意修改或者添加文件，然后进行 commit ，之后就可以执行push来提交：

	git push origin master

&#8195;&#8195;仓库地址如下图：
![仓库地址](https://img-blog.csdnimg.cn/95ce05c8135046a7807180daebf11c2f.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
**2.关联本地已有项目，上传到GitHub**
&#8195;&#8195;如果我们本地已经有一个完整的 git 仓库，并且已经进行了很多次 commit，这个时候第一种方法就不适合了。

假设我们本地有个 test2 的项目，需要上传到GitHub。可执行如下操作：
&#8195;&#8195;首先：在GitHub 上建一个 test 的项目
&#8195;&#8195;其次：把本地 test2 项目与 GitHub 上的 test 项目进行关联
切换到 test2 目录，执行如下命令：

	git remote add origin git@github.com:stormzhang/test.git
然后把本地test2 上的所有代码 commit 记录提交到 GitHub 上的 test 项目。


&#8195;&#8195;意思是==添加一个远程仓库，仓库名字是origin== ，地址是git@github.com:stormzhang/test.git 
（*只有一个远程仓库时名字一般都叫 origin 。我们可能一个项目有多个远程仓库？比如 GitHub 一个，比如公司一个，这样的话提交到不同的远程仓库就需要指定不同的仓库名字了。*）

查看我们当前项目有哪些远程仓库可以执行如下命令：

	git remote -v
&#8195;&#8195;接下来，==本地仓库向远程仓库进行代码提交==：

	git push origin master
&#8195;&#8195;默认==向 GitHub 上的 master 分支的test 目录提交代码==。在提交代码之前先要设置下自己的用户名与邮箱。
#### 3.6 ipynb文件转为md文件
在要转换的ipynb文件当前目录下打开cmd，运行以下代码
```python
jm=“jupyter nbconvert --to markdown”
jm docs/篇章4-使用Transformers解决NLP任务/4.1-文本分类.ipynb
jm docs/篇章4-使用Transformers解决NLP任务/4.2-序列标注.ipynb
```
#### 3.7 解决github或gitee上传的md文件中[TOC]标签无法生成目录
1. 打开VScode，点击扩展，搜索`Markdown All in One`插件，安装
2. 在VScode中打开md文件
3. 使用Ctrl+Shift+P快捷键，输入命令`Markdown All in One: Create Table of Contents`，回车
![在这里插入图片描述](https://img-blog.csdnimg.cn/0cac21d70d9f488bb105059bb9379083.png)
### 4.Git 进阶
#### 4.1用户名和邮箱
&#8195;&#8195;每一次commit都会产生一条log，==log标记了提交人的姓名与邮箱==，以便其他人方便的查看与联系提交人，所以我们在进行提交代码的第一步就是要设置自己的用户名与邮箱。执行以下代码：
git config --global user.name "stormzhang"
git config --global user.email "stormzhang.dev@gmail.com"
&#8195;&#8195;以上进行了==全局配置==，当然有些时候我们的某一个项目想要用特定的邮箱，这个时候只需切换到你的项目，以上代码把 --global 参数去除，再重新执行一遍就ok了。

&#8195;&#8195;*PS：我们在 GitHub 的每次提交理论上都会在 主页的下面产生一条绿色小方块的记录，如果你确认你提交了，但是没有绿色方块显示，那肯定是你提交代码配置的邮箱跟你 GitHub 上的邮箱不一致，GitHub 上的邮箱可以到 Setting -> Emails里查看）*

#### 4.2 alias设置别名
&#8195;&#8195;*频繁执行一些Git命令非常繁琐，使用alias可以配置别名

	git config --global alias.co checkout # 别名
	git config --global alias.ci commit
	git config --global alias.st status
	git config --global alias.br branch
这样原本的命令

	git commit
	git checkout
	git branch
	git status
就变成了：

	git c
	git co
	git br
	git s
&#8195;&#8195;每个人别名可以设置的不一样。除此之外还可以设置组合，比如：

	git config --global alias.psm 'push origin master'
	git config --global alias.plm 'pull origin master'
&#8195;&#8195;之后就直接就用==git psm==代替 git push origin master，==git plm== 代替git pull origin master。

&#8195;&#8195;输入git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --date=relative 查看日志会变成：
![gitlog](https://img-blog.csdnimg.cn/876e29ec689a409db327adf33e8bbd74.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
	
	git config --global alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%
	d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --date=relative"
&#8195;&#8195;输入以上命令之后，输入==git lg==就可以了。
#### 4.3 其他配置
&#8195;&#8195;默认情况下 git 用的编辑器是 vi ，可以改成vim 。

	git config --global core.editor "vim" # 设置Editor使用vim
	git config --global color.ui true#给Git着色
	git config --global core.quotepath false # 设置显示中文文件名
&#8195;&#8195;默认这些配置都在 ~/.gitconfig 文件下的，你可以找到这个文件查看自己的配置，也可以输入 git config -l 命令查看
#### 4.4 查看改动 diff
&#8195;&#8195;比如我有一个 a.md 的文件，我现在做了一些改动，然后输入 git diff 就会看到如下：
![giff](https://img-blog.csdnimg.cn/59a513878b9a464f826c6bf755a3b38c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;**红色的部分前面 - 代表删除的，绿色的部分前面 + 代表增加的**，一目了然。
&#8195;&#8195;==直接输入 git diff 只能比较当前文件和暂存区文件差异==（暂存区就是还没有执行 git add 的文件）。还可以执行以下命令：
```python
git diff <$id1> <$id2> # 比较两次提交之间的差异
git diff <branch1>..<branch2> # 在两个分支之间比较
git diff --staged # 比较暂存区和版本库差异
```
#### 4.5 checkout切换分支、tag，撤销文件
1.checkout 切换分支：

	git checkout develop#切换到devlop分支
2.也可以用来切换tag，切换到某次commit，如：
```python
	git checkout v1.0
	git checkout ffd9f2dd68f1eb21d36cee50dbdd504e95d9c8f7 # 后面的一长串是commit_id，是每次commit的SHA1值，可以根据 git log 看到。
```
3.checkout 撤销还原文件
&#8195;&#8195;举个例子，假设我们在一个分支开发一个小功能，刚写完一半，这时候需求变了，而且是大变化，之前写的代码完全用不了了，好在你刚写，甚至都没有 git add 进暂存区，这个时候很简单的一个操作就直接把原文件还原：

	git checkout a.md
&#8195;&#8195;这里稍微提下，==checkout 命令只能撤销还没有 add 进暂存区的文件。==
#### 4.6. stash紧急修改bug
&#8195;&#8195;因为某些原因，需要暂时切到别的分支，紧急修复bug。修改完再切回来，但是现在的代码也要能保留。
&#8195;&#8195;这个时候 stash 命令就大有用处了，前提是我们的代码没有进行 commit ，哪怕你执行了add 也没关系，我们先执行:

	git stash#把当前分支所有没有 commit 的代码先暂存起来
&#8195;&#8195;这个时候你再执行 git status 你会发现当前分支很干净，几乎看不到任何改动，你的代码改动也看不见了，但其实是暂存起来了。执行git stash list你会发现此时暂存区已经有了一条记录。

&#8195;&#8195;这个时候你可以切换其他分支，赶紧把bug修复好。改完再切换回来：	

	git stash list  #查看暂存区记录
	git stash apply #还原之前暂存的代码
	git stash drop  #把最近一条的 stash 记录删除了，最好这么做
	git stash pop   #上面两个的组合，还原代码，并自动删除stash 记录
	drop  stash_id  #删除指定的某条记录，不跟参数就是删除最近的
	git stash clear #清空所有暂存区的记录
	
#### 4.7merge & rebase合并分支
&#8195;&#8195;这两个都可以合并分支。

	git checkout master #切换分支master
	git merge featureA  #合并分支featureA 
	git rebase featureA #合并分支featureA 
&#8195;&#8195;区别你们可以理解成有两个书架，你需要把两个书架的书整理到一起去，
&#8195;&#8195;==merge== ，比较粗鲁暴力，就直接腾出一块地方把另一个书架的书全部放进去，虽然暴力，但是这种做法你可以知道哪些书是来自另一个书架的；
&#8195;&#8195;==rebase== ，他会把两个书架的书先进行比较，按照购书的时间来给他重新排序，然后重新放置好，这样做的好处就是合并之后的书架看起来很有逻辑，但是你很难清晰的知道哪些书来自哪个书架的。

#### 4.8 解决冲突（以后再写）
### 5.Git 分支管理
#### 5.1 常用分支命令

	git branch develop      #新建分支
	git checkout -b develop #新建并自动切换分支
	git push origin develop #把develop 分支推送到远程仓库
	git branch              #查看本地分支列表
	git branch -r           #查看远程分支列表
	git branch -d develop   #删除本地分支
	git branch -D develop   #强制删除
	git push origin :develop #删除远程分支
	git checkout -b develop origin/develop#把远程的 develop 分支迁到本地并切换（本地没有deelop）
	
&#8195;&#8195;执行上述命令的新建分支，是基于当前所在分支的基础上进行的，新建的分支和原来分支一模一样。
#### 5.2 分支管理流程 Git Flow
&#8195;&#8195; Git Flow 是一种比较成熟的分支管理流程，一张图能清晰的描述他整个的工作流程：
![git flow](https://img-blog.csdnimg.cn/581ed260eb2a4ee182a0c56e093e8e07.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
一般开发来说，大部分情况下都会拥有两个分支 master 和 develop，他们的职责分别是：
==master==：永远处在即将发布(production-ready)状态
==develop==：最新的开发状态
&#8195;&#8195; master、develop 分支大部分情况下都会保持一致，只有在上线前的测试阶段，develop 比 master 的代码要多，一旦测试没问题，==准备发布时候会将 develop 合并到master 上。==

&#8195;&#8195; 发布后继续进行下一版本的功能开发，开发中间可能遇到问题：
1.需要紧急修复 bug，
2.一个功能开发完成之后突然需求变动了
&#8195;&#8195;所以 Git Flow 除了以上 master 和 develop两个主要分支以外，还提出了以下三个辅助分支：

**==feature: 开发新功能==**, 基于 develop, 完成后 merge 回 develop
**==release 准备发布版本的分支==**，用来修复 bug，基于 develop，完成后 merge 回develop 和 master
**==hotfix: 修复 master 上的问题==**，等不及 release 版本就必须马上上线。基于 master, 完成后merge 回 master 和 develop

举例：1.**基于 develop 分支新建个分支==做功能A==：**

	git branch feature/A #其实就是一个规范，规定了所有开发的功能分支都以 feature 为前缀。
&#8195;&#8195;2.**遇到==紧急bug==需要修复**。赶紧停下手头的工作，立刻切换到 master 分支，然后再此基础上新建一个分支：

	git branch hotfix/B #紧急修复分支，修复完成之后直接合并到 develop 和 master ，然后发布
&#8195;&#8195;之后再切回我们的 feature/A 分支继续开发。

&#8195;&#8195;3.**==最终测试==**。开发完了，合并回 develop 分支，然后在 develop 分支测试环境。跟后端对接并且测试的差不多了，可以正式发布了，这时候再新建一个 release 分支：

	git branch release/1.0
&#8195;&#8195;这个时候所有的 api、数据等都是正式环境，然后在这个分支上进行==最终测试==，发现 bug 直接进行修改，直到测试 ok 达到了发布的标准，最后把该分支合并到 develop 和 master 然后进行发布。

&#8195;&#8195;以上就是 Git Flow 的概念与大概流程，看起来很复杂，但是对于人数比较多的团队协作现实
开发中确实会遇到这么复杂的情况，是目前很流行的一套分支管理流程。
#### 5.3  Git Flow实际应用
&#8195;&#8195;有人会问，每次都要各种操作，合并来合并去，有点麻烦，这点 Git Flow 早就想到了，并为此推出了一个 Git Flow 的工具，并且是开源的：[Git Flow开源地址](https://github.com/nvie/gitflow)
这个工具帮我们省下了很多步骤：

	git flow feature start A  #基于develop分支新建分支feature A
	git flow feature finish A #合并本分支到 develop 分支
&#8195;&#8195;如果是 hotfix 或者 release 分支甚至会自动帮你合并到 develop、master 两个分支
Git Flow具体安装与用法有待补充
### 6.GitHub 开源项目参与
#### 6.1 Github项目页面介绍
&#8195;&#8195;有人疑问，我自己目前还没有能力开源一个项目，但是想一起参与到别的开源项目中，该怎么操作呢？那么今天，就来给大家一起介绍下 GitHub 上的一些常见的操作。
&#8195;&#8195;以 Square 公司开源的 Retrofit 为例来介绍，打开链接：https://github.com/square/retrofit然后看到如下的项目主页：
![开源项目](https://img-blog.csdnimg.cn/7f43c74e2a8041e292ec2038ff5600c1.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;第一部分包括 Watch、Star、Fork，之前介绍过。第二部分，分别包括 Code、Issues、Pull requests、Projects、Wiki、Pulse、Graphs。后面我们会一个个解释下（6.2和6.3）。

#### 6.2 Pull requests：提交PR参与开源项目
&#8195;&#8195;GitHub 的最大魅力在于人人都可参与开发或者完善开源项目，通过 Pull requests 来完成，简称 PR。下面来详细演示如何发起 PR：
&#8195;&#8195;==第一步，Fork开源项目==
&#8195;&#8195;登录你的 GitHub 账号，然后找到你想发起 PR 的项目，这里以datawhale开源项目[《基于transformers的自然语言处理(NLP)入门》](https://github.com/datawhalechina/learn-nlp-with-transformers) 为例，点击右上角的 Fork 按钮，然后该项目就出现在了你自己账号的 Repository 里，点开后显示：
![forknlp](https://img-blog.csdnimg.cn/062cfa7d0ce74d5bbe257b318d3a5da0.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;可以看到左上角有一行小字：forked from datawhalechina/learn-nlp-with-transformers。除此之外，项目代码跟原项目一模一样，对于原项目来说，相当于别人新建了一个分支而已。
==第二步，clone远程项目到本地==
&#8195;&#8195;修改的 bug 也好，想要新增的功能也好，总之把自己做的代码改动开发完，保存好。接着，把自己做的代码改动 push 到 你自己的 GitHub 上去。
==第三步，提交PR==
&#8195;&#8195;点自己主页 Fork 过来的项目主页，点击上方Pull requests 切换页面，点击右上角New pull request 按钮，会到如下页面：
![pr](https://img-blog.csdnimg.cn/f6c44b64eb0a45d6a097e75ac35492a3.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;这个页面会==自动比较该项目与原有项目的不同之处==。最顶部声明了是datawhalechina/learn-nlp-with-transformers 项目的 main 分支与 fork 过来的 datawhalechina/learn-nlp-with-transformers项目 mian 分支所做的比较。下面英文These branches can be automatically merged表示这两个分支可以自动合并。 
&#8195;&#8195;写好标题和描述，然后我们点击中间的==Create pull request==按钮，至此我们就成功给该项目提交了一个 PR。
&#8195;&#8195;然后就等着项目原作者 review 你的代码，并且决定会不会接受你的 PR，如果接受，那么恭喜你，你已经是该项目的贡献者之一了。

#### 6.3 其它功能简介
==Code：代码文件==。项目的根目录里会添加一个介绍性的 README.md 文件。
==Issues：项目问题或 bug==。并不是说 Issues 越少越好，**Issues 被解决的越多说明该项目越受作者重视。** 有close（解决）和open （待解决）两种状态。
&#8195;&#8195;同时，大家有问题的时候都可以提 Issue，可以通过点击右上角的New Issue 来新建 。
![issues](https://img-blog.csdnimg.cn/5ddb59fba53e46e186a47aede8d0343a.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
==Projects==
&#8195;&#8195;这个这个功能就是方便你把一些 Issues、Pull requests进行分类。
==Wiki==
&#8195;&#8195;一般来说，我们项目的主页有 README.me 基本就够了，但是有些时候我们项目的一些用法很复杂，需要有详细的使用说明文档给开源项目的使用者，这个时候就用到了 Wiki。使用起来也很简单，直接点击右上角 New Page ，然后使用 markdown 语法即可进行编写。
==insights==
&#8195;&#8195;可以理解成该项目的活跃汇总。包括近期该仓库创建了多少个 Pull Request 或 Issue，有多少人参与了这个仓库的开发等，都可以在这里一目了然。根据这个页面，用户可以判断该项目受关注程度以及项目作者是否还在积极参与解决这些问题等。
![活跃图](https://img-blog.csdnimg.cn/16565b98a3514993abf0f4dee6f08cb3.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
Settings
&#8195;&#8195;如果一个项目是自己的，那么你会发现会多一个菜单 Settings，这里包括了你对整个项目的设置信息，比如对项目重命名，比如删除该项目，比如关闭项目的 Wiki 和 Issues 功能等，不过大部分情况下我们都不需要对这些设置做更改。感兴趣的，可以自行看下这里的设置有哪些功能。
![settings](https://img-blog.csdnimg.cn/110f4b8595cb4ab7949c4467f9b55f47.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)

&#8195;&#8195;以上就就是一个 GitHub 项目的所有操作，相信大家看完之后对 GitHub 上常用的操作都熟悉了。
### 7.如何找到优秀开源项目
&#8195;&#8195;有同学问了，GitHub 我大概了解了，Git 也差不多会使用了，但是 还是搞不清 GitHub 如何帮助我的工作，怎么提升我的工作效率？
&#8195;&#8195;问到点子上了，GitHub 其中一个最重要的作用就是发现全世界最优秀的开源项目，你没事的时候刷刷微博、知乎，人家没事的时候刷刷 GitHub ，看看最近有哪些流行的项目，久而久之，这差距就越来越大，那么==如何发现优秀的开源项目呢？==这篇文章我就来给大家介绍下。
#### 7.1 关注一些活跃的大牛
&#8195;&#8195;GitHub 主页有一个类似微博的时间线功能，所有你关注的人的动作，比如 star、fork 了某个项目都会出现在你的时间线上，这种方式适合我这种比较懒的人，不用主动去找项目，而这种基本是我每天获取信息的一个很重要的方式。不知道关注哪些人怎么办？找到一个大牛的账号，看看他都关注了谁。
#### 7.2 Trending热度
&#8195;&#8195;点击主页最上方的Explore 菜单跳到发现页面：
![发现](https://img-blog.csdnimg.cn/8bc460cd881a4986b2e8bf026a717a73.png#pic_center)
&#8195;&#8195;紧接着点击 Trending 按钮

&#8195;&#8195;Trending 直译过来就是趋势的意思，就是说这个页面你可以看到最近一些热门的开源项目，这个页面可以算是很多人主动获取一些开源项目最好的途径，可以选择「当天热门」、「一周之内热门」和「一月之内热门」来查看，并且还可以分语言类来查看，比如你想查看最近热门的 Android 项目，那么右边就可以选择 Python 语言。
![趋势](https://img-blog.csdnimg.cn/2e5b94268a414b018081f9f303298583.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;这样页面推荐大家每隔几天就去看下，主动发掘一些优秀的开源项目。

#### 7.3 Search优秀项目
&#8195;&#8195;除了 Trending ，还有一种最主动的获取开源项目的方式，那就是 GitHub 的 Search 功能。
&#8195;&#8195;举个例子，你是做AI 的，接触 GitHub 没多久，那么第一件事就应该==输入Python 关键字进行搜索==，然后右上角选择按照 star 来排序，结果如下图：
![search](https://img-blog.csdnimg.cn/21b519b8fc1b47ca82caef7efbff030b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;可以看到按照 star 数，排名靠前基本是一些比较火的项目，一定是很有用，才会这么火。值
得一提的是左侧依然可以选择语言进行过滤。

&#8195;&#8195;==而对于实际项目中用到一些库==，基本上都会第一时间去 GitHub 搜索下有没有类似的库，比如项目中想采用一个网络库，那么不妨输入 android http 关键字进行搜索，因为我只想找到关于Android 的项目，所以搜索的时候都会加上 android 关键字，按照 star 数进行排序，结果如下：
![http](https://img-blog.csdnimg.cn/5062ff6e1dce40adae8d1ccefcee2941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzU2NTkxODE0,size_16,color_FFFFFF,t_70#pic_center)
&#8195;&#8195;除此之外，GitHub 的 Search 还有一些小技巧，比如你想搜索的结果中 star 数大于1000的，那么可以在搜索栏填入：
android http stars:>1000

总结
GitHub 上优秀开源项目真的是一大堆，就不一一推荐了，授人以鱼不如授人以渔，请大家自行主动发掘自己需要的开源项目吧，不管是应用在实际项目上，还是对源码的学习，都是提升自己工作效率与技能的很重要的一个渠道，总有一天，你会突然意识到，原来不知不觉你已经走了这么远！

Tips：常见报错
在git clone 克隆项目时报错，这是服务器的SSL证书没有经过第三方机构的签署，所以报错。
解决办法：
git config --global http.sslVerify "false"

### 8.HuggingFace/Transformers
&#8195;&#8195;预训练模型参数不断变大,来源[Huggingface](https://huggingface.co/course/chapter1/4?fw=pt)（这网页干啥的我也没看，不清楚，后面补）
&#8195;&#8195;教程使用的transformer代码库在[HuggingFace/Transformers](https://github.com/huggingface/transformers)，也就是HuggingFace写的，是现在最流行的版本。HuggingFace主页项目还有tokenizers、datasets等。（第四章会用这里的数据集）
还有个
&#8195;&#8195;Transformer项目主页第二段也写了，Transformers包提供各种预训练模型，在[model hub](https://huggingface.co/models)可以找到。比如4.1中开头写到model_checkpoint = "distilbert-base-uncased"。故4.1使用的模型是distilbert-base-uncased。
&#8195;&#8195;下图可以看出，transformer项目里的模型checkpoint有67种，model hub里由用户和组织上传的模型成百上千种。
![checkpoint](https://img-blog.csdnimg.cn/17eacd4e54974684949b067702d68804.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_Q1NETiBA56We5rSb5Y2O,size_44,color_FFFFFF,t_70,g_se,x_16#pic_center)

&#8195;&#8195;教程4.1最后写了，自己写的模型可以上传模型到 [Model Hub](https://huggingface.co/models)。具体上传办法点[这里](https://huggingface.co/transformers/model_sharing.html)。之后就可以像教程4.1里一样，直接用模型名字就能使用自己上传的模型啦。

&#8195;&#8195;在Model Hub搜索 bert-base-chinese 结果如下：
![bert-base-chinese ](https://img-blog.csdnimg.cn/fbfaa4959b904fb5b3ac4820baf7d4af.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_Q1NETiBA56We5rSb5Y2O,size_77,color_FFFFFF,t_70,g_se,x_16#pic_center)
&#8195;&#8195;第一各模型名字就是bert-base-chinese，应该是官方的模型（作者的） ，第二个模型前面有ckiplab/表示这个模型是ckiplab写的。即个人上传的模型。

### 9.colab使用技巧
colab训练后保存的模型，并没有和自己的google drive云盘绑定，实际上只是保存在了colab的当前内存中，等你下次刷新就没了。而colab上训练好的模型保存到云盘有两种方法：
1. 挂载模型后切换到my drive目录，保存模型
```python
#connect to self drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/My Drive')
```
==保存模型==

```python
import torch
torch.save(model, './model.pt')
```

然后你存的任何东西都在drive的首页路径下了，下次读取之前也先跑上面这段代码。如果想更改其他路径，用os.chdir即可。

==恢复模型==（保证前面挂载了Google Drive的前提下，进行恢复）

```python
PATH = './drive/My Drive/tran.pth'
torch.save(best_model.state_dict(), PATH)
```
```python
PATH = './drive/My Drive/tran.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)
```
2. 利用文件复制命令，将文件保存至云端硬盘

```python
!cp -r "要保存的文件路径"  "/content/drive/My Drive/**"
```


