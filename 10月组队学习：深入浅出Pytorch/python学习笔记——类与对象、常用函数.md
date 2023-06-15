@[toc]
## 一、python的类与对象
参考[《datawhale——PythonLanguage》](https://github.com/datawhalechina/team-learning-program/blob/master/PythonLanguage/13.%20%E7%B1%BB%E4%B8%8E%E5%AF%B9%E8%B1%A1.md)、[《Python3 面向对象》](https://www.runoob.com/python3/python3-class.html)
### 1.1 基本概念
对象 = 属性 + 方法
对象是类的实例。换句话说，类主要定义对象的结构，然后我们以类为模板创建对象。类不但包含方法定义，而且还包含所有实例共享的数据
- 类(Class): 用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。
- 对象：通过类定义的数据结构实例。对象包括两个数据成员（类变量和实例变量）和方法。
- <font color='red'>方法：类中定义的函数</font>。
- <font color='red'>类属性：类里面方法外面定义的变量称为类属性。</font>类属性所属于类对象并且多个实例对象之间共享同一个类属性，说白了就是类属性所有的通过该类实例化的对象都能共享。
<!--more-->
 ```python
class A():
    a = 0  # 类属性

    def __init__(self, xx):
        # 使用类属性可以通过 （类名.类属性）调用。
        A.a = xx
```
- <font color='red'>局部变量：定义在方法中的变量，只作用于当前实例的类。</font>

- 实例属性/变量：在类的声明中，属性是用变量来表示的。 <font color='red'>实例属性和具体的某个实例对象有关系，只能在自己的对象里面使用，其他的对象不能直接使用，不同实例对象的实例属性不共享。实例变量用 self 修饰，因为self是谁调用，它的值就属于该对象。
```python
class 类名():
    __init__(self)：
        self.name = xx #实例属性
```
- 数据成员：类变量或者实例变量用于处理类及其实例对象的相关的数据。
- <font color='red'>方法重写：改写从父类继承的方法，满足子类的需求
- <font color='red'>继承：派生类（derived class）继承基类（base class）的字段和方法。所以子类自动共享父类之间数据和方法</font>
- 多态：不同对象对同一方法响应不同的行动
- 实例化：创建一个类的实例，类的具体对象。

### 1.2 主要知识点
#### 1.2.1 类属性和实例属性区别

- 类属性：类外面，可以通过实例对象.类属性和类名.类属性进行调用。类里面，通过self.类属性和类名.类属性进行调用。
- 实例属性 ：类外面，可以通过实例对象.实例属性调用。类里面，通过self.实例属性调用。
- 实例属性就相当于局部变量。出了这个类或者这个类的实例对象，就没有作用了。
- 类属性就相当于类里面的全局变量，可以和这个类的所有实例对象共享。

```python
# 创建类对象
class Test(object):
    class_attr = 100  # 类属性

    def __init__(self):
        self.sl_attr = 100  # 实例属性

    def func(self):
        print('类对象.类属性的值:', Test.class_attr)  # 调用类属性
        print('self.类属性的值', self.class_attr)  # 相当于把类属性 变成实例属性
        print('self.实例属性的值', self.sl_attr)  # 调用实例属性


a = Test()
a.func()

# 类对象.类属性的值: 100
# self.类属性的值 100
# self.实例属性的值 100

b = Test()
b.func()

# 类对象.类属性的值: 100
# self.类属性的值 100
# self.实例属性的值 100

a.class_attr = 200
a.sl_attr = 200
a.func()

# 类对象.类属性的值: 100
# self.类属性的值 200
# self.实例属性的值 200

b.func()

# 类对象.类属性的值: 100
# self.类属性的值 100
# self.实例属性的值 100

Test.class_attr = 300
a.func()

# 类对象.类属性的值: 300
# self.类属性的值 200
# self.实例属性的值 200

b.func()
# 类对象.类属性的值: 300
# self.类属性的值 300
# self.实例属性的值 100
```
注意：属性与方法名相同，属性会覆盖方法。

```python
class A:
    def x(self):
        print('x_man')

aa = A()
aa.x()  # x_man
aa.x = 1
print(aa.x)  # 1
aa.x()
# TypeError: 'int' object is not callable
```

#### 1.2.2 self 是什么？
Python 的 self 相当于 C++ 的 this 指针。
在类的内部，使用 def 关键字来定义一个方法，与一般函数定义不同，<font color='red'>类方法必须包含参数 self, 且为第一个参数，self 代表的是类的实例。</font>（对应于该实例，即该对象本身）在调用方法时，我们无需明确提供与参数 self 相对应的参数

```python
class Ball:
    def setName(self, name):
        self.name = name

    def kick(self):
        print("我叫%s,该死的，谁踢我..." % self.name)


a = Ball()
a.setName("球A")
b = Ball()
b.setName("球B")
a.kick()
# 我叫球A,该死的，谁踢我...
b.kick()
# 我叫球B,该死的，谁踢我...
```
#### 1.2.3  Python 的魔法方法__init__
类有一个名为__init__(self[, param1, param2...])的魔法方法，该方法<font color='red'>在类实例化时会自动调用。</font>

```python
class Ball:
    def __init__(self, name):
        self.name = name

    def kick(self):
        print("我叫%s,该死的，谁踢我..." % self.name)


a = Ball("球A")
b = Ball("球B")
a.kick()
# 我叫球A,该死的，谁踢我...
b.kick()
# 我叫球B,该死的，谁踢我...
```
#### 1.2.4 继承和覆盖（super().\__init\__()）
派生类（derived class）继承基类（base class）的字段和方法。<font color='red'>如果子类中定义与父类同名的方法或属性，则会自动覆盖父类对应的方法或属性。多继承时基类方法名相同，子类没有指定时，使用靠前的父类的方法。

```python
# 类定义
class people:
    # 定义基本属性
    name = ''
    age = 0
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0

    # 定义构造方法
    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print("%s 说: 我 %d 岁。" % (self.name, self.age))


# 单继承示例
class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        # 调用父类的构函数
        people.__init__(self, n, a, w)
        #super().__init__()也可以，表示继承父类的init操作
        self.grade = g

    # 覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级" % (self.name, self.age, self.grade))


s = student('小马的程序人生', 10, 60, 3)
s.speak()
# 小马的程序人生 说: 我 10 岁了，我在读 3 年级
```
注意：如果上面的程序去掉：<font color='red'>people.__init__(self, n, a, w)</font>，则输出： 说: 我 0 岁了，我在读 3 年级，因为子类的构造方法把父类的构造方法覆盖了。如果继承的方法中含有父类init的属性但是没有在子类中实现，会报错。此时可以
- <font color='red'>调用未绑定的父类方法base class.__init__(self,*kargs)
- <font color='red'>使用super函数super().\__init\__()


==多继承时需要注意圆括号中父类的顺序，若是父类中有相同的方法名，而在子类使用时未指定，Python 从左至右搜索，调用靠前的类的此方法。==
```python
# 另一个类，多重继承之前的准备
class Speaker:
    topic = ''
    name = ''

    def __init__(self, n, t):
        self.name = n
        self.topic = t

    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s" % (self.name, self.topic))
```

```python
# 多重继承
class Sample01(Speaker, Student):
    a = ''

    def __init__(self, n, a, w, g, t):
        Student.__init__(self, n, a, w, g)
        Speaker.__init__(self, n, t)

# 方法名同，默认调用的是在括号中排前地父类的方法
test = Sample01("Tim", 25, 80, 4, "Python")
test.speak()  
# 我叫 Tim，我是一个演说家，我演讲的主题是 Python
```

```python
class Sample02(Student, Speaker):
    a = ''

    def __init__(self, n, a, w, g, t):
        Student.__init__(self, n, a, w, g)
        Speaker.__init__(self, n, t)

# 方法名同，默认调用的是在括号中排前地父类的方法
test = Sample02("Tim", 25, 80, 4, "Python")
test.speak()  
# Tim 说: 我 25 岁了，我在读 4 年级
```
#### 1.2.5公有和私有
变量名或函数名前加上“__”两个下划线，那么这个函数或变量就会为私有的了。
- 私有属性在类外部无法直接进行访问。
- 私有方法只能在类的内部调用 ，不能在类的外部调用
【例子】类的私有属性实例
```python

class JustCounter:
    __secretCount = 0  # 私有变量
    publicCount = 0  # 公开变量

    def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print(self.__secretCount)


counter = JustCounter()
counter.count()  # 1
counter.count()  # 2
print(counter.publicCount)  # 2

print(counter._JustCounter__secretCount)  # 2 Python的私有为伪私有
print(counter.__secretCount)  ## 报错，实例不能访问私有变量
# AttributeError: 'JustCounter' object has no attribute '__secretCount'
```
【例子】类的私有方法实例
```python
class Site:
    def __init__(self, name, url):
        self.name = name  # public
        self.__url = url  # private

    def who(self):
        print('name  : ', self.name)
        print('url : ', self.__url)

    def __foo(self):  # 私有方法
        print('这是私有方法')

    def foo(self):  # 公共方法
        print('这是公共方法')
        self.__foo()


x = Site('老马的程序人生', 'https://blog.csdn.net/LSGO_MYP')
x.who()
# name  :  老马的程序人生
# url :  https://blog.csdn.net/LSGO_MYP

x.foo()
# 这是公共方法
# 这是私有方法
x.__foo()#报错，外部不能使用私有方法
# AttributeError: 'Site' object has no attribute '__foo'
```
### 1.3魔法方法
魔法方法总是被双下划线包围，例如__init__。魔法方法的“魔力”体现在它们总能够在适当的时候被自动调用。
魔法方法的第一个参数应为cls（类方法） 或者self（实例方法）：
- cls：代表一个类的名称
- self：代表一个实例对象的名称

#### 1.3.1 __init\__(self[, ...]) 
__init\__(self[, ...]) :构造器，当一个实例被创建的时候调用的初始化方法

#### 1.3.2 __new\__(cls[, ...])
__new__方法主要是当你继承一些不可变的 class 时（比如int, str, tuple）， 提供给你一个自定义这些类的实例化过程的途径。
```python
在一个对象实例化的时候所调用的第一个方法，在调用__init__初始化前，先调用__new__。
__new__至少要有一个参数cls，代表要实例化的类，此参数在实例化时由 Python 解释器自动提供，后面的参数直接传递给__init__。
__new__对当前类进行了实例化，并将实例返回，传给__init__的self。
但是，执行了__new__，并不一定会进入__init__，只有__new__返回了当前类cls的实例，当前类的__init__才会进入。
```

#### 1.3.3 __del\__(self) 
- 析构器，当一个对象将要被系统回收之时调用的方法。

大部分时候，Python 的 ARC(自动引用计数，用来回收对象所占用的空间） 都能准确、高效地回收系统中的每个对象。但如果系统中出现循环引用的情况，比如对象 a 持有一个实例变量引用对象 b，而对象 b 又持有一个实例变量引用对象 a，此时两个对象的引用计数都是 1，而实际上程序已经不再有变量引用它们，系统应该回收它们，此时 Python 的垃圾回收器就可能没那么快，要等专门的循环垃圾回收器（Cyclic Garbage Collector）来检测并回收这种引用循环。

#### 1.3.4 算术运算符、反算术运算符、增量赋值运算符、一元运算符
例如以下运算符。详见[《PythonLanguage/14. 魔法方法.md》](https://github.com/datawhalechina/team-learning-program/blob/master/PythonLanguage/14.%20%E9%AD%94%E6%B3%95%E6%96%B9%E6%B3%95.md)
常见+- */和+= -=等等
```python
__add__(self, other)定义加法的行为：+
__sub__(self, other)定义减法的行为：-

一元运算符
__neg__(self)定义正号的行为：+x
__pos__(self)定义负号的行为：-x
__abs__(self)定义当被abs()调用时的行为
__invert__(self)定义按位求反的行为：~x
```
#### 1.3.5 属性访问
```python
__getattr__(self, name): 定义当用户试图获取一个不存在的属性时的行为。
__getattribute__(self, name)：定义当该类的属性被访问时的行为（先调用该方法，查看是否存在该属性，若不存在，接着去调用__getattr__）。
__setattr__(self, name, value)：定义当一个属性被设置时的行为。
__delattr__(self, name)：定义当一个属性被删除时的行为。
```
#### 1.3.6 定制序列__len__()和__getitem__()
容器类型的协议
- 如果说你希望定制的容器是不可变的话，你只需要定义__len__()和__getitem__()方法。
- 如果你希望定制的容器是可变的话，除了__len__()和__getitem__()方法，你还需要定义__setitem__()和__delitem__()两个方法。
- __len\__(self)定义当被len()调用时的行为（返回容器中元素的个数）。
- __getitem\__(self, key)定义获取容器中元素的行为，相当于self[key]。
- __setitem\__(self, key, value)定义设置容器中指定元素的行为，相当于self[key] = value。
- __delitem\__(self, key)定义删除容器中指定元素的行为，相当于del self[key]。

【例子】编写一个不可改变的自定义列表，要求记录列表中每个元素被访问的次数。

```python
class CountList:
    def __init__(self, *args):
        self.values = [x for x in args]
        self.count = {}.fromkeys(range(len(self.values)), 0)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        self.count[item] += 1
        return self.values[item]


c1 = CountList(1, 3, 5, 7, 9)
c2 = CountList(2, 4, 6, 8, 10)
print(c1[1])  # 3
print(c2[2])  # 6
print(c1[1] + c2[1])  # 7

print(c1.count)
# {0: 0, 1: 2, 2: 0, 3: 0, 4: 0}

print(c2.count)
# {0: 0, 1: 1, 2: 1, 3: 0, 4: 0}
```
【例子】编写一个可改变的自定义列表，要求记录列表中每个元素被访问的次数。

```python
class CountList:
    def __init__(self, *args):
        self.values = [x for x in args]
        self.count = {}.fromkeys(range(len(self.values)), 0)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        self.count[item] += 1
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __delitem__(self, key):
        del self.values[key]
        for i in range(0, len(self.values)):
            if i >= key:
                self.count[i] = self.count[i + 1]
        self.count.pop(len(self.values))


c1 = CountList(1, 3, 5, 7, 9)
c2 = CountList(2, 4, 6, 8, 10)
print(c1[1])  # 3
print(c2[2])  # 6
c2[2] = 12
print(c1[1] + c2[2])  # 15
print(c1.count)
# {0: 0, 1: 2, 2: 0, 3: 0, 4: 0}
print(c2.count)
# {0: 0, 1: 0, 2: 2, 3: 0, 4: 0}
del c1[1]
print(c1.count)
# {0: 0, 1: 0, 2: 0, 3: 0}
```
#### 1.3.7 迭代器__iter\__() 与 __next\__() 

 - 迭代是 Python 最强大的功能之一，是访问集合元素的一种方式。<font color='red'>字符串，列表或元组对象都可用于创建迭代器。
 -  迭代器是一个可以记住遍历的位置的对象。迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。
  - 迭代器有两个基本的方法：iter() 和 next()。

下面两个例子返回的结果是一样的
```python
string = 'lsgogroup'
for c in string:
    print(c)
for c in iter(string):
    print(c)
```

```python
links = {'B': '百度', 'A': '阿里', 'T': '腾讯'}
for each in links:
    print('%s -> %s' % (each, links[each]))
    
for each in iter(links):
    print('%s -> %s' % (each, links[each]))
```
- iter(object) 函数用来生成迭代器。
- next(iterator[, default]) 返回迭代器的下一个项目。iterator表示可迭代对象，default -- 可选，用于设置在没有下一个元素时返回该默认值，如果不设置，又没有下一个元素则会触发 StopIteration 异常。

```python
links = {'B': '百度', 'A': '阿里', 'T': '腾讯'}
it = iter(links)
print(next(it))  # B
print(next(it))  # A
print(next(it))  # T
print(next(it))  # StopIteration

while True:
    try:
        each = next(it)
    except StopIteration:
        break
    print(each)

# B
# A
# T
```
把一个类作为一个迭代器使用需要在类中实现两个魔法方法 __iter\__() 与 __next\__() 。

把一个类作为一个迭代器使用需要在类中实现两个魔法方法 __iter__() 与 __next__() 。

```python
__iter__(self)定义当迭代容器中的元素的行为，返回一个特殊的迭代器对象， 这个迭代器对象实现了
__next__() 方法并通过 StopIteration 异常标识迭代的完成。
 
__next__() 返回下一个迭代器对象。

StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况，在 __next__() 方法中
我们可以设置在完成指定循环次数后触发 StopIteration 异常来结束迭代。
```

```python
class Fibs:
    def __init__(self, n=10):
        self.a = 0
        self.b = 1
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > self.n:
            raise StopIteration
        return self.a


fibs = Fibs(100)
for each in fibs:
    print(each, end=' ')

# 1 1 2 3 5 8 13 21 34 55 89
```

### 1.4 扩展知识点
#### 1.4.1 组合

```python
class Turtle:
    def __init__(self, x):
        self.num = x

class Fish:
    def __init__(self, x):
        self.num = x

class Pool:
    def __init__(self, x, y):
        self.turtle = Turtle(x)
        self.fish = Fish(y)

    def print_num(self):
        print("水池里面有乌龟%s只，小鱼%s条" % (self.turtle.num, self.fish.num))
        
p = Pool(2, 3)
p.print_num()
# 水池里面有乌龟2只，小鱼3条
```
#### 1.4.2类的专有方法和内置函数
具体见[《datawhale——PythonLanguage》](https://github.com/datawhalechina/team-learning-program/blob/master/PythonLanguage/13.%20%E7%B1%BB%E4%B8%8E%E5%AF%B9%E8%B1%A1.md)
类的专有方法：

```python
__init__ : 构造函数，在生成对象时调用
__del__ : 析构函数，释放对象时使用
__repr__ : 打印，转换
__setitem__ : 按照索引赋值
__getitem__: 按照索引获取值
__len__: 获得长度
__call__: 函数调用
```
内置函数：
- issubclass(class, classinfo) 方法用于判断参数 class 是否是类型参数 classinfo 的子类。一个类被认为是其自身的子类。
- isinstance(object, classinfo) 方法用于判断一个对象是否是一个已知的类型，类似type()。
- hasattr(object, name)用于判断对象是否包含对应的属性。
- getattr(object, name[, default])用于返回一个对象属性值。
- setattr(object, name, value)对应函数 getattr()，用于设置属性值，该属性不一定是存在的。
- delattr(object, name)用于删除属性。
- class property([fget[, fset[, fdel[, doc]]]])用于在新式类中返回属性值

## 二、 基础函数
### 2.1 Python5个内建高阶函数（map、reduce、filter、sorted/sort、zip）
参考[《PythonLanguage/Python高阶函数使用总结.md》](https://github.com/datawhalechina/team-learning-program/blob/master/PythonLanguage/Python%E9%AB%98%E9%98%B6%E5%87%BD%E6%95%B0%E4%BD%BF%E7%94%A8%E6%80%BB%E7%BB%93.md)
![在这里插入图片描述](https://img-blog.csdnimg.cn/f2c08f549cb548b796f96222c0154e55.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/98622336d11f4aedba819c9ee2232977.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/afd9fd38372e4851bca74f05444bb3f3.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/bd8a8a20a4ab45429a97c0beddef245a.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA56We5rSb5Y2O,size_20,color_FFFFFF,t_70,g_se,x_16)


### 2.2 正则表达式
参考[《Python3 正则表达式》](https://www.runoob.com/python3/python3-reg-expressions.html)
