---
# 文章标题
title: CS336_Assignment 1 (basics)
# 设置写作时间
date: 2025-06-30
# 一个页面可以有多个分类
category:
  - CS336_Lab
# 一个页面可以有多个标签
tag:
  - 公开课
  - LLM
  - Lab
# 此页面会在文章列表置顶
sticky: true
# 此页面会出现在文章收藏中
star: true
# 侧边栏的顺序
# 数字越小越靠前，支持非整数和负数，比如 -10 < -9.5 < 3.2, order 为 -10 的文章会最靠上。
# 个人偏好将非干货或随想短文的 order 设置在 -0.01 到 -0.99，将干货类长文的 order 设置在 -1 到负无穷。每次新增文章都会在上一篇的基础上递减 order 值。
order: -5
---

## Assignment Overview

> course web:https://stanford-cs336.github.io/spring2025/

**这个homework有一定的算力要求~~毕竟要训练model~~**

&emsp;&emsp;CS336: Language Modeling from Scratch，是Stanford开的一门课，目的是带领学生`从头`构建llm，通过尽量少调用现有库的方式`手搓`llm。  
&emsp;&emsp;事实上由于我的大部分知识学习与工作都是通过`即用即学`的方式完成的，这种系统的学习确实是我所缺乏的，我将通过五个homework尽量补全llm所缺乏的知识，并希望以此提升我的code skill，方便开发新的work。  
&emsp;&emsp;CS336的第一个homework被叫做basic，但是我看到这个homework的overview就觉得不是很basic，事实上这个homework涉及到的theory确实是basic的，但是完成的方式和basic还是有一定区别。  
&emsp;&emsp;这个homework要完成以下几件事：
* 实现一个BPE分词器
* 实现transformer的各个组件
* 实现loss function、optimizer（经典的Adam）、scheduler和整个train过程
* 使用不同数据集、调整超参和消融实验

&emsp;&emsp;以上这些任务都强调不使用封装好的torch库~~真是充满手搓的暴力美学~~，一看就不是很basic...

### Byte-Pair Encoding (BPE) Tokenizer

&emsp;&emsp;这部分就是经典的tokenizer的实现了，就是要实现`字符 -> 字节 -> 整数序列`的转换，事实上就是要实现从自然语言到机器语言的转换，具体可以参考这两篇经典文章：

> [Neural Machine Translation of Rare Words with Subword Units] https://arxiv.org/abs/1508.07909  
> [Neural Machine Translation with Byte-Level Subwords]https://arxiv.org/abs/1909.03341

#### The Unicode Standard

&emsp;&emsp;`Unicode`标准是规定字符->整数的一种标准。  
&emsp;&emsp;在Python中，可以使用`ord()`函数将单个Unicode字符转换为其整数表示形式。`chr()`函数将整数Unicode代码点转换为具有相应字符的字符串。

```python
# Unicode example
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

Answer Problem (unicode1):  

```python
# A: What Unicode character does chr(0) return?
# Q: '\x00'
>>> chr(0)
'\x00'
# A: How does this character’s string representation (__repr__()) differ from its printed representa-tion?
# Q: It's a invisible string, so no output.
>>> print(chr(0))

# A: What happens when this character occurs in text? 
# It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
# >>> chr(0)
# >>> print(chr(0))
# >>> "this is a test" + chr(0) + "string"
# >>> print("this is a test" + chr(0) + "string")
# Q:
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

#### Unicode Encodings

&emsp;&emsp;Unicode`似乎已经`可以成功帮我们把string转成interger了，`似乎已经`可以帮我们转成可训练的机器语言了。  
&emsp;&emsp;但是Unicode的dictionary本身有两个缺陷：  
* dictionary本身很大（150K）
* dictionary会很稀疏（经常使用的词汇命中率会很高）

&emsp;&emsp;我们将使用Unicode编码，它将Unicode字符转换为字节序列。  
&emsp;&emsp;Unicode标准本身定义了三种编码：UTF-8、UTF-16和UTF-32，UTF-8是Internet的主要编码。

> `UTF-8`: 变长字符编码，被定义为将码点编码为 1 至 4 个字节，具体取决于码点数值中有效二进制位的数量
> `UTF-16`: 变长字符编码, 这种编码方式比较特殊, 它将字符编码成 2 字节 或者 4 字节
> `UTF-32`: 固定长度的编码，始终占用 4 个字节，足以容纳所有的 Unicode 字符，所以直接存储 Unicode 码即可，不需要任何编码转换

&emsp;&emsp;Unicode字符串编码与反编码，可以通过Python自带的原语实现：
* encode(): Unicode字符串编码为UTF-8
* decode(): UTF-8字节字符串解码为Unicode字符串

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

&emsp;&emsp;通过将我们的Unicode代码点转换为字节序列，我们实际上是在获取代码点序列（0到154,997范围内的整数）并将其转换为字节值序列（0到255范围内的整数）。

Answer Problem (unicode2):

```python
# Q: What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? 
# It may be helpful to compare the output of these encodings for various input strings.

# A: UTF-8 is a dynamic length code, so it just need a byte for ASCII, making it efficient for English.

# Q: Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. 
# Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

# A: The functino just support to encode utf-8 byte by byte, but it will go wrong when we decode muti-byte string such as "你好". 

# Q: Give a two byte sequence that does not decode to any Unicode character(s).
# A: b'\xc3\x41' is a illegal two byte sequence.
```

#### Subword Tokenization

&emsp;&emsp;字节级标记化可以缓解单词级标记器面临的词汇表外问题，但将文本标记化为字节会导致极长的输入序列。这会带来两点坏处：
1. 处理这些较长的序列需要模型的每一步都需要更多的计算。
2. 较长的输入序列会在数据中产生长期依赖关系。

&emsp;&emsp;为了应对这个问题，我们采用`子词标记化`的方式，这是单词级标记器和字节级标记器之间的中点。

&emsp;&emsp;子词标记器权衡了更大的词汇表大小，以更好地压缩输入字节序列。