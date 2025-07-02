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

> [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)  
> [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/abs/1909.03341)

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

&emsp;&emsp;子词标记器权衡了更大的词汇表大小，以更好地压缩输入字节序列。如果字节序列`the`经常出现在我们的原始文本训练数据中，那么在词汇表中为其分配一个条目，会将这个3个标记序列减少为单个标记。  

&emsp;&emsp;我们可以通过BPE算法选择这些子词单元来添加到我们的词汇表中，具体来说，就是迭代地用单个新的未使用索引合并出现最频繁的字节对。构建BPE标记器词汇表的过程称为`train` BPE tokenizer。

#### BPE Tokenizer Training

&emsp;&emsp;BPE的train过程主要包括三个step。  

##### Vocabulary initialization

&emsp;&emsp;`tokenizer vocabulary`是从字节串标记到整数ID的一对一映射, 由于我们使用byte级别的BPE tokenizer，所以初始vocabulary大小是256（UFT-8的0-255号字节值）

##### Pre-tokenization

&emsp;&emsp; 有vocabulary之后，我们就可以计算在文本中字节相邻出现的频率，从出现最频繁的byte pair开始合并，但是这存在两点问题，使我们一般不这么做：
1. 每次合并都需要对text全面检查，这会造成巨大的开销
2. byte pair可能只会在标点上有区别，却会被标记为不同的byte pair，这会导致很多byte pair其实会有很高的相似性

&emsp;&emsp; 为了避免这个问题，我们使用pre-tokenization的方法进行处理。事实上就是对text的粗粒度标记。`text`可能是出现10次的pre-tokenization，那在train BPE tokenizer 统计`t` `e` 相邻出现次数时，直接加10就好了，无需再查看context。  

&emsp;&emsp; 原始的BPE实现用`(s"")`，即空格实现pre-tokenization的划分，我们可以采用更为经典的基于正则表达式的pre-tokenization（used by GPT-2 from https://github.com/openai/tiktoken/pull/234/files）  

```python
# a useful regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

&emsp;&emsp; 上文的regex是一个非常便捷的拆分文本的正则表达式，我们可以通过一个例子进行演示：  

```python
>>> import regex as re
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# use findall
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']

# use finditer is recommended, for flexble usage of memory
>>> [match.group(0) for match in re.finditer(PAT, "some text that i'll pre-tokenize")]         
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

##### Compute BPE merges

&emsp;&emsp; 现在我们已经将text转换为pre-tokenization，并将每个pre-tokenization表示为UTF-8字节的序列，我们可以开始计算BPE merge（即train BPE tokenizer）。  

&emsp;&emsp; BPE算法迭代计算byte pair并识别频率最高的pair`("A"、"B")`。然后合并这个最频繁pair("A"、"B")的每次出现，即替换为一个新的标记`"AB"`。这个新的合并token被添加到我们的vocabulary中；因此，BPE训练后的最终vocabulary是初始vocabulary的大小，加上训练期间执行的BPE合并操作的数量。  

> 注意：在计算合并时，通过优先选择dictionary上更大的pair来确定地打破对频率上的联系。
> eg：max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
> ('BA', 'A')

##### Special tokens

&emsp;&emsp; 一些special token 被加入vocabulary是必要的（例如`<|endoftext|>`），special token可以用作区分text开始与结尾的label等等。

```python
# bpe example
def pre_tokenize(input_text: str, special_tokens: set):
    split_words = input_text.split(' ')
    pre_tokenization = {}
    
    for word in split_words:
        if word in special_tokens:
            pre_tokenization[(word,)] = pre_tokenization.get((word,), 0) + 1
        else:
            word_as_tuple = tuple(word) 
            pre_tokenization[word_as_tuple] = pre_tokenization.get(word_as_tuple, 0) + 1
            
    return pre_tokenization


def get_pair_frequencies(input : dict):
    output = {}
    for word, count in input.items():
        for i in range(len(word) - 1):
            left_char = word[i]
            right_char = word[i + 1]
            pair = (left_char, right_char)
            output[pair] = output.get(pair, 0) + count
    return output

def find_key(input : dict):
    max_value = max(input.values())
    max_key = [k for k, v in input.items() if v == max_value]
    return max(max_key)

def merge(vocabulary : dict, pair):
    new_vocab = {}
    for word_tuple, count in vocabulary.items():
        new_word_tuple = merge_single_word(word_tuple, pair)
        new_vocab[new_word_tuple] = count
    return new_vocab

def merge_single_word(word_tuple, pair_to_merge):
    new_word = list()
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair_to_merge:
            new_word.append(word_tuple[i] + word_tuple[i + 1])
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    return tuple(new_word)

def main():
    special_tokens = {'<|endoftext|>'}
    input_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    num_merges = 6 # 执行6次合并
    
    # 1. 初始化
    vocab = pre_tokenize(input_text, special_tokens) # 你的 pre_tokenization 字典
    
    # 2. 循环执行合并
    for i in range(num_merges):
        print(f"--- 合并第 {i+1} 次 ---")
        
        # 2a. 统计当前词汇表中的字节对频率
        pair_freqs = get_pair_frequencies(vocab)
        
        # 2b. 找到最佳对
        if not pair_freqs: # 如果没有对可以合并了
            break
        best_pair = find_key(pair_freqs)
        
        # 2c. 执行合并，更新词汇表
        vocab = merge(vocab, best_pair)
        
        print(f"合并了: {best_pair}")
        print("合并后的词汇表:")
        print(vocab)

    print("\n--- 最终结果 ---")
    print(vocab)
if __name__ == '__main__':
    main()

```

#### Experimenting with BPE Tokenizer Training

##### Parallelizing pre-tokenization

&emsp;&emsp;在训练前，我们可以发现一个主要的瓶颈是pre-tokenization，大文本量和不确定的分界线造就了这一点。  
&emsp;&emsp;幸运的是，课程已经为我们准备好了一个example程序`cs336_basics/pretokenization_example.py`供我们参考。  

##### Removing special tokens before pre-tokenization

&emsp;&emsp;在使用正则表达式模式（使用re.finditer）运行`pre-tokenization`之前，应该从text中删除所有特殊标记。  
&emsp;&emsp;确保对特殊标记进行拆分，以便它们分隔的文本之间不会发生合并。 函数`test_train_bpe_special_tokens`将检查这一点。  

##### Optimizing the merging step

&emsp;&emsp;BPE在每次合并时，它都会迭代所有字节对以识别最频繁的pair。然而，每次合并后唯一改变的pair count是那些与合并对重叠的对。  
&emsp;&emsp;BPE训练速度可以通过索引所有对的计数并逐步更新这些计数来提高，而不是显式地迭代每对字节来计算对频率。  

Problem (train_bpe):

提供一个通过test的example，建议自行完成，尤其需要思考`merge的优化`和`正则匹配的分割`：

```python
from .pretokenization_example import find_chunk_boundaries
import regex as re
import os
from typing import Dict, List, Tuple

def pre_tokenize(input_text: str, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
    word_counts = {}
    
    # 构建一个能分割文本的、包含所有特殊token的正则表达式
    # re.escape确保了特殊token中的元字符被正确处理
    # f"({ ... })" 创建了一个捕获组，这样 re.split 可以在结果中保留分隔符
    special_pattern = "|".join(re.escape(s) for s in special_tokens)
    special_splitter = re.compile(f'({special_pattern})')

    base_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 1. 按照特殊token分割文本
    text_parts = special_splitter.split(input_text)

    # 2. 遍历分割后的部分
    for i, part in enumerate(text_parts):
        if not part:  # 跳过 re.split 可能产生的空字符串
            continue
            
        if i % 2 == 1:  # 奇数索引是分隔符本身，即特殊token
            token_bytes = part.encode('utf-8')
            word_tuple = (token_bytes,)
            word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1
        else:  # 偶数索引是普通文本
            # 对普通文本部分应用基础的正则表达式
            for match in base_pattern.finditer(part):
                token_str = match.group(0)
                token_bytes = token_str.encode('utf-8')
                word_tuple = tuple(bytes([b]) for b in token_bytes)
                word_counts[word_tuple] = word_counts.get(word_tuple, 0) + 1
                
    return word_counts

def get_pair_frequencies(input : dict):
    output = {}
    for word, count in input.items():
        for i in range(len(word) - 1):
            left_char = word[i]
            right_char = word[i + 1]
            pair = (left_char, right_char)
            output[pair] = output.get(pair, 0) + count
    return output

def find_key(input : dict):
    max_value = max(input.values())
    max_key = [k for k, v in input.items() if v == max_value]
    return max(max_key)

def merge_single_word(word_tuple, pair_to_merge):
    new_word = list()
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair_to_merge:
            new_word.append(word_tuple[i] + word_tuple[i + 1])
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    return tuple(new_word)

def bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个字节级别的BPE分词器。
    """
    # --- 1. 初始化 ---
    final_vocab: Dict[int, bytes] = {}
    merges: List[Tuple[bytes, bytes]] = []
    
    # 首先添加特殊tokens到词汇表开头
    current_id = 0
    for token_str in special_tokens:
        final_vocab[current_id] = token_str.encode('utf-8')
        current_id += 1
    
    # 然后填充基础词汇表 (0-255字节)
    for i in range(256):
        final_vocab[current_id] = bytes([i])
        current_id += 1
        
    # --- 2. 全局预分词和频率统计 ---
    word_counts: Dict[Tuple[bytes, ...], int] = {}
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 32, "<|endoftext|>".encode("utf-8"))
            
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
            
            # 对每个块进行预分词
            chunk_word_counts = pre_tokenize(chunk_text, special_tokens)
            
            # 聚合到全局词频统计中
            for word, count in chunk_word_counts.items():
                word_counts[word] = word_counts.get(word, 0) + count

    # --- 3. BPE 迭代合并 ---
    num_merges_needed = vocab_size - len(final_vocab)

    for i in range(num_merges_needed):
        pair_freqs = get_pair_frequencies(word_counts)
        if not pair_freqs:
            break
        
        best_pair = find_key(pair_freqs)
        merges.append(best_pair)
        
        words_to_update = {}
        # 1. 找出所有包含 best_pair 的词
        for word, count in word_counts.items():
            if best_pair[0] in word and best_pair[1] in word:
                # 这是一个潜在的候选词，但我们还需要检查它们是否相邻
                for j in range(len(word) - 1):
                    if (word[j], word[j+1]) == best_pair:
                        words_to_update[word] = count
                        break # 找到一个匹配就够了，处理下一个词
        
        # 2. 更新这些词
        for word, count in words_to_update.items():
            # 从旧字典中移除
            del word_counts[word]
            # 创建新词并添加到字典中
            new_word = merge_single_word(word, best_pair)
            word_counts[new_word] = word_counts.get(new_word, 0) + count

        new_token_bytes = best_pair[0] + best_pair[1]
        final_vocab[current_id] = new_token_bytes
        current_id += 1
    return final_vocab, merges
```