from pyhanlp import *
from collections import deque

def load_dictionary():
    """
    加载HanLP中的mini词库
    :return: 一个set形式的词库
    """
    # 利用JClass获取HanLP中的IOUtil工具类
    IOUtil = JClass('com.hankcs.hanlp.corpus.io.IOUtil')
    # 取得HanLP的配置项Config中的词典路径，并替换成CoreNatureDictionary.mini.txt词典
    path = HanLP.Config.CoreDictionaryPath.replace('.txt', '.mini.txt')
    # 读入加载列表中指定多个词典文件，返回的是Java Map对象
    dic = IOUtil.loadDictionary([path])
    print(type(dic))
    # 不关心词性和词频引出只获取Map对象的键值KeySet，并将其转换成Python的set集合
    return set(dic.keySet())

def forward_segment(text, dic,maxlength):
    """
    :param text: 待分词的中文文本
    :param dic: 词典
    :param maxlength: 词典最长单词的长度
    :return: 分词结果
    """
    word_list = []
    n = len(text)
    i = 0
    while i < n:
        word = text[i]
        for j in range(maxlength+1,1,-1):
            if i+j <= n and text[i:i+j] in dic:
                word = text[i:i+j]
                break
        word_list.append(word)
        i += len(word)
    return word_list

def backward_segment(text, dic,maxlength):
    """
    :param text: 待分词的中文文本
    :param dic: 词典
    :param maxlength: 词典最长单词的长度
    :return: 分词结果
    """
    word_list = deque()
    n = len(text)
    i = n - 1
    while i >= 0:
        word = text[i]
        for j in range(maxlength,0,-1):
            if i-j >= 0 and text[i-j:i+1] in dic:
                word = text[i-j:i+1]
                break
        word_list.appendleft(word)
        i -= len(word)
    return list(word_list)

def count_single_char(word_list: list):  # 统计单字成词的个数
    """
    统计单字词的个数
    :param word_list:分词后的list列表
    :return: 单字词的个数
    """
    return sum(1 for word in word_list if len(word) == 1)

def bidirectional_segment(text, dic,maxlength):
    """
    双向最长匹配
    :param text:待分词的中文文本
    :param dic:词典
    :param maxlength: 词典最长单词的长度
    :return:正向最长匹配和逆向最长匹配中最优的结果
    """
    f = forward_segment(text, dic,maxlength)
    b = backward_segment(text, dic,maxlength)
    # 词数更少优先级更高
    if len(f) < len(b):
        return f
    elif len(f) > len(b):
        return b
    else:
        # 单字词更少的优先级更高
        if count_single_char(f) < count_single_char(b):
            return f
        else:
            # 词数以及单字词数量都相等的时候，逆向最长匹配优先级更高
            return b

if __name__ == '__main__':
    # 加载词典
    dic = load_dictionary()
    print(forward_segment('就读北京大学', dic,5))
    print(bidirectional_segment('项目的研究', dic,5))
    print(backward_segment('研究生命起源', dic,5))
    print(bidirectional_segment('研究生命起源', dic,5))