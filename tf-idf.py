from collections import defaultdict
import math
class TFIDF(object):
    def __init__(self,word_list):
        '''
        1.维护一个包含每个sentence出现的word的tf字典的列表
        2.维护一个全局的每个word出现文档的次数的字典
        '''
        self.tf_list, self.idf_dic = self.initialize(word_list)
        self.n = len(word_list)

    def initialize(self,word_list):
        tf_list = []
        idf_dic = defaultdict(int)
        for words in word_list:
            num = len(words)
            tf_dic = defaultdict(int)
            for word in words:
                tf_dic[word] += 1
            for word in tf_dic:
                idf_dic[word] += 1
                tf_dic[word] /= num
            tf_list.append(tf_dic)
        return tf_list, idf_dic

    def idf(self,word):
        return math.log(self.n/(self.idf_dic[word]+1))

    def tfidf(self,word,doc_id):
        return self.tf_list[doc_id][word] * self.idf(word)

    def output(self):
        tfidf_list = []
        for doc_id in range(self.n):
            tfidf_dic = {}
            for word in self.tf_list[doc_id]:
                tfidf_dic[word] = self.tfidf(word,doc_id)
            tfidf_list.append(tfidf_dic)
        return tfidf_list

if __name__ == '__main__':
    corpus = ['this is the first document',
              'this is the second second document',
              'and the third one',
              'is this the first document']
    words_list = [corpus[i].split(' ') for i in range(len(corpus))]
    print(TFIDF(words_list).output())