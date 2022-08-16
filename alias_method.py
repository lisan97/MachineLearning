import random
class Alias(object):
    def __init__(self,dic):
        self.area, self.alias,self.n, self.token2id = self.alias_method(dic)

    def alias_method(self,dic):
        n = len(dic)
        token2id = {}
        low_nums = []
        high_nums = []
        area = [0] * n
        alias = [0] * n
        i = 0
        for k,v in dic.items():
            if v <= 1.0 / n:
                low_nums.append(i)
            else:
                high_nums.append(i)
            token2id[i] = k
            i += 1
        while high_nums and low_nums:
            low_index = low_nums.pop()
            low_prob = dic[token2id[low_index]]
            high_index = high_nums.pop()
            high_prob = dic[token2id[high_index]]

            area[low_index] = low_prob
            alias[low_index] = high_index

            diff = 1.0 / n - low_prob
            remain = high_prob - diff
            dic[token2id[high_index]] = remain
            if remain > 1.0 / n:
                high_nums.append(high_index)
            else:
                low_nums.append(high_index)
        #可能会有正好等于1/n的还没pop出来
        while high_nums:
            high_index = high_nums.pop()
            area[high_index] = 1.0 / n
        while low_nums:
            low_index = low_nums.pop()
            area[low_index] = 1.0 / n

        return area, alias, n, token2id

    def sample(self):
        num1 = random.randint(0,self.n)
        num2 = random.uniform(0,1.0/self.n)
        if num2 < self.area[num1]:
            return self.token2id[num1]
        else:
            return self.token2id[self.alias[num1]]