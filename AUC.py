from sklearn.metrics import roc_auc_score

#O(n^2)
def auc1(y_true,y_prob):
    pos = []
    nega = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            pos.append(y_prob[i])
        else:
            nega.append(y_prob[i])
    score = 0
    for pp in pos:
        for np in nega:
            if pp > np:
                score += 1
            elif pp == np:
                score += 0.5
            else:
                continue
    return score / (len(pos) * len(nega))

#O(nlog(n))
def auc2(y_true,y_prob):
    new_data = [[prob,label] for prob,label in zip(y_prob,y_true)]
    new_data.sort(key=lambda x:x[0])
    total_num = len(y_true)
    pos_num = 0
    pos_score = 0
    for i, (prob,label) in enumerate(new_data):
        if label == 1:
            pos_num += 1
            pos_score += i + 1
    return (pos_score - (pos_num * (pos_num + 1) / 2))/ (pos_num * (total_num-pos_num))

if __name__ == '__main__':
    y_true = [1, 0, 0, 0, 1, 0, 1, 0, ]
    y_prob = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]
    print(roc_auc_score(y_true,y_prob))
    print(auc1(y_true,y_prob))
    print(auc2(y_true,y_prob))
