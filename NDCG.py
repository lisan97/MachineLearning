import numpy as np


def getDCG(scores):
    '''
    r(i)只有（0，1）两种取值
    '''
    return np.sum(
        (np.power(2, scores) - 1)/np.log2(np.arange(len(scores))+2)
    )

def getNDCG(scores, ideal_scores):
    idcg = getDCG(ideal_scores)
    dcg = getDCG(scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg/idcg

    return round(ndcg,6)

if __name__ == '__main__':
    scores = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ideal_scores = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(getNDCG(scores,ideal_scores))
