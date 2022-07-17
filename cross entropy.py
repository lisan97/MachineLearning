import math

def CrossEntropy(labels,probs):
    '''
    labels:[num_examples × 1]
    probs:[num_examples × num_classes]
    '''
    res = 0
    for i in range(len(labels)):
        res += math.log(probs[labels[i]])
    return res / len(labels)