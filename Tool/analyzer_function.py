import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import chebyshev

#Kullback-Leibler divergence
def KL(a,b,epsilon=.00001):
        #0を対数で取れないのでepsilonを足す
        a+=epsilon
        a/=np.sum(a)
        b+=epsilon
        b/=np.sum(b)
        return entropy(a,b)

#chebyshev distance
def CD(a,b):
        return chebyshev(a,b)

#canberra metric
def CM(a,b):
        D=np.abs(a-b)
        S=a+b
        return np.sum(D/S)

#classification
def Class(p,gt):
        #aは予測値pの最大値のインデックス
        a=np.argmax(p)
        #bは正解gtの最大値のインデックスのリスト
        b=np.where(gt == gt.max())[0]
        #bはタプルでindexのリストは0番目に入っている
        if np.any(b == a) == True:
                return 1
        else:
                return 0

def check_max(index,a):
    b=np.where(a == a.max())
    #bはタプルでindexのリストは0番目に入っている
    return np.any(b[0] == index)
                


