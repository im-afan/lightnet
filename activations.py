import numpy as np

def leakyRelu(x, a=0.1):
    ret = []
    for i in range(x):
        if(x[i] > 0):
            ret.append(x[i])
        else:
            ret.append(a*x[i])

    return np.array(ret)