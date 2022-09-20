



###### @copy by sobhan siamak
#### Batch gradient descent


import numpy as np
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.utils import shuffle




# Read and preprocessing Data
hour = pd.read_csv('hour.csv').dropna()
print(hour.info())

#changing the yyyy-mm-dd format to dd
list_dd=[]
list2=[]
for i in hour['dteday']:
    list1 = i.split('-')
    list_dd.append(int(list1[2]))
    list2.append(1)

dfh = pd.DataFrame(list_dd, columns=['dteday'])
hour[['dteday']]=dfh[['dteday']]
bias = pd.DataFrame(list2, columns=['bias'])
hour[['instant']]=bias[['bias']]


X = hour.iloc[:,0:-3]
Y = hour.iloc[:,-1]
# print(X.tail())
# print(np.shape(X))

x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, random_state=32)
# x = pd.DataFrame(x)
# x = x.insert(0,"bias",1)
# print(x)
# print(len(x))
# o = np.ones((len(x),1))
# x1 = np.c_[o, x]
# print(x1)
# SHUFFLING
# xs , ys = shuffle(x, y)
# print(xs.tail())
# print(ys.tail())
# a = xs.shape[1]
# print(a)

def BatchGD(x, y):
    theta = np.ones([x.shape[1]])
    lr = 0.0009
    batches = 100
    iteration = 100
    xs, ys = shuffle(x, y)
    # xs , ys = x, y
    def compueloss(id, theta):
        loss= 0
        for i in range(batches):
            y1 = ys.iloc[i]
            x1 = xs.iloc[i, :]
            x2 = xs.iloc[i,id]
            # loss += np.dot((np.dot(transpose(theta), x1)-y1),x2)
            loss += (1/batches)*((np.dot(transpose(theta), x1)-y1)*x2)

        return loss
    for j in range(iteration):
        for id in range(len(theta)):
            theta[id] = theta[id] - lr*compueloss(id, theta)

    return theta


#Calculate Error
def MSE(thetaF, x, y):
    tError = 0
    yhat = np.matmul(x , thetaF)
    yy = yhat-y
    ys = yy**2
    ys2 = np.sum(ys)/(len(x)*10e+6)
    tError = ys2


    # for i in range(len(yhat)):
    #     yt=y.iloc[i]
    #     yht=yhat[i]
    #     tError += (yt - yht)**2


    # for i in range(0, len(x)):
    #     x = [i, 0]
    #     y = Dataset[i, 1]
    #     tError += (y - (theta1 * x + theta0)) ** 2
    # return tError /(float(len(x)))
    return tError



thetaF = BatchGD(x, y)
print("the list of Thetas are:")
print(thetaF)
errtrain = MSE(thetaF, x, y)
print("train MSE Error is:")
print(errtrain)


errtest = MSE(thetaF, x_test, y_test)
print("test MSE Error is:")
print(errtest)