#!/usr/bin/python3
import numpy as np
from sklearn.datasets import load_boston
from sklearn import linear_model

boston = load_boston()
X = boston.data
y = boston.target

# shuffle data
# np.random.shuffle(X)
# np.random.shuffle(y)

N,D = np.shape(X)
y = np.reshape(y, [N, 1])
train_test_split = 400

# split
Xtrain = X[0:train_test_split]
ytrain = y[0:train_test_split]
Xtest = X[train_test_split:]
ytest = y[train_test_split:]

Ntrain, D = np.shape(Xtrain)
Ntest, Dtest = np.shape(Xtest)
if Dtest != D:
    raise ValueError("data dimensions not consistent")

# standardize data
Xmean = np.mean(Xtrain, axis=0)
Xstd = np.std(Xtrain, axis=0)
Xtrain = (Xtrain - Xmean) / Xstd
Xtest = (Xtest - Xmean) / Xstd

# add intercept column
Xtrain = np.append(np.ones([Ntrain,1]), Xtrain, axis=1)
Xtest = np.append(np.ones([Ntest,1]), Xtest, axis=1)

# function to calculate mean square error
def mse(theta, method, X, y):
    # err_2norm = np.linalg.norm(ytest - np.matmul(Xtest, theta), ord=2)
    err_2norm = np.linalg.norm(y - X.dot(theta), ord=2)
    # err_2norm = np.linalg.norm(ytrain - np.matmul(Xtrain, theta), ord=2)
    mse = 1 / len(y) * pow(err_2norm, 2)
    # mse = 1 / Ntrain * pow(err_2norm, 2)
    print(method + " mse = {0}".format(mse))

## LEAST SQUARES
XTX = np.matmul(Xtrain.transpose(), Xtrain)
XTXinv = np.linalg.inv(XTX)
premult = np.matmul(XTXinv, Xtrain.transpose())
theLSQ = np.matmul(premult, ytrain)
mse(theLSQ,   "LSQ             ", Xtest, ytest)

## RIDGE REGRESSION
Itrain = np.identity(D + 1)
Itrain[0][0] = 0 # no penalty on intercept
# import sys
# gamma = float(sys.argv[1])
gamma = 55 # manually tuned with multiple runs
# split further into train set and validation set
# validation set is separate from the test set, and is
# used in the loop to manually calibrate gamma
vsplit = train_test_split-200
Xtrain_orig = Xtrain
ytrain_orig = ytrain
Xtrain = Xtrain_orig[0:vsplit]
ytrain = ytrain_orig[0:vsplit]
Xvalid = Xtrain_orig[vsplit:]
yvalid = ytrain_orig[vsplit:]

XTX = np.matmul(Xtrain.transpose(), Xtrain)
XTXinv = np.linalg.inv(XTX + gamma * Itrain)
premult = np.matmul(XTXinv, Xtrain.transpose())
theRidge = np.matmul(premult, ytrain)
mse(theRidge, "Ridge validation", Xvalid, yvalid)
mse(theRidge, "Ridge test      ", Xtest, ytest)

# scikit lasso
gamma = 0.7 # manually tuned with multiple runs
reg = linear_model.Lasso(alpha=gamma, fit_intercept=False)
reg.fit(Xtrain,ytrain)
yhatLASSO = reg.predict(Xtest)
theLASSO = np.reshape(reg.coef_, [D+1,1])
mse(theLASSO, "LASSO validation", Xvalid, yvalid)
mse(theLASSO, "LASSO test      ", Xtest, ytest)
print("theta LASSO:\n{0}".format(theLASSO))
