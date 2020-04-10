from sklearn.datasets import load_boston
from sklearn import linear_model
import numpy as np
import statsmodels.api as sm

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
Xtrain_nonstd = X[0:train_test_split]
ytrain = y[0:train_test_split]
Xtest_nonstd = X[train_test_split:]
ytest = y[train_test_split:]

Ntrain, D = np.shape(Xtrain_nonstd)
Ntest, Dtest = np.shape(Xtest_nonstd)
if Dtest != D:
    raise ValueError("data dimensions not consistent")

# standardize
Xmean = np.mean(Xtrain_nonstd, axis=0)
Xstd = np.std(Xtrain_nonstd, axis=0)
Xtrain = (Xtrain_nonstd - Xmean) / Xstd
Xtest = (Xtest_nonstd - Xmean) / Xstd

import pdb;pdb.set_trace()

# add intercept column
Xtrain = np.append(np.ones([Ntrain,1]), Xtrain, axis=1)
Xtest = np.append(np.ones([Ntest,1]), Xtest, axis=1)

# function to calculate mean square error
def mse(theta, method):
    # err_2norm = np.linalg.norm(ytest - np.matmul(Xtest, theta), ord=2)
    err_2norm = np.linalg.norm(ytest - Xtest.dot(theta), ord=2)
    # import pdb;pdb.set_trace()
    # err_2norm = np.linalg.norm(ytrain - np.matmul(Xtrain, theta), ord=2)
    mse = 1 / Ntest * pow(err_2norm, 2)
    # mse = 1 / Ntrain * pow(err_2norm, 2)
    print(method + " mse = {0}".format(mse))

## LEAST SQUARES
XTX = np.matmul(Xtrain.transpose(), Xtrain)
XTXinv = np.linalg.inv(XTX)
premult = np.matmul(XTXinv, Xtrain.transpose())
theLSQ = np.matmul(premult, ytrain)
mse(theLSQ, "LSQ")

## RIDGE REGRESSION
Itrain = np.identity(D + 1)
Itrain[0][0] = 0 # no penalty on intercept
gamma = 0.1
XTX = np.matmul(Xtrain.transpose(), Xtrain)
XTXinv = np.linalg.inv(XTX + gamma * Itrain)
premult = np.matmul(XTXinv, Xtrain.transpose())
theRidge = np.matmul(premult, ytrain)
mse(theRidge, "Ridge")

# scikit lasso
reg = linear_model.Lasso(alpha=0.1, fit_intercept=False)
reg.fit(Xtrain,ytrain)
yhatLASSO = reg.predict(Xtest)
theLASSO = np.reshape(reg.coef_, [D+1,1])
mse(theLASSO, "LASSO")
