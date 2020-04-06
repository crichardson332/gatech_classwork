from sklearn.datasets import load_boston
from sklearn import linear_model
import numpy as np

boston = load_boston()
X = boston.data
y = boston.target
N,D = np.shape(X)
y = np.reshape(y, [N, 1])

# standardize
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Xlsq = np.append(np.ones([506,1]), X, axis=1)
X = np.append(np.ones([506,1]), X, axis=1)

# split
Xtrain = X[0:400]
ytrain = y[0:400]
Xtest = X[401:]
ytest = y[401:]

# reshape

# least squares
XTX = np.matmul(Xtrain.transpose(), Xtrain)
XTXinv = np.linalg.inv(XTX)
premult = np.matmul(XTXinv, Xtrain.transpose())
theLSQ = np.matmul(premult, ytrain)

err_2norm = np.linalg.norm(ytest - np.matmul(Xtest, theLSQ), ord=2)
mse = 1 / N * pow(err_2norm, 2)
print("LSQ mse = {0}".format(mse))


# scikit lasso
reg = linear_model.Lasso(alpha = 6)
reg.fit(Xtrain,ytrain)
yhatLASSO = reg.predict(Xtest)

theLASSO = np.reshape(reg.coef_, [D+1,1])
err_2normLASSO = np.linalg.norm(ytest - np.matmul(Xtest, theLASSO), ord=2)
# err_2normLASSO = np.linalg.norm(ytest - yhatLASSO, ord=2)
mseLASSO = 1 / N * pow(err_2normLASSO, 2)
print("LASSO mse = {0}".format(mseLASSO))
print("LASSO score = {0}".format(reg.score(Xtest, ytest)))

# import pdb;pdb.set_trace()
