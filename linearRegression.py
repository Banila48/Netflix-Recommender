import numpy as np
import projectLib as lib
from openpyxl import Workbook
from openpyxl.chart import (
    LineChart,
    Reference,
)

import pandas as pd

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    for training_row in range(len(training)):
        A[training_row, training[training_row][0]] = 1
        A[training_row, training[training_row][1]+trStats["n_movies"]] = 1
    return A

# we also get c
def getc(rBar, ratings):
    c = []
    for i in range(len(ratings)):
        c.append(ratings[i] - rBar)
    return c

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    b = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T),c)
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    b = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A) + l*np.identity(trStats["n_movies"]+trStats["n_users"])),A.T),c)
    return b

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p


# Regulariser
lambdas = [x/10 for x in range(50)]
vlRMSEs = []
final_result = list()


# Trial and error different Lambda values
for l in lambdas:
    b = param_reg(A, c, l)
    trRMSE = lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
    vlRMSE = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
    vlRMSEs.append(vlRMSE)

    result = [l, trRMSE, vlRMSE]
    final_result.append(result)

# Save best Lambda value
best_vlRMSE = np.min(vlRMSEs)
best_lambda = lambdas[np.argmin(vlRMSEs)]


df = pd.DataFrame.from_records(final_result)
df.columns = ["lambda", "trRMSE", "vlRMSE"]
df.to_csv('linearRegression.csv', index=False)



