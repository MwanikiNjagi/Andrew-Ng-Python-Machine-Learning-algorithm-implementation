import numpy as np
import pandas as pd 

data = pd.read_csv('C:\ex1data2.txt', sep = ',', header = None)
X = data.iloc[:,0:2] #reads first two columns
Y = data.iloc[:,2] #reads the third column
m = len(Y)
print(data.head())

X = (X - np.mean(X)/np.std(X))#feature normalization

ones = np.ones((m,1))
X = np.hstack((ones, X))
alpha = 0.01
iterations = 400
theta = np.zeros((3,1))
Y = Y[:,np.newaxis]
