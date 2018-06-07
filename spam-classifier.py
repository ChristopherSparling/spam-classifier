from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv("./data/spambase.csv").as_matrix()
np.random.shuffle(data)
print(data[1:100,:])

X = data[:,:48] # Select first 48 columns
Y = data[:, -1] # Select last column, containing spam/not-spam identifier

x_train = X[:]
y_train = 