import numpy as np
import pandas as pd
import sklearn

data = pd.read_csv("mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"
x = np.array(data.drop([predict]), 1)
y = np.array(data[predict])

x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)