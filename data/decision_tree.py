import pandas as pd
from sklearn import tree
import pickle

data = pd.read_csv('iris.csv', delimiter=',')
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:5]

model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X,y)

#lưu
with open('dt_model.pkl','wb') as file:
    pickle.dump(model, file)