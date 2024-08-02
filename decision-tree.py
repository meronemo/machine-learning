import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
feature_cols = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'pedigree', 'age']
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)
pima = pima.iloc[1:]

X = pima[feature_cols] # features
y = pima.label # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

dt = DecisionTreeClassifier(criterion="entropy", max_depth=8)
dt = dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))