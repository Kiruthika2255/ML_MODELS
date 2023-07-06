import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('C:/models/deci_tree.csv')
X=data.drop(['Outcome'],axis=1) #select the colums
y=data['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

#with default PRAMETER = to predict and accuracy
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)

# Predict the dependent variable for the test set
pred=clf.predict(X_test)
acc=accuracy_score(y_test,pred)
print('prediction',pred)
print('accuracy',acc*100)

#another with criterion='entropy',max_depth=3 - to predict and accuracy
clf1=DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf1.fit(X_train,y_train)
pred1=clf1.predict(X_test)
acc1=accuracy_score(y_test,pred1)
print('prediction',pred1)
print('accuracy',acc1*100)

plt.figure(figsize=(20,16))
tree.plot_tree(clf,fontsize=14,rounded=True,filled=True,max_depth=True)
plt.show()