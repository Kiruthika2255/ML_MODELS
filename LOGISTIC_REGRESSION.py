#LOGISTIC REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix

data=pd.read_csv('C:/models/logReg.csv')

#SLICE THE DATASET WITH LABELS
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

print('first 5 rows')
print(data.head(5))
print('last 5 rows')
print(data.tail(5))

print('independent variable',X)
print('dependent variable',y)

#SPLIT THE TRAIN ANDTEST DATASET
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=2)

#CONVERT THE DATASET INTO STANDARD FORMAT
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

logreg=LogisticRegression(random_state=0)
logreg.fit(X_train,y_train)

# Predict the dependent variable for the test set
pred=logreg.predict(X_test)
acc=accuracy_score(y_test,pred)
cm=confusion_matrix(y_test,pred)

print('prediction',pred)
print('confusion matrix',cm)
print('accuracy score',acc*100)