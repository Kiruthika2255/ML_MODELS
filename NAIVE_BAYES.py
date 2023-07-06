import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv('C:/models/iris.csv')

X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

print('Independent varable ',X)
print('dependent varable ',y)
print('display first 5 rows')
print(data.head(5))
print('display last 5 rows')
print(data.tail(5))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

gnb=GaussianNB()
gnb.fit(X_train,y_train)

predict=gnb.predict(X_test)
acc=accuracy_score(y_test,predict)
print('model accuracy ',acc*100)