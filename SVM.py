#BUILD MODEL USING SVM SUPPORT VECTOR MACHINE USING DIFFERENT KERNALS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

dataset=pd.read_csv('C:/models/iris.csv')
dataset['sepal length(cm)']=dataset['sepal length (cm)'].astype('int')


X=dataset.drop(['target'],axis=1)#used to delet the target colum  axis=1 colum
y=dataset['sepal length(cm)']

print(X)
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)


#sig=svc(kernel='poly',degree=8)

pol=SVC(kernel='poly',degree=8)
pol.fit(X_train,y_train)
predict=pol.predict(X_test)
acc=accuracy_score(y_test,predict)
cm=confusion_matrix(y_test,predict)
report=classification_report(y_test,predict)

print('prediction',predict)
print('confusion matrix',cm)
print('classification_report',report)
print('accuracy score',acc*100)



#sig=svc(kernel='sigmoid')

sig=SVC(kernel='sigmoid',degree=1)
sig.fit(X_train,y_train)
predict1=sig.predict(X_test)
acc1=accuracy_score(y_test,predict)
cm1=confusion_matrix(y_test,predict)
report1=classification_report(y_test,predict)

print('prediction',predict1)
print('confusion matrix',cm1)
print('classification_report',report1)
print('accuracy score',acc*100)


#sig=svc(kernel='rbf',degree=1)

gnb=SVC(kernel='rbf')
gnb.fit(X_train,y_train)
predict2=gnb.predict(X_test)
acc2=accuracy_score(y_test,predict)
cm2=confusion_matrix(y_test,predict)
report2=classification_report(y_test,predict)

print('prediction',predict2)
print('confusion matrix',cm2)
print('classification_report',report2)
print('accuracy score',acc*100)
