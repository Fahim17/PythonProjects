import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

X_train = np.loadtxt("test/X_train.dat")
y_train = np.loadtxt("test/y_train.dat")
X_test = np.loadtxt("test/X_test.dat")
y_test = np.loadtxt("test/y_test.dat")

X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]


TPR = (TP/(TP+FN)) 
print("TPR: {:0.2f}".format(TPR))

TNR = (TN/(TN+FP)) 
print("TNR: {:0.2f}".format(TNR))

PPV = (TP/(TP+FP)) 
print("PPV: {:0.2f}".format(PPV))

NPV = (TN/(TN+FN)) 
print("NPV: {:0.2f}".format(NPV))

ACC = (TP+TN)/(TP+TN+FP+FN)
print("ACC: {:0.2f}".format(ACC))
