import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix

#X = np.loadtxt("X_main.dat")
#y = np.loadtxt("y_main.dat")
#
#X = np.loadtxt("X_final.dat")
# y = np.loadtxt("y_final.dat")
##############################
X = np.loadtxt("data_ex/X_all_pqrst.dat")
y = np.loadtxt("data_ex/y_all_pqrst.dat")
############################## data scaling ##############################
# Normalize data ####{over all accuracy will be reduce}
#X = preprocessing.normalize(X, norm='l1')
X = preprocessing.normalize(X, norm='l2') #ektu better hoy

# Split the data into Training and Testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#print(X_test)

############################## Cross Validation ##############################
# classifier = svm.SVC(gamma='auto') #not good enough
# classifier = svm.SVC(kernel='linear') #need to run in lab
classifier = svm.LinearSVC()

scores = cross_val_score(classifier, X, y, cv=10, scoring = 'accuracy')
print("SVM: ")
print("CV score = ",scores.mean())

#################Fitting Classifier to the training set#################
# classifier = svm.LinearSVC()
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

#print("Y_pred: \n",y_pred)

# Making the confusion matrix 
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

