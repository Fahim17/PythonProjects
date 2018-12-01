import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics as ms
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# X = np.loadtxt("X_main.dat")
# y = np.loadtxt("y_main.dat")
##############################
# X = np.loadtxt("X_final.dat")
# y = np.loadtxt("y_final.dat")
# ##############################
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
classifier = LogisticRegression(random_state = 0)
scores = cross_val_score(classifier, X, y, cv=10, scoring = 'accuracy')
print("Logistic Regression:")
print("CV score = ",scores.mean())
#################Fitting logistic regression to the training set#################
# classifier = LogisticRegression(random_state = 0)
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

unique, counts = np.unique(y_train, return_counts=True)
unique = [0,1]
oz_count = dict(zip(unique, counts))
print("datacount: ",oz_count)
# ###################### SMOTE #############################
# print("After using SMOTE:")
# smt = SMOTE(random_state=12, ratio = 1.0)
# # Xs, ys = smt.fit_sample(X, y)
# X_train, y_train = smt.fit_sample(X_train, y_train)
# unique, counts = np.unique(y_train, return_counts=True)
# unique = [0,1]
# oz_count = dict(zip(unique, counts))
# print(oz_count)

# # X_train,X_test,y_train,y_test = train_test_split(Xs,ys,test_size=0.25,random_state=0)

# # Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

# scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring = 'accuracy')

# print("CV score = ",scores.mean())

# classifier.fit(X_train,y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# #print("Y_pred: \n",y_pred)

# # Making the confusion matrix 
# cm = confusion_matrix(y_test,y_pred)
# print(cm)

# TN = cm[0,0]
# FP = cm[0,1]
# FN = cm[1,0]
# TP = cm[1,1]


# TPR = (TP/(TP+FN)) 
# print("TPR: {:0.2f}".format(TPR))

# TNR = (TN/(TN+FP)) 
# print("TNR: {:0.2f}".format(TNR))

# PPV = (TP/(TP+FP)) 
# print("PPV: {:0.2f}".format(PPV))

# NPV = (TN/(TN+FN)) 
# print("NPV: {:0.2f}".format(NPV))

# ACC = (TP+TN)/(TP+TN+FP+FN)
# print("ACC: {:0.2f}".format(ACC))

 ############################## ############################## ##############################

# fpr, tpr, thresholds = ms.roc_curve(y_test, y_pred)

# print(list(fpr))
# print(list(tpr))

# np.savetxt('ROC_fpr.dat', fpr, fmt='%.3e')
# np.savetxt('ROC_tpr.dat', tpr, fmt='%.3e')


# plt.plot(fpr, tpr)
# plt.show()