import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import tree
Accy = []
print("**************** Running Decision Tree ****************")
####################################### Disease: RBBB #########################
print("****************Disease: RBBB****************")
column = [
		16, 17, 18, 
		28, 29, 30, 
		40, 41, 42, 
		52, 53, 54, 
		64, 65, 66, 
		76, 77, 78, 
		88, 89, 90, 
		100, 101, 102, 
		112, 113, 114, 
		124, 125, 126, 
		136, 137, 138, 
		148, 149, 150,
		160,161,162, 
		170,171,172, 
		180,181,182, 
		190,191,192,
		200,201,202,
		210,211,212,
		220,221,222,
		230,231,232,
		240,241,242,
		250,251,252,
		260,261,262,
		270,271,272,
		]
##data load
X_train = np.loadtxt("X_RBBB.dat")
y_train = np.loadtxt("y_RBBB.dat")
X_test_main = np.loadtxt("X_test.dat")
y_test = np.loadtxt("y_test.dat")

#####################
X_test = np.array(X_test_main[:,column])
X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
##########################
print("Train:",X_train.shape)
print("Test:",X_test.shape)

classifier = tree.DecisionTreeClassifier(criterion="entropy")
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
Accy.append(ACC)
################################################################
####################################### Disease: Coronary Artery Block #########################
print("****************Disease: Coronary Artery Block****************")
column = [14,4,5,6,7,8,
			15, 17, 25, 26, 
			27, 29, 37, 38, 
			39, 41, 49, 50, 
			51, 53, 61, 62, 
			63, 65, 73, 74, 
			75, 77, 85, 86, 
			87, 89, 97, 98, 
			99, 101, 109, 110, 
			111, 113, 121, 122, 
			123, 125, 133, 134, 
			135, 137, 145, 146, 
			147, 149, 157, 158,
		160,161,162,165,166, 
		170,171,172,175,176, 
		180,181,182,185,186, 
		190,191,192,195,196,
		200,201,202,205,206,
		210,211,212,215,216,
		220,221,222,225,226,
		230,231,232,235,236,
		240,241,242,245,246,
		250,251,252,255,256,
		260,261,262,265,266,
		270,271,272,275,276,
		]
##data load
X_train = np.loadtxt("X_CAD.dat")
y_train = np.loadtxt("y_CAD.dat")
X_test_main = np.loadtxt("X_test.dat")
y_test = np.loadtxt("y_test.dat")

#####################
X_test = np.array(X_test_main[:,column])
X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
##########################
print("Train:",X_train.shape)
print("Test:",X_test.shape)


classifier = tree.DecisionTreeClassifier(criterion="entropy")
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
Accy.append(ACC)
################################################################
######################################## Disease: MI #########################
print("****************Disease: MI****************")
column = [14,4,5,6,7,8,
 			15, 17, 25, 26, 
 			27, 29, 37, 38, 
 			39, 41, 49, 50, 
 			51, 53, 61, 62, 
 			63, 65, 73, 74, 
 			75, 77, 85, 86, 
 			87, 89, 97, 98, 
 			99, 101, 109, 110, 
 			111, 113, 121, 122, 
 			123, 125, 133, 134, 
 			135, 137, 145, 146, 
 			147, 149, 157, 158,
 			160, 162, 166, 
 			170, 172, 176, 
 			180, 182, 186, 
 			190, 192, 196, 
 			200, 202, 206, 
 			210, 212, 216, 
 			220, 222, 226, 
 			230, 232, 236, 
 			240, 242, 246, 
 			250, 252, 256, 
 			260, 262, 266, 
 			270, 272, 276
 		]
##data load
X_train = np.loadtxt("X_MI.dat")
y_train = np.loadtxt("y_MI.dat")
X_test_main = np.loadtxt("X_test.dat")
y_test = np.loadtxt("y_test.dat")

#####################
X_test = np.array(X_test_main[:,column])
X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
##########################
print("Train:",X_train.shape)
print("Test:",X_test.shape)


classifier = tree.DecisionTreeClassifier(criterion="entropy")
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
Accy.append(ACC)
################################################################
######################################## Disease: Sinus tachycardy #########################
print("****************Disease: Sinus tachycardy****************")
column = [1,14,
# 		15, 16, 17, 
# 		27, 28, 29, 
# 		39, 40, 41, 
# 		51, 52, 53, 
# 		63, 64, 65, 
# 		75, 76, 77, 
# 		87, 88, 89, 
# 		99, 100, 101, 
# 		111, 112, 113, 
# 		123, 124, 125, 
# 		135, 136, 137, 
# 		147, 148, 149,
# 		160,161,162, 
# 		170,171,172, 
# 		180,181,182, 
# 		190,191,192,
# 		200,201,202,
# 		210,211,212,
# 		220,221,222,
# 		230,231,232,
# 		240,241,242,
# 		250,251,252,
# 		260,261,262,
# 		270,271,272,
 		]
##data load
X_train = np.loadtxt("X_TACHY.dat")
y_train = np.loadtxt("y_TACHY.dat")
X_test_main = np.loadtxt("X_test.dat")
y_test = np.loadtxt("y_test.dat")

#####################
X_test = np.array(X_test_main[:,column])
X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
##########################
print("Train:",X_train.shape)
print("Test:",X_test.shape)


classifier = tree.DecisionTreeClassifier(criterion="entropy")
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
Accy.append(ACC)
################################################################
######################################## Disease: Sinus bradycardy #########################
print("****************Disease: Sinus bradycardy****************")
column = [1,14,
# 		15, 16, 17, 
# 		27, 28, 29, 
# 		39, 40, 41, 
# 		51, 52, 53, 
# 		63, 64, 65, 
# 		75, 76, 77, 
# 		87, 88, 89, 
# 		99, 100, 101, 
# 		111, 112, 113, 
# 		123, 124, 125, 
# 		135, 136, 137, 
# 		147, 148, 149,
# 		160,161,162, 
# 		170,171,172, 
# 		180,181,182, 
# 		190,191,192,
# 		200,201,202,
# 		210,211,212,
# 		220,221,222,
# 		230,231,232,
# 		240,241,242,
# 		250,251,252,
# 		260,261,262,
# 		270,271,272,
 		]
##data load
X_train = np.loadtxt("X_BRADY.dat")
y_train = np.loadtxt("y_BRADY.dat")
X_test_main = np.loadtxt("X_test.dat")
y_test = np.loadtxt("y_test.dat")

#####################
X_test = np.array(X_test_main[:,column])
X_train = preprocessing.normalize(X_train, norm='l2')
X_test = preprocessing.normalize(X_test, norm='l2')

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
##########################
print("Train:",X_train.shape)
print("Test:",X_test.shape)


classifier = tree.DecisionTreeClassifier(criterion="entropy")
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
Accy.append(ACC)
################################################################
############################## Plot ##################################

plt.title('Decision Tree\n')
# Data to plot
labels = 'RBBB', 'Coronary Artery Block', 'Myocardial\nInfarction', 'Sinus Tachycardy', 'Sinus Bradycardy'
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'pink']
Bccy = list(Accy)
maxIndx = Bccy.index(max(Bccy))
high = [0, 0, 0, 0, 0]  # explode 1st slice
high[maxIndx] = 0.1
Bccy[maxIndx] = 0
maxIndx = Bccy.index(max(Bccy))
print(Bccy)
high[maxIndx] = 0.1
explode = tuple(high)

plt.tight_layout()
# Plot
patches = plt.pie(Accy, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
# plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.show()