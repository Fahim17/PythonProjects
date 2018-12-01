import matplotlib.pyplot as plt
import numpy as np

le = ['LR', 'NB', 'NN', "DT", "SVM", "ANN"]

# py = [
# 	[0.0, 0.079, 1.0],[0.0, 0.950, 1.0],
# 	[0.0, 0.214, 1.0],[0.0, 0.816, 1.0],
# 	[0.0, 0.214, 1.0],[0.0, 0.883, 1.0],
# 	[0.0, 0.286, 1.0],[0.0, 0.950, 1.0],
# 	[0.0, 0.214, 1.0],[0.0, 0.967, 1.0]
# ]     RBBB

roc_fpr = np.loadtxt("ROC_fpr.dat")
roc_tpr = np.loadtxt("ROC_tpr.dat")


t = 0
plt.plot([0,1],[0,1],linestyle = "--")

for i in range(6):
	fpr = roc_fpr[t]
	tpr = roc_tpr[t]
	plt.plot(fpr, tpr, label=le[i])
	t+=1


plt.legend(loc = 4)
plt.show()