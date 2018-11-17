import matplotlib.pyplot as plt
import numpy as np

##*************************** Without Scaling CV score *******************************
n_groups = 5
algo_names = ['Logistic\nRegression', 'Naive Bayes', 'Decision Tree', "SVM", "Nearest Neighbour"]
score_wos = [0.72, 0.74, 0.77, 0.68, 0.67]
score_ws = [0.72, 0.74, 0.71, 0.78, 0.68]
index = np.arange(n_groups)
barWidth =0.4

plt.figure(figsize=(10, 8))
plt.ylim(0, 1) 
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Cross Validation Score')
# Text on the top of each barplot
for i in range(5):
	plt.text(x = i-0.045 , y = score_wos[i]+0.025, s = score_wos[i], size = 9)
	plt.text(x = i-0.045+barWidth , y = score_ws[i]+0.025, s = score_ws[i], size = 9) 
plt.xticks(index + (barWidth/2), algo_names)

plt.bar(algo_names, score_wos, width = barWidth, color = (0.2,0.1,0.9,0.7), label='Without Scaling')
plt.bar(index+barWidth, score_ws, width = barWidth, color = (0.1,0.9,0.1,0.7), label='With Scaling')
plt.legend()
plt.show()

##*************************** With Scaling CV score *******************************
score_wos = [0.63, 0.64, 0.78, 0.68, 0.65]
score_ws = [0.72, 0.68, 0.68, 0.74, 0.68]

plt.figure(figsize=(10, 8))
plt.ylim(0, 1) 
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Random Train-Test Split Score')
# Text on the top of each barplot
for i in range(5):
	plt.text(x = i-0.045 , y = score_wos[i]+0.025, s = score_wos[i], size = 9)
	plt.text(x = i-0.045+barWidth , y = score_ws[i]+0.025, s = score_ws[i], size = 9) 
plt.xticks(index + (barWidth/2), algo_names)

plt.bar(algo_names, score_wos, width = barWidth, color = (0.2,0.1,0.9,0.7), label='Without Scaling')
plt.bar(index+barWidth, score_ws, width = barWidth, color = (0.1,0.9,0.1,0.7), label='With Scaling')
plt.legend()
plt.show()






