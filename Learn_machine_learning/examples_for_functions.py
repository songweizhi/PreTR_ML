
labels =      [0, 1, 1, 1, 2, 2, 3, 3, 4]
predictions = [0, 2, 1, 1, 1, 2, 3, 4, 4]


# accuracy_score
from sklearn.metrics import accuracy_score
accuracy_score_normalized = accuracy_score(labels, predictions, normalize=True)
accuracy_score_unnormalized = accuracy_score(labels, predictions, normalize=False)
print('Accuracy_score_normalized: %s' % accuracy_score_normalized)
print('\nAccuracy_score_unnormalized: %s' % accuracy_score_unnormalized)


# classification_report
from sklearn.metrics import classification_report
classification_report = classification_report(labels, predictions)
print('\nClassification_report: \n%s' % classification_report)


# confusion_matrix
# By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations known to be
# in group i but predicted to be in group j.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(labels, predictions)
print('\nConfusion_matrix: \n%s' % confusion_matrix)






