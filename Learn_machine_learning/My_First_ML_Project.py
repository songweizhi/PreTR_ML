
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Check the versions of libraries

# Python version
# import sys
#
# print('Python: {}'.format(sys.version))
# # scipy
# import scipy
#
# print('scipy: {}'.format(scipy.__version__))
# # numpy
# import numpy
#
# print('numpy: {}'.format(numpy.__version__))
# # matplotlib
# import matplotlib
#
# print('matplotlib: {}'.format(matplotlib.__version__))
# # pandas
# import pandas
#
# print('pandas: {}'.format(pandas.__version__))
# # scikit-learn
# import sklearn
#
# print('sklearn: {}'.format(sklearn.__version__))


# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "/Users/songweizhi/Desktop/ML_wd/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#print(dataset)


# shape
#print(dataset.shape)

# # print the first 20 rows
# print(dataset.head(20))
#
# # summary
# print(dataset.describe())
#
# # class distribution
# print(dataset.groupby('class').size())
#
#
# # box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
#
#
# # histograms
# dataset.hist()
# plt.show()
#
#
# # scatter plot matrix
# scatter_matrix(dataset)
# plt.show()


# Split-out validation dataset
array = dataset.values
# print(dataset)
# print(array)

measurements = array[:, 0:4]
species = array[:, 4]
validation_size = 0.20

measurements_train, measurements_validation, species_train, species_validation = model_selection.train_test_split(measurements, species, test_size=0.20, random_state=7)

# print(measurements_train)
# print(measurements_validation)
# print(species_train)
# print(species_validation)

print('measurements_train: %s' % len(measurements_train))
print('measurements_validation: %s' % len(measurements_validation))
print('species_train: %s' % len(species_train))
print('species_validation: %s' % len(species_validation))


# Spot Check Algorithms
models = [('LR', LogisticRegression()),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC())]

# Train each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, measurements_train, species_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #print(name)
    #print(cv_results)
    print(msg)
    #print('\n')

#
# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(measurements_train, species_train)
predictions = knn.predict(measurements_validation)
print(accuracy_score(species_validation, predictions))
print(classification_report(species_validation, predictions))


