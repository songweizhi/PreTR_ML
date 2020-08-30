# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
import pandas
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# Load dataset
url = "/Users/songweizhi/Desktop/ML_wd/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# Split-out validation dataset
# http://scikit-learn.org/stable/model_selection.html
array = dataset.values
measurements = array[:, 0:4]
species = array[:, 4]
validation_size = 0.20
measurements_train, measurements_validation, species_train, species_validation = model_selection.train_test_split(measurements,
                                                                                                                  species,
                                                                                                                  test_size=0.20,
                                                                                                                  random_state=7)
# Evaluate model
cv_results = model_selection.cross_val_score(LinearDiscriminantAnalysis(),
                                             measurements_train,
                                             species_train,
                                             cv=model_selection.KFold(n_splits=10, random_state=7),
                                             scoring='accuracy')

message = "Evaluation:\nModel\tMean\tstd\n%s\t%f\t%f" % ('LDA', cv_results.mean(), cv_results.std())
print(message)


# Make predictions on validation dataset
LDA = LinearDiscriminantAnalysis()
LDA.fit(measurements_train, species_train)
species_predictions = LDA.predict(measurements_validation)
print('\nAccuracy_score: \n%s' % accuracy_score(species_validation, species_predictions))
print('\nClassification_report: \n%s' % classification_report(species_validation, species_predictions))
