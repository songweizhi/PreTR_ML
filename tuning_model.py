# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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
url = "/Users/songweizhi/Dropbox/Research/PreTR_ML/pattern_summary.csv"

names = ['R', 'ED', 'RED', 'IVYWREL', 'Hydrophobic', 'NQ_NQED', 'class']  # optimize

dataset = pandas.read_csv(url, names=names)
# Split-out validation dataset, http://scikit-learn.org/stable/model_selection.html
array = dataset.values
measurements = array[:, 0:6]  # optimize
species = array[:, 6]  # optimize
validation_size = 0.20
measurements_train, measurements_validation, species_train, species_validation = model_selection.train_test_split(measurements,
                                                                                                                  species,
                                                                                                                  test_size=0.20,
                                                                                                                  random_state=7)
# Evaluate model
cv_results = model_selection.cross_val_score(DecisionTreeClassifier(),
                                             measurements_train,
                                             species_train,
                                             cv=model_selection.KFold(n_splits=10, random_state=7),
                                             scoring='accuracy')

message = "Evaluation:\nModel\tMean\tstd\n%s\t%f\t%f" % ('LDA', cv_results.mean(), cv_results.std())
print(message)


# Make predictions on validation dataset
DT = DecisionTreeClassifier()
DT.fit(measurements_train, species_train)
species_predictions = DT.predict(measurements_validation)
print('\nAccuracy_score: \n%s' % accuracy_score(species_validation, species_predictions))
print('\nClassification_report: \n%s' % classification_report(species_validation, species_predictions))



