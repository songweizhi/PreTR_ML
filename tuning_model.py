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
input_data = "/Users/songweizhi/Dropbox/Research/PreTR_ML/pattern_summary_with_head.csv"

dataset = pandas.read_csv(input_data, header=0)  # header=0 means the first line in the input file is column names
array = dataset.values
row_num = array.shape[0]
col_num = array.shape[1]

measurements = array[:, 0:(col_num - 1)]
species = array[:, (col_num - 1)]
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
