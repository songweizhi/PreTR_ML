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
genomic_feature_matrix  = "/Users/songweizhi/Desktop/PreTR_ML/PreTR_ML_magnified.csv"
model_chosen            = 'LDA'  # Choose from
validation_size         = 0.20
test_size               = 0.20
random_state            = 7
n_splits                = 10


# read in dataframe
dataset = pandas.read_csv(genomic_feature_matrix, header=0)  # header=0 means the first line in the input file is column names
array = dataset.values
genome_cate = array[:, 0]
feature_matrix = array[:, 1:]


# what is the differences between validation_size and test_size ??????
features_train, features_validation, species_train, species_validation = model_selection.train_test_split(feature_matrix,
                                                                                                          genome_cate,
                                                                                                          test_size=test_size,
                                                                                                          random_state=random_state)
# Evaluate model
cv_results = model_selection.cross_val_score(DecisionTreeClassifier(),
                                             features_train,
                                             species_train,
                                             cv=model_selection.KFold(n_splits=n_splits, random_state=random_state),
                                             scoring='accuracy')

# Make predictions on validation dataset
DT = DecisionTreeClassifier()
DT.fit(features_train, species_train)
species_predictions = DT.predict(features_validation)


print('Model evaluation:')
print('Model:\t%s' % model_chosen)
print('Mean:\t%s' % cv_results.mean())
print('Std:\t%s' % cv_results.std())
print('Accuracy:\t%s' % accuracy_score(species_validation, species_predictions))
print('Classification_report: \n%s' % classification_report(species_validation, species_predictions))

