import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from data_visualisation import visualise_feature_importance
from config import folds, svm_param_grid, svm_fold

# initialise required methods
scaler = StandardScaler()
smote = SMOTE(random_state=23)
rfc = RandomForestClassifier(random_state=23)
svm = SVC(kernel='rbf', random_state=23)
dummy_clf = DummyClassifier(strategy='stratified', random_state=23)


def split_dataset(input_features, target_feature):
    # splits dataset into  train and test
    x_train, x_test, y_train, y_test = train_test_split(input_features, target_feature, test_size=0.25, random_state=23)
    return x_train, x_test, y_train, y_test


def calculate_feature_importance(x_train, x_test, y_train, y_test):
    # create a pipeline with StandardScaler and RandomForestClassifier
    pipeline = make_pipeline(scaler, rfc)
    # fit the pipeline on the training data
    pipeline.fit(x_train, y_train.values.ravel())
    # calculate feature importance
    result = permutation_importance(pipeline, x_test, y_test, scoring='recall_macro', n_repeats=10, random_state=23)
    importance_scores = result.importances_mean
    feature_names = list(x_test.columns)
    visualise_feature_importance(importance_scores, feature_names)
    return


def fit_model_get_scores(pipeline, x_train, x_test, y_train, y_test):
    # model and calculate training time
    start_time = time.time()
    cv_scores = cross_val_score(pipeline, x_train, y_train.values.ravel(), cv=folds, scoring='recall_macro')
    cv_mean = cv_scores.mean()
    pipeline.fit(x_train, y_train.values.ravel())
    end_time = time.time()
    total_time = end_time - start_time
    y_pred = pipeline.predict(x_test)
    # Calculate the recall score on the test data
    test_recall = recall_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    return cv_scores, cv_mean, test_recall, total_time, report


def write_report_to_file(classifier, cv_mean, test_recall, total_time, report, run=0):
    # Write results into a file
    x = 'a'
    if run == 1:
        x = 'w'
    with open('classification_results.txt', x) as f:
        f.write(f'### {classifier.upper()} ###\n\n')
        f.write(f'recall mean score on train set: {cv_mean}\n')
        f.write(f'recall score on test set: {test_recall}\n')
        f.write(f'Training time: {total_time} seconds\n')
        f.write('Classification report:\n')
        f.write(report)
        f.write("\n\n")


def random_forest_classifier(x_train, x_test, y_train, y_test):
    # Initialise rfc pipeline that applies smote, scaler and rfc model
    rfc_pipeline = make_pipeline(smote, scaler, rfc)
    # fit model and calculate scores through function
    rfc_cv_scores, rfc_cv_mean, rfc_test_recall, rfc_time, rfc_report = \
        fit_model_get_scores(rfc_pipeline, x_train, x_test, y_train, y_test)
    # print results
    print('### RANDOM FOREST ###')
    print('recall mean score on train set:', rfc_cv_mean)
    print('recall score on test set:', rfc_test_recall)
    print('Training time:', rfc_time, 'seconds')
    print(rfc_report)
    write_report_to_file("random forest", rfc_cv_mean, rfc_test_recall, rfc_time, rfc_report, 1)
    return rfc_cv_scores, rfc_test_recall, rfc_time


def svm_classifier(x_train, x_test, y_train, y_test):
    # Initialise rfc pipeline that applies smote, scaler and svm model
    svm_pipeline = make_pipeline(smote, scaler, svm)
    # fit model and calculate scores through function
    svm_cv_scores, svm_cv_mean, svm_test_recall, svm_time, svm_report = \
        fit_model_get_scores(svm_pipeline, x_train, x_test, y_train, y_test)
    # print results
    print('### SVM ###')
    print('recall mean score on train set:', svm_cv_mean)
    print('recall score on test set:', svm_test_recall)
    print('Training time:', svm_time, 'seconds')
    print(svm_report)
    write_report_to_file("svm", svm_cv_mean, svm_test_recall, svm_time, svm_report)
    return svm_cv_scores, svm_test_recall, svm_time


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def mlp_classifier(x_train, x_test, y_train, y_test):
    # Create sequential model
    def create_model():
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(18,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', recall_m])
        return model

    # change the structure of target variable to one hot encoded
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    mlp = create_model()
    # Calculate time and fit the model
    start_time = time.time()
    mlp.fit(x_train, y_train, epochs=50, batch_size=32)
    end_time = time.time()
    mlp_time = end_time - start_time
    # Get the predicted class labels for the test data
    y_pred = np.argmax(mlp.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    training_time = end_time - start_time
    # Get recall and classification report
    mlp_test_recall = recall_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    # print results
    print('### MLP ###')
    print('recall score on test set:', mlp_test_recall)
    print('Training time:', mlp_time, 'seconds')
    print('Classification report:\n', report)
    with open('classification_results.txt', 'a') as f:
        f.write(f'### MLP ###\n\n')
        f.write(f'recall score on test set: {mlp_test_recall}\n')
        f.write(f'Training time: {training_time} seconds\n')
        f.write('Classification report:\n')
        f.write(report)
        f.write("\n\n")
    return mlp_test_recall, mlp_time


def svm_grid_search(x_train, x_test, y_train, y_test):
    svm_pipeline = make_pipeline(smote, scaler, svm)
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=svm_fold, scoring='recall_macro')
    svm_grid.fit(x_train, y_train.values.ravel())
    best_params = svm_grid.best_params_
    best_train_score = svm_grid.best_score_
    best_param_test_score = svm_grid.score(x_test, y_test)
    print(f"Best params after hypertuning SVM {best_params}")
    print(f"Train score on best params {best_train_score}")
    print(f"Test score on best params {best_param_test_score}")
    with open('classification_results.txt', 'a') as f:
        f.write(f'### SVM hypertuning ###\n\n')
        f.write(f"Best params after hypertuning SVM {best_params}\n")
        f.write(f"Train score on best params {best_train_score}\n")
        f.write(f"Test score on best params {best_param_test_score}")
        f.write("\n\n")


def dummy_classifier(x_train, y_train):
    dummy_cv_scores = cross_val_score(dummy_clf, x_train, y_train.values.ravel(), cv=folds, scoring='recall_macro')
    return dummy_cv_scores
