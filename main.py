from data_preprocess import merge_all_data, map_result_variable, handle_missing_values, map_categorical_variables, \
    get_features_df
from data_visualisation import visualise_eda, visualise_model_boxplots, plot_score_vs_time
from modeling import split_dataset, calculate_feature_importance, random_forest_classifier, svm_classifier, \
    mlp_classifier, svm_grid_search, dummy_classifier


def main():
    # Merge all relevent data
    print('Merging data')
    student_info_df = merge_all_data()
    # Map final outcome to required classifications
    student_info_df = map_result_variable(student_info_df)
    # Visualise exploratory data analysis
    print('Visualising EDA')
    visualise_eda(student_info_df)
    # Data Preprocessing
    student_info_df = handle_missing_values(student_info_df)
    student_info_df = map_categorical_variables(student_info_df)
    # Filter for relevent features
    print('Selecting features and splitting dataset')
    input_features, target_features, features_df = get_features_df(student_info_df)
    # Split dataset into 75 train 25 split ratio
    x_train, x_test, y_train, y_test = split_dataset(features_df[input_features], features_df[target_features])
    # Calculate feature importance by permutation importance on random forest classifier
    print('Calculating feature importance')
    calculate_feature_importance(x_train, x_test, y_train, y_test)
    # Model Random forest classifier and get scores
    print('Random forest model')
    rfc_cv_scores, rfc_test_recall, rfc_time = random_forest_classifier(x_train, x_test, y_train, y_test)
    # Model svm classifier and get scores
    print('SVM model')
    svm_cv_scores, svm_test_recall, svm_time = svm_classifier(x_train, x_test, y_train, y_test)
    # Model multi layer perceptron in keras
    print('MLP model')
    mlp_test_recall, mlp_time = mlp_classifier(x_train, x_test, y_train, y_test)
    # Model using dummy classifier for model comparison
    print('dummy model')
    dummy_cv_scores = dummy_classifier(x_train, y_train)
    # Visualise Model comparison  through boxplots
    print('visualising model comparison')
    visualise_model_boxplots(dummy_cv_scores, rfc_cv_scores, svm_cv_scores)
    # Plot model performance vs time taken to train model
    time_vals = [rfc_time, svm_time, mlp_time]
    recall_scores = [rfc_test_recall, svm_test_recall, mlp_test_recall]
    plot_score_vs_time(time_vals, recall_scores)
    # Hyperparameter tuning svm through grid search
    print('Grid search hyperparameter tuning for SVM')
    svm_grid_search(x_train, x_test, y_train, y_test)
    return


# Call the main function
main()
