import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualise_result_dist(student_info_df):
    fig = plt.figure(figsize=(9, 7))
    result_dist = student_info_df['result_variable'].value_counts(normalize=True) * 100
    # Plot barchart
    plt.bar(result_dist.index, result_dist.values)
    # Set the title and axis labels
    plt.title('Percentage of each result category')
    plt.xlabel('Result Category')
    plt.ylabel('Percentage')
    plt.savefig('saved_figs/result_distribution.png', bbox_inches='tight')
    return


def visualise_missing_vals(student_info_df):
    fig = plt.figure(figsize=(9, 7))
    missing_values = student_info_df.isnull().sum() / len(student_info_df) * 100
    missing_values = missing_values[missing_values > 0]
    # Plot barchart
    plt.bar(missing_values.index, missing_values.values)
    # Set the title and axis labels
    plt.title('Percentage of Missing Values')
    plt.xlabel('Column')
    plt.ylabel('Percentage')
    plt.savefig('saved_figs/missing_val_distribution.png', bbox_inches='tight')
    return


def visualise_variable_boxplot(student_info_df):
    numeric_columns_df = student_info_df.select_dtypes(include=['int64', 'float64'])
    # remove irrelevant features
    numeric_columns_df = numeric_columns_df.drop(columns=['id_student', 'submit_map', 'result_variable_map'])
    # Set figure dimensions
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    # Plot boxplot
    numeric_columns_df.boxplot()
    # set the plot title and axis labels
    ax.set_title('Box plots of Different Columns')
    ax.set_xlabel('Variable')
    ax.set_ylabel('Value')
    fig.savefig('saved_figs/variable_boxplot.png', bbox_inches='tight')
    # create a list of the numeric columns to be use the in pairplot
    numeric_columns = numeric_columns_df.columns.tolist()
    return numeric_columns


def visualise_variable_pairplot(student_info_df, numeric_columns):
    # append result_variable in columns to add for hue
    plot_columns = numeric_columns.copy()
    plot_columns.append('result_variable')
    # create a pair plot of the numeric columns
    sns.pairplot(data=student_info_df[plot_columns], hue='result_variable', diag_kind='kde')
    # display the plot
    plt.savefig('saved_figs/variable_pairplot.png', bbox_inches='tight')
    return


def plot_feature_dist(student_info_df, feature):
    # Convert data into crosstab and plot
    pd.crosstab(student_info_df[feature], student_info_df['result_variable']).apply(lambda x: (x / sum(x)) * 100,
                                                                                    axis=1).plot.bar(
        figsize=(16, 9))
    plt.savefig(f'saved_figs/{feature}_distribution.png', bbox_inches='tight')
    return


def visualise_categorical_dist(student_info_df):
    # Plot feature distribution for each categorical variable
    categorical_features = ['code_module', 'gender', 'highest_education', 'region']
    for feature in categorical_features:
        plot_feature_dist(student_info_df, feature)
    return


def visualise_numeric_correlation(student_info_df, numeric_columns):
    fig = plt.figure(figsize=(9, 7))
    # select numeric_features
    plot_columns = numeric_columns.copy()
    plot_columns.append('result_variable_map')
    # Plot correlation heatmap
    sns.heatmap(student_info_df[plot_columns].corr(), annot=True)
    fig.savefig(f'saved_figs/correlation_heatmap.png', bbox_inches='tight')
    return


def visualise_eda(student_info_df):
    visualise_result_dist(student_info_df)
    visualise_missing_vals(student_info_df)
    numeric_columns = visualise_variable_boxplot(student_info_df)
    visualise_variable_pairplot(student_info_df, numeric_columns)
    visualise_categorical_dist(student_info_df)
    visualise_numeric_correlation(student_info_df, numeric_columns)
    return student_info_df


def visualise_feature_importance(importance_scores, feature_names):
    # Sort feature importance scores
    sorted_idx = importance_scores.argsort()
    importance_scores_sorted = importance_scores[sorted_idx]
    feature_names_sorted = [feature_names[i] for i in sorted_idx]
    # Plot feature importance scores
    fig, ax = plt.subplots()
    ax.barh(feature_names_sorted, importance_scores_sorted)
    ax.set_xlabel('Importance score')
    ax.set_title('Permutation Feature Importance')
    fig.savefig(f'saved_figs/feature_importance.png', bbox_inches='tight')
    return


def visualise_model_boxplots(dummy_cv_scores, rfc_cv_scores, svm_cv_scores):
    # Create a list of the data to plot
    data_to_plot = [dummy_cv_scores, rfc_cv_scores, svm_cv_scores]
    # Create a list of the boxplot labels
    labels = ['Dummy', 'Random forest', 'SVM']
    # Create the boxplot
    fig, ax = plt.subplots()
    ax.boxplot(data_to_plot, labels=labels)
    # Add a title and axis labels
    ax.set_title('Model comparison with boxplots')
    ax.set_xlabel('Boxplots')
    ax.set_ylabel('Recall scores')
    fig.savefig(f'saved_figs/model_boxplots.png', bbox_inches='tight')
    return


def plot_score_vs_time(time_vals, recall_scores):
    fig, ax = plt.subplots()
    # Define the colors and labels for each point
    colors = ['r', 'g', 'b']
    labels = ['rfc', 'svm', 'mlp']
    # Plot the points with different colors and labels
    for i in range(len(time_vals)):
        ax.scatter(time_vals[i], recall_scores[i], c=colors[i], label=labels[i])
    # Add a legend to the plot
    ax.legend()
    # Give title
    ax.set_title('Recall vs Time plot')
    ax.set_xlabel('time')
    ax.set_ylabel('Recall scores')
    fig.savefig(f'saved_figs/score_vs_time_plot.png', bbox_inches='tight')
    return

