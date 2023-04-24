import pandas as pd
from config import STUDENT_INFO, ASSESSMENTS, STUDENT_ASSESSMENTS, STUDENT_VLE, STUDENT_REGISTRATION


def process_assessment_data():
    assessments_df = pd.read_csv(ASSESSMENTS)
    student_assessment_df = pd.read_csv(STUDENT_ASSESSMENTS)
    # Excluding 'Exam' assessment which indicate final result
    assessments_df = assessments_df[assessments_df['assessment_type'] != 'Exam']
    student_assessment_df = student_assessment_df.dropna().reset_index(drop=True)
    # Joining student assessment_df to assessment_df, which will provide information on assessment deadline
    student_assessment_df = student_assessment_df.merge(assessments_df, on=['id_assessment'], how='inner')
    # Calculating relative date submit, when the student submitted with respect to deadline
    student_assessment_df['relative_date_submit'] = student_assessment_df.apply(
        lambda x: x['date'] - x['date_submitted'], axis=1)
    # Calculating average scores without weight
    student_average_score_df = student_assessment_df.groupby(['code_module', 'code_presentation', 'id_student'])[
        'score'].mean().reset_index()
    # Eliminating assessments which don't have a weight
    student_assessment_df = student_assessment_df[student_assessment_df['weight'] != 0].reset_index(drop=True)
    # Calculating weighted score
    student_assessment_df['weighted_score'] = student_assessment_df.apply(lambda x: x['score'] * (x['weight'] / 100),
                                                                          axis=1)
    # Grouping assessment dataframe by taking mean of relative date submit
    student_submit_df = \
        student_assessment_df.groupby(['code_module', 'code_presentation', 'id_student'])[
            'relative_date_submit'].mean().reset_index()
    # Grouping assessment dataframe by taking mean of student scores
    student_weighted_mean_score_df = \
        student_assessment_df.groupby(['code_module', 'code_presentation', 'id_student'])[
            'weighted_score'].mean().reset_index()
    # Merging submit data and means score data to get the final assessment data with required features
    student_scores_df = student_submit_df.merge(student_weighted_mean_score_df,
                                                on=['code_module', 'code_presentation', 'id_student'])
    student_scores_df = student_scores_df.merge(student_average_score_df,
                                                on=['code_module', 'code_presentation', 'id_student'])
    return student_scores_df


def process_student_vle_data():
    student_vle_df = pd.read_csv(STUDENT_VLE)
    # Grouping students to get total clicks done by the student in a module
    student_sum_click_df = student_vle_df.groupby(["code_module", "code_presentation", "id_student"])[
        'sum_click'].sum().reset_index()
    return student_sum_click_df


def map_result_variable(student_info_df):
    student_info_df['result_variable'] = student_info_df['final_result'].map(
        {'Withdrawn': 'Withdrawn',
         'Fail': 'Fail',
         'Pass': 'Pass',
         'Distinction': 'Pass'})
    student_info_df['result_variable_map'] = student_info_df['final_result'].map(
        {'Withdrawn': 0,
         'Fail': 1,
         'Pass': 2,
         'Distinction': 2})
    return student_info_df


def merge_all_data():
    student_info_df = pd.read_csv(STUDENT_INFO)
    print(student_info_df.shape)
    student_scores_df = process_assessment_data()
    student_info_df = student_info_df.merge(student_scores_df,
                                            on=['code_module', 'code_presentation', 'id_student'],
                                            how='left')
    student_sum_click_df = process_student_vle_data()
    student_info_df = student_info_df.merge(student_sum_click_df, on=['code_module', 'code_presentation', 'id_student'],
                                            how='left')
    student_info_df['sum_click'] = student_info_df['sum_click'].fillna(0)
    # check for assessmet submistion by creating a flag
    student_info_df['submit_map'] = 1
    student_info_df.loc[student_info_df['relative_date_submit'].isnull(), 'submit_map'] = 0
    student_info_df[['relative_date_submit']] = student_info_df[['relative_date_submit']].fillna(0)
    student_info_df[['score', 'weighted_score', 'sum_click']] = student_info_df[
        ['score', 'weighted_score', 'sum_click']].fillna(0)
    # Merge with student Registration data
    student_registration_df = pd.read_csv(STUDENT_REGISTRATION)
    student_registration_df = student_registration_df.drop(columns='date_unregistration')
    student_info_df = student_info_df.merge(student_registration_df,
                                            on=['code_module', 'code_presentation', 'id_student'], how='left')
    return student_info_df


def handle_missing_values(student_info_df):
    # Calculate the distribution of missing values
    norm_missing_val = student_info_df.isnull().sum().sort_values(ascending=False) / len(student_info_df)
    # Remove columns with more than 1% data missing
    features = norm_missing_val[norm_missing_val < 0.01].index.tolist()
    student_info_df = student_info_df[features]
    # Drop all further rows with missing values
    student_info_df = student_info_df.dropna().reset_index(drop=True)
    return student_info_df


def map_categorical_variables(student_info_df):
    student_info_df['gender_map'] = student_info_df['gender'].map({'M': 0, 'F': 1})
    student_info_df['disability_map'] = student_info_df['disability'].map({'N': 0, 'Y': 1})
    student_info_df['age_band_map'] = student_info_df['age_band'].map({'0-35': 25, '35-55': 45, '55<=': 65})
    student_info_df['highest_education_map'] = student_info_df['highest_education'].map(
        {'No Formal quals': 0,
         'Lower Than A Level': 1,
         'A Level or Equivalent': 2,
         'HE Qualification': 3,
         'Post Graduate Qualification': 4})
    # One hot encoding code module feature and dropping one column to avoid perfect multi collinearity
    dummy_vars = pd.get_dummies(student_info_df['code_module'], drop_first=True)
    student_info_df = pd.concat([student_info_df, dummy_vars], axis=1)
    return student_info_df


def get_features_df(student_info_df):
    # Excluding student id and region column, selecting the encoded and generated features.
    input_features = ['gender_map',
                      'disability_map',
                      'age_band_map',
                      'highest_education_map',
                      'studied_credits',
                      'num_of_prev_attempts',
                      'submit_map',
                      'sum_click',
                      'score',
                      'weighted_score',
                      'relative_date_submit',
                      'date_registration',
                      'BBB',
                      'CCC',
                      'DDD',
                      'EEE',
                      'FFF',
                      'GGG']
    target_feature = ['result_variable_map']
    features = input_features + target_feature
    features_df = student_info_df[features]
    return input_features, target_feature, features_df
