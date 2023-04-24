
# Data File paths
STUDENT_INFO = 'oulad_dataset/studentInfo.csv'
COURSES = 'oulad_dataset/courses.csv'
ASSESSMENTS = 'oulad_dataset/assessments.csv'
STUDENT_ASSESSMENTS = 'oulad_dataset/studentAssessment.csv'
VLE = 'oulad_dataset/vle.csv'
STUDENT_VLE = 'oulad_dataset/studentVle.csv'
STUDENT_REGISTRATION = 'oulad_dataset/studentRegistration.csv'

# cross_validation_folds
folds = 5

# hyperparameters
svm_fold = 3
svm_param_grid = {'svc__C': [1, 10, 100], 'svc__gamma': [1, 0.1, 0.01], 'svc__kernel': ['rbf']}
