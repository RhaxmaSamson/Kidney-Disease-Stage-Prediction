mport warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Suppressing FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Importing Dataset:
dataset = pd.read_csv("kidney_data1.csv")

# Dropping unnecessary feature:
dataset.drop('id', axis=1, inplace=True)

# Replacing Categorical Values with Numericals
replace_values = {'rbc': {'normal': 0, 'abnormal': 1},
                  'pc': {'normal': 0, 'abnormal': 1},
                  'pcc': {'notpresent': 0, 'present': 1},
                  'ba': {'notpresent': 0, 'present': 1},
                  'htn': {'yes': 1, 'no': 0},
                  'dm': {'yes': 1, 'no': 0},
                  'cad': {'yes': 1, 'no': 0},
                  'appet': {'good': 1, 'poor': 0, 'no': np.nan},
                  'pe': {'yes': 1, 'no': 0},
                  'ane': {'yes': 1, 'no': 0},
                  'classification': {'ckd\t': 'ckd'}
                  }

dataset.replace(replace_values, inplace=True)

# Converting Objective into Numericals:
dataset['eGFR'] = pd.to_numeric(dataset['eGFR'], errors='coerce')
dataset['ckd_stage'] = dataset['ckd_stage'].map({'Stage 1': 1, 'Stage 2': 2, 'Stage 3': 3, 'Stage 4': 4, 'Stage 5': 5,
                                                 'stage1': 1, 'stage2': 2, 'stage3a': 3, 'stage3b': 3, 'stage4': 4, 'stage5': 5})

# Converting columns to numeric types
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
                'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
                'appet', 'pe', 'ane', 'eGFR', 'ckd_stage']

dataset[numeric_cols] = dataset[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Handling Missing Values:
# Impute missing values with mode for categorical variables and with median for numerical variables
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
    else:
        dataset[col] = dataset[col].fillna(dataset[col].median())

# Dropping feature (Multicollinearity):
dataset.drop('pcv', axis=1, inplace=True)

# Independent and Dependent Features:
X = dataset[['age', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc', 'sg','bp','bgr','eGFR','sc','sod']]
y = dataset['classification']

# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
print(X_train.head())  # Print the first few rows
print(X_train.info())  # Print information about the DataFrame


# RandomForestClassifier:
RandomForest = RandomForestClassifier()
RandomForest.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'Kidney.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))

