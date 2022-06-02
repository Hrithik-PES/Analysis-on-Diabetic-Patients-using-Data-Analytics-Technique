import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/User/Documents/Data_Analytics_Project/dataset_diabetes/final_eda_dataset.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(df)

features = list(df.axes[1])[2:48]

encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(df[features])
diabetic_df_encoded = pd.DataFrame(data_encoded, columns=features)
#print(data_encoded)

encoder = LabelEncoder()
target_encoded = encoder.fit_transform(df['readmitted'])
diabetic_df_encoded['readmitted'] = target_encoded
encoder.inverse_transform(target_encoded)

print(diabetic_df_encoded)

#X_train, X_test, y_train, y_test = train_test_split(diabetic_df_encoded.drop('readmitted', axis=1), diabetic_df_encoded['readmitted'], test_size=0.3, random_state=143)

X_train = diabetic_df_encoded.iloc[0:65000, 0:46]
X_test = diabetic_df_encoded.iloc[65000:65514, 0:46]
y_train = diabetic_df_encoded.iloc[0:65000, -1:]
y_test = diabetic_df_encoded.iloc[65000:65514, -1:]


#X_train, X_test, y_train, y_test = train_test_split(diabetic_df_encoded.iloc[:, 0:46], diabetic_df_encoded.iloc[:, -1:], test_size = 0.30)
# NAIVE BAYES 
cnb = CategoricalNB()
cnb.fit(X_train, y_train)
y_pred_cnb = cnb.predict(X_test)
print(' \n accuracy of naive bayes is : ', accuracy_score(y_test, y_pred_cnb))

# KNN 
knn = KNeighborsClassifier(n_neighbors=200)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(' \n accuracy of knn is ',accuracy_score(y_test, y_pred_knn))

#print(y_pred_knn)
#print(list(y_test['readmitted']))
z = 0
for i,j in (zip(list(y_pred_knn)[0:15], list(y_test['readmitted'])[0:15])):
    if (i == j):
        print('X - Test : ', X_test.iloc[z])
        if (i == 0):
            print(' Change in the treatment of the patient (Treatment was not proper) \n')
        elif (i == 1):
            print(' Small Change in the treatment of the patient due to patent health condition\n')
        else:
            print(' Treatment was good, so patient was not readmitted\n')
    z = z + 1        
    print('\n')
# RANDOM FOREST
clf = RandomForestClassifier(n_estimators = 105)
clf.fit(X_train, y_train)
y_pred_rfc = clf.predict(X_test)
print(' \n accuracy of random forest is ',accuracy_score(y_test, y_pred_rfc))