import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("loan_approval_dataset.csv")

df.columns = df.columns.str.strip()#remove extra spaces before column name
print(df['education'].unique())

df.drop(columns=['loan_id'],inplace=True)

le = LabelEncoder()

df['loan_status'] = le.fit_transform(df['loan_status'])

categorical_columns =['education','self_employed']

numeric_columns = ['no_of_dependents','income_annum','loan_amount','loan_term','cibil_score',
                   'residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']


ct = ColumnTransformer(transformers=[
    ("cat",OneHotEncoder(),categorical_columns),
    ("num",StandardScaler(),numeric_columns)
])

#Split into independent and dependent variables
x = df.iloc[:,:-1]
y = df['loan_status']

x_transformed = ct.fit_transform(x)

#Split into train and test split
x_train,x_test,y_train,y_test = train_test_split(x_transformed,y,test_size=0.25,random_state=1)


dt = DecisionTreeClassifier(random_state=1)

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)



'''''
#train models
pipelines = {
    'lr': make_pipeline(LogisticRegression()),
    'dt':make_pipeline(DecisionTreeClassifier()),
    'rf':make_pipeline(RandomForestClassifier()),
    'NB':make_pipeline(GaussianNB()),
    'knn':make_pipeline(KNeighborsClassifier())
}


#get confusion matrix for each algorithm
for model_name,model_pipelines in pipelines.items():
    model_pipelines.fit(x_train,y_train)

    y_pred = model_pipelines.predict(x_test)

    #acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    print(f"{model_name} has {cm} score")
    sns.heatmap(cm,annot= True,fmt='d',cmap='Blues',xticklabels=['class0','class1'],yticklabels=['class0','class1'])
    plt.title(model_name)
    plt.show()

'''

with open('model.pkl','wb') as f:
    pickle.dump(dt,f)

with open('preprocessor.pkl','wb')as pf:
    pickle.dump(ct,pf)


with open("label_encoder.pkl", "wb") as lf:
    pickle.dump(le,lf)  # Load the label encoder used for 'loan_status'
