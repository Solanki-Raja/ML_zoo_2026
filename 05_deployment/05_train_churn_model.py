import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score
from tqdm.auto import tqdm

import pickle


# parameters 

C_regularized=[0.01, 0.001, 0.0001, 0.00001, 0.1, 1.0]

best_C = None
best_mean = -1
best_std = float("inf")


# Data Loading

df=pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# Data Prepration 

df.columns=df.columns.str.lower().str.replace(' ', '_')
categorical_variables=list((df.dtypes[df.dtypes=='str'].index).drop('totalcharges'))

for c in categorical_variables:
    df[c]=df[c].str.lower().str.replace(' ', '_')
    df[c]=df[c].fillna('NA')

categorical_variables.remove('churn')
df['totalcharges']=pd.to_numeric(df['totalcharges'].str.strip().str.replace(' ', ''), errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
numerical_variable=list(df.dtypes[((df.dtypes=='int64') | (df.dtypes=='float64'))].index)
df.churn=(df.churn=='yes').astype('int')


# Train test Split

df_full_train,df_test=train_test_split(
    df,
    test_size=0.2,
    random_state=1
)
df_train,df_val=train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=1
)

df_full_train=df_full_train.reset_index(drop=True)
df_train=df_train.reset_index(drop=True)
df_val=df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)

y_full_train=df_full_train.churn.values
y_train=df_train.churn.values
y_val=df_val.churn.values
y_test=df_test.churn.values


# Training  and Predicting 

def train(df,y_train, C):
    
    dicts=df[categorical_variables+numerical_variable].to_dict(orient="records")
    dv=DictVectorizer(sparse=False)
    
    X=dv.fit_transform(dicts)
    model=LogisticRegression(C=C, max_iter= 1000)
    model.fit(X,y_train)
    
    return dv, model


def predict(df, dv, model):
    
    dicts=df[categorical_variables+numerical_variable].to_dict(orient="records")
    X=dv.transform(dicts)
    
    y_pred=model.predict_proba(X)[:,1]
    
    return y_pred


#  Cross-Validation 

def cross_validation(df, number_of_splits, shuffle, random_state , C):
    
    score=[]
    kfold=KFold(n_splits=number_of_splits,
                shuffle=shuffle,
                random_state=random_state
                )
    for train_idx, val_idx in tqdm(kfold.split(df),total=number_of_splits):
        
        df_train=df.iloc[train_idx]
        df_val=df.iloc[val_idx]
        
        y_train=df_train.churn.values
        y_val=df_val.churn.values
        
        
        dv, model=train(df_train, y_train, C)
        y_pred= predict(df_val, dv, model)
        
        auc=roc_auc_score(y_val, y_pred)
        score.append(auc)
    
    return score        


cros_score={}
for C in C_regularized:
    score=cross_validation(df=df_full_train,number_of_splits=5, shuffle=True, random_state= 1 , C=C)
    cros_score[C]= {'mean': float(round(np.mean(score),3)), 'std': float(round(np.std(score),3))}
    



for C, r in cros_score.items():
    mean = round(r["mean"], 3)
    std = round(r["std"], 3)

    if mean > best_mean or (mean == best_mean and std < best_std):
        best_C = C
        best_mean = mean
        best_std = std


# Final Call
dv, model=train(df_full_train, y_full_train, best_C)
y_pred= predict(df_test,dv,model)

auc=roc_auc_score(y_test,y_pred)
auc


# Saving the Model

output_file = f'model_c={best_C}.bin'
# f_out= open(output_file, 'wb')
# pickle.dump((dv,model), f_out)
# f_out.close()


with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)
    



