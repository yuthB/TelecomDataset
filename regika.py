import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Checking value_counts for each object type column in dataframe 'df'
def column_values(df):
    column_val=[]
    for i in df.columns:
        if df[i].dtype=='O':
            column_val.append(df[i].value_counts())
    return column_val

# changing 'yes-no' type categories to '1-0'
def convert_binary(df, varlist):
    df[varlist]=df[varlist].apply(lambda x:x.map({'yes':1,'Yes':1,'no':0,'No':0}))

# Calculating Variance Inflation Factor
def vif(X):
    vif=pd.DataFrame()

    vif['Features']=X.columns
    vif['vif']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    vif['vif']=round(vif['vif'],2)
    vif=vif.sort_values(by='vif',ascending=False)
    return vif

# Calculating Paramerters of a Logistic Regression Model
def log_model_params(y_actual,y_predicted):
    from sklearn import metrics
    confusion=metrics.confusion_matrix(y_actual,y_predicted)
    TP=confusion[1,1]
    FP=confusion[0,1]
    TN=confusion[0,0]
    FN=confusion[1,0]
    accuracy=(TP+TN)/float(TP+TN+FP+FN)*100    
    sensitivity=TP/float(FN+TP)*100
    specificity=TN/float(TN+FP)*100
    FP_rate=100-specificity
    Positive_pred_value=TP/float(TP+FP)
    Negative_pred_value=TN/float(TN+FN)
    precision=TP/float(TP+FP)*100
    params={'accuracy':accuracy,'TN':TN,'FP':FP,'FN':FN,'TP':TP,'sensitivity/recall':sensitivity,'specificity':specificity,'precision':precision,'False_Positive_Rate':FP_rate,'Positve_Prediction_value':Positive_pred_value,'Negative_Prediction_value':Negative_pred_value}
    return params