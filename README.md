## Problem Statement
We have a telecom firm which has collected data of all its customers. The main types of attributes are:

Demographics (age, gender etc.)
Services availed (internet packs purchased, special offers taken etc.)
Expenses (amount of recharge done per month etc.)
 

Based on all this past information, we want to build a model which will predict whether a particular customer will churn or not, i.e. whether they will switch to a different service provider or not. So the variable of interest, i.e. the target variable here is ‘Churn’ which will tell us whether or not a particular customer has churned. It is a binary variable - 1 means that the customer has churned and 0 means the customer has not churned.

This will involve all steps such as:

- Data cleaning and preparation
- Preprocessing steps
- Test-train split
- Feature scaling
- Model Building using RFE, p-values and VIFs

*I have created my own small library here called **regika.py** which contains basic functions that I generally use in model building*

## Model Building
A logistic regression model was built in Python using the function GLM() under statsmodel library. This model contained all the variables, some of which had insignificant coefficients. Hence, some of these variables were removed first based on an automated approach, i.e. RFE and then a manual approach based on VIF and p-value.