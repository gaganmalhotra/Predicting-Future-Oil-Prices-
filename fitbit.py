#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitbit Data Analysis

@author: Gagan Malhotra
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error

# Setup paths for the input data
path = "PET_PRI_SPT_S1_M.xls"
# Path to save the test results of predicted next 6 months prices
test_path = "testResults.csv"

#Array of Dataframes input
data = []

for i in range(8):
    if(i!=0):
        data.append(pd.read_excel(path, sheetname = i, skiprows = [0,1]))

#result storing average value and price change per year
resultdf = []

#result storing average value and price change per year
resultdf_monthly = []

#Types of Data Available
types = ['Crude Oil', 'Conventional Gasoline', 'RBOB Regular Gasoline', 'Heating Oil', 'Diesel Fuel', 'Kerosene Type Jet Fuel', 'Propane']


#### This method applies the initial analysis of Finding Monthly and Yearly Change in Price
def analyze(data2, ind):
    monthList = []
    yearList = []
    monthDiff = []
    nextData = []
    row_iterator = data2.iterrows()
    
    str_cols = data2.columns[data2.dtypes==object]
    data2[str_cols] = data2[str_cols].fillna('.')
    data2.fillna(0,inplace=True)
    
    for index, row in row_iterator:
        current_date = row['Date'] 
        monthList.append(current_date.month)
        yearList.append(current_date.year)
        if(index == len(data2)-1):
            nextData.append(data2.iloc[index][1])
            monthDiff.append(0)
        else:
            nextData.append(data2.iloc[index+1][1])
            monthDiff.append(data2.iloc[index+1][1] - row[1])
    
    data2['Month'] = monthList
    data2['Year'] = yearList
    data2['Monthly Change in Price'] = monthDiff     
        
           
    #generate map from keys
    map_month = dict.fromkeys(data2.Month.unique())
    
    for month in data2.Month.unique():
        df1 = data2.loc[data2['Month'] == month]
        print('*******')
        print('For month', month)
        print(df1.iloc[:,1])
        mean_mnth = np.asarray(df1.iloc[:,1]).mean()
        print('mean:',mean_mnth )
        map_month[month] = mean_mnth
    
    yearls_m = list(map_month.keys())
    print(map_month)

    meanls_m = list(map_month.values())
    
    res = pd.DataFrame(np.column_stack([yearls_m, meanls_m]), 
                               columns=['Month', 'Mean Price of '+str(types[ind])])

    resultdf_monthly.append(res) 

    
    #generate map from keys
    map_ = dict.fromkeys(data2.Year.unique())
    
    
    for yr in data2.Year.unique():
        df = data2.loc[data2['Year'] == yr]
        mean = np.asarray(df.iloc[:,1]).mean()
        map_[yr] = mean
    
    yearls = list(map_.keys())

    meanls = list(map_.values())
    
    result = pd.DataFrame(np.column_stack([yearls, meanls]), 
                               columns=['Year', 'Mean Price of '+str(types[ind])])
    
    yearDiff = []
    for index, row in result.iterrows():
        if(index == len(result)-1):
            yearDiff.append(0)
        else:
            yearDiff.append(result.iloc[index+1][1] - row[1])
    
    result['Yearly Change in Price'] = yearDiff 
    
    resultdf.append(result) 


### Method where different machine learning models are trained and comparitive study is done
def trainModels(file, imp_attr):
    #split into train and test
    train, test =  train_test_split(file, test_size = 0.2)

    #Removing the target/predictor from the train data
    targ = train[list(target_col)]
    
    ### Random Forest Model
    rand_forest_model =  RandomForestRegressor(n_estimators = 1000 , max_features = 2, oob_score = True ,  random_state = 115)
    rand_forest_model.fit(train[list(imp_attr)],targ)
    
    ### Decision Tree Model
    decision_tree_model = DecisionTreeRegressor(max_depth=4)
    decision_tree_model.fit(train[list(imp_attr)],targ)    
   
    ### Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(train[list(imp_attr)], targ)
    
    return rand_forest_model, decision_tree_model, linear_model, test

#### Method which tests the given model and Prints out the statistics regarding each of them
def  testNEvalModels(test, rf_model, dt_model, lm_model, imp_attr):
   
    print('\n Evaluation Staistics:')
    ### Evaluating Random Forest
    print("\n ***Random Forest Regressor***")
    #Evaluation metric: r square
    r2 = r2_score(test[list(target_col)] , rf_model.predict(test[list(imp_attr)]))
    print("R-Square Value:", r2)
    
    # extracting the test target values and convert to float
    true_vals = test[list(target_col)].values
    true_vals_flt = true_vals.astype(np.float)
    
    prediction = rf_model.predict(test[list(imp_attr)])
    
    #reshaping the array is required to convert it into numpy array
    aa = prediction.reshape(-1,1)
    
    mean_squared_error(true_vals_flt, prediction)
    
    mse = np.mean((true_vals_flt - aa)**2)
    print("Mean Squared Error", mse)
    
    print("Explained Variance Score", explained_variance_score(true_vals_flt, prediction))     
    print("Mean Absolute Error", mean_absolute_error(true_vals_flt, prediction))
    print("Median Absolute Error", median_absolute_error(true_vals_flt, prediction))    
    
    ### Evaluating Decision tree
    print("\n ***Decision Tree Regressor***")
    y_2 = dt_model.predict(test[list(imp_attr)])
    r2_dt = r2_score(true_vals_flt , y_2)
    print("R-Square Value:", r2_dt)
    
    print("Mean Squared Error", mean_squared_error(true_vals_flt, y_2))  
    
    print("Explained Variance Score", explained_variance_score(true_vals_flt, y_2))     
    print("Mean Absolute Error", mean_absolute_error(true_vals_flt, y_2))
    print("Median Absolute Error", median_absolute_error(true_vals_flt, y_2))    
    
    
    ### Evaluating Linear Model
    print("\n ***Linear Regression Model***")
    pred_lm = lm_model.predict(test[list(imp_attr)])

    r2_lm = r2_score(true_vals_flt, pred_lm)
    
    print("R-Square Value:", r2_lm)
    
    print("Mean Squared Error", mean_squared_error(true_vals_flt, pred_lm))  
    
    print("Explained Variance Score", explained_variance_score(true_vals_flt, pred_lm))     
    print("Mean Absolute Error", mean_absolute_error(true_vals_flt, pred_lm))
    print("Median Absolute Error", median_absolute_error(true_vals_flt, pred_lm))    


#### The chosen model is then applied to predict the future crude oil prices
def applyModel(model, test):
    #Apply selected model to test data
    pred = model.predict(test[list(imp_attr)])

    #save predicted into target column in test
    test_dtm['Predicted Price'] = pred
    
    #save df to file
    test_dtm.to_csv(test_path)


# Create a Excel Writer Object to Write Yearly analysis Price Change
writer = pd.ExcelWriter('output.xlsx')
writer_m = pd.ExcelWriter('output_m.xlsx')

for i in range(len(data)):
    analyze(data[i], i)

#Storing resultant yearly analysis
for i in range(len(resultdf)):
    resultdf[i].to_excel(writer,'data'+str(i))
writer.save()

#Storing resultant yearly analysis
for i in range(len(resultdf_monthly)):
    resultdf_monthly[i].to_excel(writer_m,'data'+str(i))
writer_m.save()


##### Code to predict the next 6 months Prices 

#Crude oil data being taken as input
train_data = data[0]
target_col = ["Cushing, OK WTI Spot Price FOB (Dollars per Barrel)"]

#features selected on which the model would be based on
imp_attr = ['Month','Year']

# Training different models and choosing the best one for prediction
rf,dt,lm,test = trainModels(train_data, imp_attr)
testNEvalModels(test, rf, dt, lm, imp_attr)

#create test data for next 6 months
test_dtm={}
test_dtm['Month']=[5,6,7,8,9,10]
test_dtm['Year']=[2017, 2017, 2017, 2017, 2017, 2017]
test_dtm = pd.DataFrame.from_dict(test_dtm)

# As per the Evaluation Statistics we get to know that Random Forest 
# is performing better than other considered models, so we apply rf to test data
applyModel(rf, test_dtm)

print("Yearly Change in Price is stored in file named: output.xlsx")
print("Monthly Change in Price is stored in file named: output_m.xlsx")
print("The resultant Predicted Values are stored in a File named: testResults.csv")
