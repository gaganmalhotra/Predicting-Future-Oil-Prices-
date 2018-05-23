# Data analysis for Spot Price for Crude Oil and Petroleum Products
Built a model using python and comparitive study of different models and their resulting performance

### Insights from the Initial Analysis

- The crude oil prices per barrel significantly increased during 2004 to
2009 followed by a drop in prices again in 2015 matching lowest in last 10
years
- Monthly trends show that Crude Oil price soar highest during the months of
April - August with May being the most expensive
- The price of Diesel fuel increases as it is taken away from US Gulf coast
area(Texas region) toward New York
- The price change in the Heating oil & Propane remains less significant as
compared to other petroleum products
- The spot prices of Blended Crude Stream (Brent) is always on the lower
side of the benchmarked price of oil in US

### Prediction Models and Evaluation Criteria
 
| Models        | R-Square| Mean Squared Error  | Explained Variance  | Mean Absolute Error |Median Absolute Error |
| ------------- |:-------------:| -----:| -----:|-----:| -----:| 
| ***Random Forest Regressor***| 0.981409534225 | 18.0940824985 | 0.981454612847  |2.73066328947 | 1.70786 |
|  ***Decision Tree Regressor***     |   0.918915639339    |  78.9193304196 |  0.919298696549 | 5.13448524613 | 2.8114516129 |
| ***Linear Regression*** | 0.645523302658      |  345.011829352 | 0.647685690745 | 15.215983657 | 12.1013037738 |


Based on the stats above **Random Forest** model was chosen!

### Few Crude Oil Data Trends

![ScreenShot](https://github.com/gaganmalhotra/Predicting-Future-Oil-Prices-/blob/master/CrudeOilYearlyDist.png)
![ScreenShot](https://github.com/gaganmalhotra/Predicting-Future-Oil-Prices-/blob/master/CrudeOilMonthDist.png)
