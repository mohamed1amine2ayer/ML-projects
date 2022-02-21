import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics



# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv('gld_price_data.csv')
# print first 5 rows in the dataframe
print(gold_data.head())
# number of rows and columns
print(gold_data.shape)
print(gold_data.info())
# checking the number of missing values
print(gold_data.isnull().sum())
# checking the number of missing values
print(gold_data.isnull().sum())
correlation = gold_data.corr()
# constructing a heatmap to understand the correlatiom
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='BrBG_r')
# correlation values of GLD
print(correlation['GLD'])
# checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'],color='green')
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
#Model Training: Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)
# training the model
regressor.fit(X_train,Y_train)
# prediction on Test Data
test_data_prediction = regressor.predict(X_test)
# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
#Compare the Actual Values and Predicted Values in a Plot
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()