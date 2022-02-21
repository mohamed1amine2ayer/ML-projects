import numpy as np
from numpy.core.numeric import correlate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics



house_price_dataset=sklearn.datasets.load_boston()    # importing dataset
house_price_dataset_dataframe=pd.DataFrame(house_price_dataset,columns=house_price_dataset.feature_names)
# add the target column to the data frame
house_price_dataset_dataframe['price']=house_price_dataset.target # add the target column to the dataframe
print(house_price_dataset_dataframe.head())
# checking for missing values
print(house_price_dataset_dataframe.info())
#statistical measures of the dataset
print(house_price_dataset_dataframe.describe())
# understanding the correlation between various in dataset
correlation=house_price_dataset_dataframe.corr()
# constructing heatmap to understand correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()
#spliting the data and the target
x=house_price_dataset_dataframe.drop(['price'],axis=1)
y=house_price_dataset_dataframe['price']
print(x)
print(y)
x_train, x_test, y_train, y_test(x,y,test_size=0.2, random=2)
#model training
#xgboost regressor
model=XGBRegressor()
model.fit(x_train,y_train)
# evaluation
   # prediction on training data
      # accuracy for prediction on training data
training_data_prediction=model.predict(x_train)
      # r squared error
score_1=metrics.r2_score(y_train ,training_data_prediction)
     # mean absolute error
score_2=metrics.mean_absolute_error(y_train,training_data_prediction)
print('r square error : ',score_1)
print('mean absolute error  : ',score_2)
# Visualizing the actual Prices and predicted prices 
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()
#RQ the closer the points are the more our model is better(prediction)