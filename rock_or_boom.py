import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import column_or_1d
#loading the dataset to pandas Dataframe
sonar_data=pd.read_csv('Copy of sonar data.csv',header=None)
print(sonar_data.head())
print(sonar_data.describe())
print(sonar_data[60].value_counts())
# R=rock M=mine 
#separating data and label
x=sonar_data.drop(columns=60,axis=1)
y=sonar_data[60]
print(x)
print(y)
#training and testing data
x_train , x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)
#model training -> logisticRegression
model=LogisticRegression()
#training the logistic regression model with training data
print(x_train)
print(y_train)
model.fit(x_train,y_train)
#model evaluation
    #accuracy on training data
x_train_predict=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_predict,y_train)
print('accuracy on training data:',training_data_accuracy)
# making a prediction system
input_data = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)
# changing the input_data to a numpy array
input_data_asnumpy_array=np.asarray(input_data)
#reshape the np array as we are predicting for one instance 
input_data_reshaped=input_data_asnumpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if prediction[0]=='R':
 print('the object is Rock')
else:
    print('the object is Mine')
