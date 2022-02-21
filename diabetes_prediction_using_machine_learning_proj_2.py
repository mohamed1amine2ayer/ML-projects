import numpy as np
from numpy.random.mtrand import standard_cauchy
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes_dataset=pd.read_csv('diabetes.csv')
print(diabetes_dataset.head())
print(diabetes_dataset.describe())
print(diabetes_dataset.info())
print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())
# separating data  and labels 
x=diabetes_dataset.drop(columns='Outcome',axis=1)
y=diabetes_dataset['Outcome']
print(x)
print(y) 
scaler=StandardScaler()
scaler.fit(x)
standardized_data=scaler.transform(x)
print(standardized_data)
x=standardized_data
y=diabetes_dataset['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

# training the model
classifier=svm.SVC(kernel='linear')
#training svm 
classifier.fit(x_train,y_train)
# model evaluation
x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score of the training data:',training_data_accuracy)
x_train_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_train_prediction,y_test)
print('Accuracy score of the training data:',test_data_accuracy)
input_data=()
#changing into numpy array
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')