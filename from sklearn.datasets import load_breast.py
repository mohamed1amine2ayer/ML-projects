from types import DynamicClassAttribute
from sklearn import feature_selection
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2 ,f_classif
# laod breast cancer data
breastdata=load_breast_cancer()
# x data
x=breastdata.data
print('x data is \n',x[:10])
print('x shape is :\n',x.shape)
print('x features are \n',breastdata.feature_names)
# y data 
y=breastdata.target
print('y data is \n',y[:10])
print('y shape is : \n',y.shape)
print('y columns are :\n',breastdata.target_names)
# features selection by percentile
featureSelection=SelectPercentile(score_func=chi2,percentile=20)
x=featureSelection.fit_transform(x,y)
print(x.shape)

