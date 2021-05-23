import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\class 20 _oct 31,2020\Position_Salaries.csv')
# this dataset contains positions and salaries 
# we have to find out when a 6.5 experience guy salary when joins a organisation.
# comapre the accuracy and come to conclusion.

# check for missing values 
dataset.isnull().any()

# seperate the dataset into dependent and independent variables
X=dataset.iloc[:,1:-1]
y=dataset.iloc[:,-1]

from sklearn.svm import SVR # support vector regression  it support the predicted line to know  the accuracy 
reg=SVR(kernel='rbf') # radial basis fucntion used for svm classification
reg.fit(X,y)
y_pred=reg.predict([[6.5]]) # we got 130k which is less to expected salary. because if u observe for level 1 to level 10, salary which is present rhe dataset is not matching with the level that our machine doesnt not understand we have to scale it to 0 to 1
plt.scatter(X,y,color='red')
plt.plot(X,reg.predict(X),color='blue')
plt.title('true or false')
plt.xlabel('position_level')
plt.ylabel('salary')
# here if u observe there is not datapoint touching the predicted line.
# it is linear regression
# if u observe the data points are non linearly plotted and u have to fit the bestfit line to get the proper prediction
# we require feature scaling here to bring it to between 0 to 1
 from sklearn.ensemble import RandomForestRegressor
 reg=RandomForestRegressor(n_estimators=10,random_state=0)
 reg.fit(X,y)
 y_pred=reg.predict([[6.5]])
 