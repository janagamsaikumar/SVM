import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\class 20 _oct 31,2020\Position_Salaries.csv')
X=dataset.iloc[:,1:-1]
y=dataset.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X)
y=StandardScaler().fit_transform(y.values.reshape(-1,1))

from sklearn.svm import SVR # support vector regression  it support the hiper plane
reg=SVR(kernel='rbf') # radial basis fucntion used for svm classification
reg.fit(X,y)

y_pred=reg.predict([[6.5]])
plt.scatter(X,y,color='red')
plt.plot(X,reg.predict(X),color='blue')
plt.title('true or false')
plt.xlabel('position_level')
plt.ylabel('salary')
plt.show()
