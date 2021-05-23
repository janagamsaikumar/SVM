import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\AMXWAM_ TASK\TASK-27\Social_Network_Ads.csv')
X=dataset.iloc[:,2:-1]
y=dataset.iloc[:,-1].values
dataset.isnull().sum() # there are no missing values in the dataset 

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

# train and test splitting the data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# support vector classifier
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

#predicting our xtest to check accuracy
y_pred=classifier.predict(X_test)

# performance measures
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print('your accuracy score using svc is')
print(cm)

# checking accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

# getting report
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)

#Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#----------------------------------------------------------------------

X_pred=classifier.predict(X_train)
cm_train=confusion_matrix(y_train,X_pred)
acc_train=accuracy_score(y_train,X_pred)
cr_train=classification_report(y_train,X_pred)


# visualizing training dataset
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#====================================================================




