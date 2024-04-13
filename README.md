# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean squared error, and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Thirunavukkarasu p
RegisterNumber:  212222040173
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print(mse)

r2=metrics.r2_score(y_test,y_pred)
print(r2)

dt.predict([[5,6]])

```

## Output:
![1](https://github.com/Thirunavukkarasu05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119291645/59508f0e-3ad5-4715-a9f2-a0369ac2ffbf)
![2](https://github.com/Thirunavukkarasu05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119291645/1e16b8eb-f574-43f7-a983-0b02aa2d639e)
![3](https://github.com/Thirunavukkarasu05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119291645/5ffa4743-9b41-449a-988a-5af74f4f8425)
![4](https://github.com/Thirunavukkarasu05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119291645/647c72dd-dd81-4f84-96ee-2682b93aca30)
![5](https://github.com/Thirunavukkarasu05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119291645/32916e44-0bb7-4a62-8daf-4e63d1c06e61)
![6](https://github.com/Thirunavukkarasu05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119291645/ac2e681a-d0c0-4b77-a64f-848869395bf2)
![7](https://github.com/Thirunavukkarasu05/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119291645/fbd1f1f9-04af-4054-bd7e-72f364bd1881)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
