# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Start the program
2. Import the required python libraries such as numpy,pandas,matplotlib
3. Read the dataset of student scores
4. Assign the column hours to x and column scores to y
5. From sklearn library select the model to train and to test the dataset
6. Plot the training set and testing set in the graph using matplotlib library
7. Stop the program
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sriram V
RegisterNumber: 212222103002
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
![Screenshot 2024-02-28 201117](https://github.com/Darkwebnew/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143114486/7700ce91-22fd-41ad-9cd2-c9f918c7b37d)
![Screenshot 2024-02-28 201130](https://github.com/Darkwebnew/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143114486/23cb4c66-a6f0-46ce-a770-026955c4adc6)
![Screenshot 2024-02-28 201138](https://github.com/Darkwebnew/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143114486/e6b35381-eeb6-431d-8208-b9d1717f1e00)
![Screenshot 2024-02-28 201210](https://github.com/Darkwebnew/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143114486/6a4c3e83-97b8-4bf3-acb6-15d65b2889d8)
![Screenshot 2024-02-28 201219](https://github.com/Darkwebnew/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/143114486/99a3b891-28f8-48b7-9730-1749e8c10151)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
