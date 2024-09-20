# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: p.Sasinthar
RegisterNumber: 212223230199 
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

# 1.PLACEMENT DATA:

![alt text](<Screenshot 2024-09-20 102426.png>)

# 2.SALARY DATA:

![alt text](<Screenshot 2024-09-20 102531.png>)

# 3.CHECKING THE NULL() FUNCTION:

![alt text](<Screenshot 2024-09-20 102640.png>)

# 4.DATA DUPLICATE:

![alt text](<Screenshot 2024-09-20 102736.png>)

# 5.PRINT DATA:

![alt text](<Screenshot 2024-09-20 102801.png>)

# 6.DATA STATUS:

![alt text](<Screenshot 2024-09-20 102951-1.png>)

# 7.Y_PREDICATION ARRAY

![alt text](<Screenshot 2024-09-20 103052.png>)

# 8.ACCURACY VALUE:

![alt text](<Screenshot 2024-09-20 103056.png>)

# 9.CONFUSION ARRAY:

![alt text](<Screenshot 2024-09-20 103401.png>)

# 10.CLASSIFICATION REPORT:

![alt text](<Screenshot 2024-09-20 103511.png>)

# 11.PREDICTION OF LR:

![alt text](<Screenshot 2024-09-20 103620.png>)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
