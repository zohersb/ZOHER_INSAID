
<p align="center"><img src="https://github.com/insaid2018/Term-1/blob/master/Images/INSAID_Full%20Logo.png?raw=true" width="260" height="110" /></p>

---
# **Table of Contents**
---

1. [**Introduction**](#Section1)<br>
2. [**Problem Statement**](#Section2)<br>
3. [**Installing & Importing Libraries**](#Section3)<br>
  3.1 [**Installing Libraries**](#Section31)<br>
  3.2 [**Upgrading Libraries**](#Section32)<br>
  3.3 [**Importing Libraries**](#Section33)<br>
4. [**Data Acquisition & Description**](#Section4)<br>
5. [**Data Pre-Profiling**](#Section5)<br>
6. [**Data Pre-Processing**](#Section6)<br>
7. [**Data Post-Profiling**](#Section7)<br>
8. [**Exploratory Data Analysis**](#Section8)<br>
9. [**Summarization**](#Section9)</br>
  9.1 [**Conclusion**](#Section91)</br>
  9.2 [**Actionable Insights**](#Section91)</br>

---

---
<a name = Section1></a>
# **1. Introduction**
---

 - XYZ Health Insurance Co. Ltd. is one of the prominent insurance providers in the country.
 - They offer impressive health plans and services to cater to the needs of different people.
 - The insurance company also provides access to fitness assessment centers, wellness centers, diagnostic centers in addition to hospitalization centers.

Current Scenario
The company is planning to introduce a new system that will help to easily monitor and predict the medical insurance prices of their customers.



---
<a name = Section2></a>
# **2. Problem Statement**
---

- This section is emphasised on providing some generic introduction to the problem that most companies confronts.

- **Problem Statement:**

 - The company uses manpower to predict the medical expenses of its insurers. Many factors are considered such as age, BMI, smoking habits, number of children, etc.
 - It is a time and resource-intensive process and many times, inaccurate.
 - The company plans to modernize its legacy systems and wants to implement an automated way of predicting the medical expenses of its insurers based on various factors.
 
They have hired you as a data science consultant. They want to supplement their analysis and prediction with a more robust and accurate approach.

Your Role

 - You are given a historical dataset that contains the medical charges of some of the insurers and many factors that determine those charges.
 - Your task is to build a regression model using the dataset.
 - Because there was no machine learning model for this problem in the company, you don’t have a quantifiable win condition. You need to build the best possible model.



---
<a id = Section3></a>
# **3. Installing & Importing Libraries**
---

- This section is emphasised on installing and importing the necessary libraries that will be required.

### **Installing Libraries**

#!pip install -q datascience                                         # Package that is required by pandas profiling
#!pip install -q pandas-profiling                                    # Library to generate basic statistics about data
# To install more libraries insert your code here..

### **Upgrading Libraries**

- **After upgrading** the libraries, you need to **restart the runtime** to make the libraries in sync.

- Make sure not to execute the cell under Installing Libraries and Upgrading Libraries again after restarting the runtime.

### **Importing Libraries**

- You can headstart with the basic libraries as imported inside the cell below.

- If you want to import some additional libraries, feel free to do so.


#-------------------------------------------------------------------------------------------------------------------------------
import pandas as pd                                                 # Importing package pandas (For Panel Data Analysis)
from pandas_profiling import ProfileReport                          # Import Pandas Profiling (To generate Univariate Analysis)
#-------------------------------------------------------------------------------------------------------------------------------
import numpy as np                                                  # Importing package numpys (For Numerical Python)
#-------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt                                     # Importing pyplot interface to use matplotlib
import seaborn as sns                                               # Importing seaborn library for interactive visualization
%matplotlib inline
#-------------------------------------------------------------------------------------------------------------------------------
import scipy as sp                                                  # Importing library for scientific calculations
#-------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split                # To split the data in training and testing part     
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression                 # To create the Logistic Regression Model
from sklearn import metrics
#-------------------------------------------------------------------------------------------------------------------------------
import warnings                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore") 

---
<a name = Section4></a>
# **4. Data Acquisition & Description**
---

- This section is emphasised on the accquiring the data and obtain some descriptive information out of it.

- You could either scrap the data and then continue, or use a direct source of link (generally preferred in most cases).

- You will be working with a direct source of link to head start your work without worrying about anything.

- Before going further you must have a good idea about the features of the data set:

|Id|Feature|Description|
|:--|:--|:--|
|01|age| The age of the use.| 
|02|sex| Determine the gender of the user.| 
|03|bmi| Determine the bmi of the user.| 
|04|children| Define the number of children the user have.|
|05|smoker| Whether the user is smoker or not.|
|06|region| The region where the user belongs to.|
|07|id| The unique id of the user.|


#Loading CSV file

train_data = pd.read_csv('C:/Users/zoher/Desktop/Term 4/Project/Medical insuarance/Medical-Cost-Prediction/train_data.csv')
train_data.shape

train_data

### **Data Description**

- To get some quick description out of the data you can use describe method defined in pandas library.

train_data.describe()                                       

### **Data Information**

train_data.info()                                          

train_data.isnull().sum()

There are no missing values in the dataset.

---
<a name = Section5></a>
# **5. Data Pre-Profiling**
---

- This section is emphasised on getting a report about the data.

- You need to perform pandas profiling and get some observations out of it...

# profile = ProfileReport(df = data)
# profile.to_file(outputfile = 'Pre Profiling Report.html')
# print('Accomplished!')



# Insert your code here...



---
<a name = Section8></a>
# **6. Exploratory Data Analysis**
---

- This section is emphasised on asking the right questions and perform analysis using the data.

- Note that there is no limit how deep you can go, but make sure not to get distracted from right track.

#Understanding the entire train data using a pair plot
sns.pairplot(train_data)    

Check the relation between Age and Charges

sns.regplot(x='age', y='charges', data=train_data)
plt.title('Age vs Charges')
plt.show()

sns.regplot(x='bmi', y='charges', data=train_data)
plt.title('BMI vs Charges')
plt.show()

Corelation between different atttibute of data

plt.figure(figsize=(12,8))
sns.heatmap(train_data.corr(),annot=True)
plt.show()

Distribution of changes for sexwise

plt.figure(figsize=(12,5))
plt.title("Box plot for charges of women")
sns.boxplot(y='smoker',x='charges',data=train_data[(train_data.sex == 'female')] , orient='h',palette='magma')

plt.figure(figsize=(12,5))
plt.title("Box plot for charges of men")
sns.boxplot(y='smoker',x='charges',data=train_data[(train_data.sex == 'male')] , orient='h',palette='rainbow')

# Distribution of changes against age
sns.set()
plt.figure(figsize=(15,5))
plt.title("Distribution of age")
sns.distplot(train_data['age'], color = 'g')
plt.show()

# Distribution of changes against sex
plt.figure(figsize = (8,8))
sns.countplot(x = 'sex', data = train_data)
plt.title("Distribution of Sex")
plt.show()

train_data["sex"].value_counts()

# Distribution of changes against BMI
plt.figure(figsize=(15,8))
plt.title("Distribution of BMI")
sns.distplot(train_data['bmi'], color = 'r')
plt.show()

# Distribution of charges for patients with BMI greater than 30
plt.figure(figsize=(12,5))
plt.title("Distribution of charges for patients with BMI greater than 30")
ax = sns.histplot(train_data[(train_data.bmi >= 30)]['charges'], color = 'm', kde=True,linewidth=0)

# Distribution of charges for patients with BMI less than 30
plt.figure(figsize=(12,5))
plt.title("Distribution of charges for patients with BMI less than 30")
ax = sns.histplot(train_data[(train_data.bmi<30)]['charges'], color='b', kde=True,linewidth=0)

## Distribution of charges for patients against age
plt.figure(figsize = (15,8))
plt.bar(x = train_data['age'], height = train_data['charges'])
plt.show()

# Distribution of changes against Children
sns.catplot(x="children", kind="count", palette="ch:.25",data=train_data, height=6)


train_data['children'].value_counts()

# Distribution of changes against Children
plt.figure(figsize=(8,8))
sns.catplot(x='smoker', kind='count', palette='pink', data=train_data)

train_data["smoker"].value_counts()

f = plt.figure(figsize=(12,5))

ax = f.add_subplot(121)
sns.histplot(train_data[(train_data.smoker=='yes')]['charges'], color='c',ax=ax,kde=True, linewidth=0)
ax.set_title('Distribution of charges for smokeres')

ax = f.add_subplot(122)
sns.histplot(train_data[(train_data.smoker=='no')]['charges'],color='b',ax=ax,kde=True,linewidth=0)
ax.set_title('Distribution of charges for non-smokers')

# Distribution of changes against Region
plt.figure(figsize=(8,8))
sns.catplot(x='region', kind='count', palette='pink', data=train_data)

train_data["region"].value_counts()

plt.figure(figsize = (10,7))
sns.distplot(train_data['charges'], color = 'g')
plt.title("Distribution of charges")
plt.show()

# Replacing the values of Sex into numerical value Male = 0 and Female = 1

train_data.replace({"sex":{"male":0, "female":1}}, inplace = True)

# Replacing the values of smoker into numerical value smoker = 0 and non smoker = 1

train_data.replace({"smoker":{"yes":0, "no":1}}, inplace = True)

# Replacing the values of region into numerical value 

train_data.replace({"region":{"southeast":0, "southwest":1, "northeast":2, "northwest":3}}, inplace = True)

train_data

#For testing dataset


test_data.replace({"sex":{"male":0, "female":1}}, inplace = True)

# Replacing the values of smoker into numerical value smoker = 0 and non smoker = 1

test_data.replace({"smoker":{"yes":0, "no":1}}, inplace = True)

# Replacing the values of region into numerical value 

test_data.replace({"region":{"southeast":0, "southwest":1, "northeast":2, "northwest":3}}, inplace = True)

#Splitting the train data set into training and testing.

X = train_data.drop(columns = "charges", axis = 1)
y = train_data["charges"]

X

y

 - **Splitting 80/20 ratio**

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#Using Linear Regression
linreg = LinearRegression()


linreg.fit(x_train, y_train) 

# Prediction on training data
train_data_predict = linreg.predict(X_train)

#Finding the R2
r2_train = metrics.r2_score(y_train,train_data_predict)
print("R Squared Value:", r2_train)

#Prediction on testing data
test_data_predict = linreg.predict(X_test)

#Finding the R2
r2_test = metrics.r2_score(y_test,test_data_predict)
print("R Squared Value:", r2_test)

#Building a suitable system
input_data = (45,1,25.175,2,1,2,764)
input_data_numpy = np.asarray(input_data)
input_data_reshaped = input_data_numpy.reshape(1,-1)
prediction = linreg.predict(input_data_reshaped)
prediction

prediction = linreg.predict(test_data)

print(prediction)

res = pd.DataFrame(prediction)
res.index = test_data.index
res.index = test_data['id']
res.columns = ["charges"]
res.to_csv("medical_cost_prediction_results_zoherbehrainwala@gmail.com.csv", index = False, header = False)

test_data = pd.read_csv('C:/Users/zoher/Desktop/Term 4/Project/Medical insuarance/Medical-Cost-Prediction/test_data.csv')
test_data

test_data = pd.read_csv('C:/Users/zoher/Desktop/Term 4/Project/Medical insuarance/Medical-Cost-Prediction/test_data.csv')
test_data.shape

---
<a name = Section9></a>
# **9. Summarization**
---

<a name = Section91></a>
### **9.1 Conclusion**

 - We studied in breifly about the data, its characteristics and its distribution.

 - We investigated about the charges of for the users.
 
 - The prediction is done based on R square value as required.

 - Since the data here is continous the best fit model is Linear Regression model for this dataset.

