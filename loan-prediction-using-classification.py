#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#to import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#to read file
df = pd.read_csv('Loan_Data.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# input x
x = df.iloc[:,1:12]
y = df.iloc[:,12:]
x.head()


# In[7]:


# now we check null value
x.isnull().sum()


# # handle int/float null values

# In[8]:


# Interpolate null values using linear interpolation
x = x.interpolate()
x.isna().sum()


# # handle obj null values

# In[9]:


x['Gender'].value_counts()


# In[10]:


x.Gender = x.Gender.fillna('Male')


# In[11]:


x['Married'].value_counts()


# In[12]:


x.Married = x.Married.fillna('Yes')


# In[13]:


x['Dependents'].value_counts()


# In[14]:


x.Dependents = x.Dependents.fillna('0')


# In[15]:


x['Self_Employed'].value_counts()


# In[16]:


x.Self_Employed = x.Self_Employed.fillna('No')


# In[17]:


x['LoanAmount'].value_counts()


# In[18]:


x.LoanAmount = x.LoanAmount.fillna(120)


# In[19]:


x.isna().sum()


# # Label Encoding

# In[20]:


#there is many string value we need to use encoder for that 
from sklearn.preprocessing import LabelEncoder 
# Initialize the LabelEncoder 
label_encoder = LabelEncoder()  
# Fit and transform the categorical column 
for i in range(0,5):   
    x.iloc[:,i] = label_encoder.fit_transform(x.iloc[:,i])  
# for last column 
x.iloc[:,10] = label_encoder.fit_transform(x.iloc[:,10])  
# for output 
y = label_encoder.fit_transform(y)


# In[21]:


x


# # Train Test split

# In[22]:


#train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.25,random_state=0)


# In[23]:


x_train


# # Scaling

# In[24]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[25]:


x_train


# # Logistic Regression

# In[26]:


#Fitting Logistic Regression to the training set  
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000,random_state=12)
lr.fit(x_train,y_train)

#Predicting the test set result  
y_pred = lr.predict(x_test)
y_pred


# In[27]:


#Creating the Confusion matrix 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acs = accuracy_score(y_test, y_pred)
cm, acs


# In[28]:


# Confusion matrix graph using seaborn
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm') 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# # Decision Tree

# In[29]:


#import decision tree 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train, y_train)

#Predicting the test set result  
y_pred_dtc = dtc.predict(x_test)
y_pred_dtc


# In[30]:


cm = confusion_matrix(y_test, y_pred_dtc)
acs = accuracy_score(y_test, y_pred_dtc)
cm, acs


# In[31]:


sns.heatmap(cm, annot=True, fmt='.2f', cmap='viridis') 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# # Random Forest

# In[32]:


# Fitting Decision Tree classifier to the training set 
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
rfc.fit(x_train, y_train) 

y_pred_rfc = rfc.predict(x_test)
y_pred_rfc


# In[33]:


cm = confusion_matrix(y_test, y_pred_rfc)
acs = accuracy_score(y_test, y_pred_rfc)
cm, acs


# In[34]:


sns.heatmap(cm, annot=True, fmt='.2f', cmap='cividis') 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# # Naive Bayes

# In[35]:


# Fitting Naive Bayes to the Training set  
from sklearn.naive_bayes import GaussianNB  
nb = GaussianNB()  
nb.fit(x_train, y_train)

y_pred_nb = nb.predict(x_test)
y_pred_nb


# In[36]:


cm = confusion_matrix(y_test, y_pred_nb)
acs = accuracy_score(y_test, y_pred_nb)
cm, acs


# In[37]:


sns.heatmap(cm, annot=True, fmt='.2f', cmap='plasma') 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# # K-Nearest Neighbor(KNN)

# In[38]:


#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
knn.fit(x_train, y_train) 

y_pred_knn = knn.predict(x_test)
y_pred_knn


# In[39]:


cm = confusion_matrix(y_test, y_pred_knn)
acs = accuracy_score(y_test, y_pred_knn)
cm, acs


# In[40]:


sns.heatmap(cm, annot=True, fmt='.2f', cmap='inferno') 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# # Support Vector Machine(svm)

# In[41]:


from sklearn.svm import SVC # "Support vector classifier"  
svc = SVC(kernel='linear', random_state=0)  
svc.fit(x_train, y_train)  

y_pred_svc = svc.predict(x_test)
y_pred_svc


# In[42]:


cm = confusion_matrix(y_test, y_pred_svc)
acs = accuracy_score(y_test, y_pred_svc)
cm, acs


# In[43]:


sns.heatmap(cm, annot=True, fmt='.2f', cmap='magma') 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# #### Logistic Regression is best for this

# In[ ]:




