#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
df_Abalone_Prep_Test = nb.get_data('11121674188059683', '@SYS.USERID', 'True',{},[])
df_Abalone_Prep_Test

# In[ ]:


import pandas as pd
from time import perf_counter as get_time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Define the necessary variables here
_data   = df_Abalone_Prep_Test    # pd.DataFrame: Full data to process
_target = 'sex'    # string: Column name of the target variable

if _data is None or _target is None:
    raise Exception(f'Both _data and _target must be specified')
elif not (isinstance(_data, pd.DataFrame) and isinstance(_target, str)):
    raise Exception(f'Datatype of _data must be pd.DataFrame; that of _target must be str')

# Separating the independent and dependent variables into X and y respectively
y = _data[_target]
X = _data.drop(columns=_target)
print(f'Shape of complete data: {_data.shape}')

# Splitting the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(f'Shape of training data: {X_train.shape}')
print(f'Shape of testing data : {X_test.shape}')

# Creating the classifier and fitting it to the training data
svc = SVC()
time_now = get_time()
svc.fit(X_train, y_train);
print(f'Model {svc} trained')
print(f'Seconds elapsed: {round(get_time() - time_now, 3)}')

# Making predictions on the training data
predict_train = svc.predict(X_train)
print(f'Predictions on training data made')

# Finding the accuracy score of the training predictions
accuracy_train = accuracy_score(y_train, predict_train)
print(f'Accuracy score of training predictions: {round(accuracy_train, 3)}')

# Printing the classification report of the training predictions
report_train = classification_report(y_train, predict_train, digits=3)
print(f'Classification report of training predictions:')
print(report_train)

# Making predictions on the testing data
predict_test = svc.predict(X_test)
print(f'Predictions on testing data made')

# Finding the accuracy score of the testing predictions
accuracy_test = accuracy_score(y_test, predict_test)
print(f'Accuracy score of testing predictions: {round(accuracy_test, 3)}')

# Printing the classification report of the testing predictions
report_test = classification_report(y_test, predict_test, digits=3)
print(f'Classification report of testing predictions:')
print(report_test)

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
saved_model = nb.save_model(model = svc, modelName = 'SVC_V1', modelType = 'ml', X = X_train, y = y_train, estimator_type='classification')
#X and y are training datasets to get explainer dashboard.
#estimator_type is to specify algorithm type i.e., classification and regression.
#Only 'ml’ models with tabular data as input will support in Explainer Dashboard.
#Choose modelType = 'ml' for machine learning models, modelType = 'cv' for computer vision models and modelType = 'dp' for data transformation pickle files. 
#Provide ‘column_headers’ as a parameter if they have to be saved in the model.
#If using custom layer in keras, use native save functionality from keras.

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
loaded_model = nb.load_saved_model('11121677653771238')

# In[ ]:


nb.predict(model = loaded_model, dataframe = X_test, modeltype='ml') 
 #Choose modeltype 'ml' for machine learning models and 'cv' for computer vision model 
 #ex: For machine learning model nb.predict(model = model, modeltype = 'ml', dataframe = df) 
 #ex: For computer vision keras model nb.predict(model = model, modeltype = 'cv', imgs = imgs, imgsize = (28, 28), dim = 1, class_names = class_names) 
 #and for pytorch model(model = model, modeltype = 'cv', imgs = imgs, class_names = class_names) 
 #Note: incase any error in prediction user squeezed image data in keras
