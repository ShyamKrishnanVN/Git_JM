#!/usr/bin/env python
# coding: utf-8

# In[ ]:


    from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
    nb = NotebookExecutor()
    df_order_sales_data = nb.get_data('27911694761892217', '@SYS.USERID', 'True', {}, ['27911694761942588'])
    df=df_order_sales_data
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
 
    def get_sentiment(text):
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        if sentiment_score > 0:
            return 'Positive'
        elif sentiment_score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply the sentiment analysis function to create a "Sentiment" column
df


# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
df_order_sales_data = nb.get_data('27911694761892217', '@SYS.USERID', 'True', {}, ['27911694761942588'])
df=df_order_sales_data

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
df_order_sales_data = nb.get_data('27911694761892217', '@SYS.USERID', 'True', {}, ['27911694761942588'])
df=df_order_sales_data

import pandas as pd
from time import perf_counter as get_time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

_target = 'Order_Item'    # string: Column name of the target variable
print(_target)

numeric_columns = df.select_dtypes(include=['number'])

# Select specific categorical columns by specifying their names
categorical_columns = df[['Order_Item','App','Branch_Id','Day_Name','Feedback','Restaurant_Name']]  # Replace with your actual column names

# Perform one-hot encoding on categorical columns
categorical_encoded = pd.get_dummies(categorical_columns, drop_first=True)

    # Combine numeric and one-hot encoded categorical DataFrames
_data = pd.concat([numeric_columns, categorical_encoded], axis=1)
print(_data)

# Define the necessary variables here
#_data   = df['Order_Item','App', 'Branch_Id','Day_Name', 'Feedback','Price', 'Quantity', 'Rating', 'Restaurant_Name', 'Tax', 'Total_Amount']    # pd.DataFrame: Full data to process
#_target ='Order_Item'  # string: Column name of the target variable

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
ran_for_clf = RandomForestClassifier()
time_now = get_time()
ran_for_clf.fit(X_train, y_train);
print(f'Model {ran_for_clf} trained')
print(f'Seconds elapsed: {round(get_time() - time_now, 3)}')

# Making predictions on the training data
predict_train = ran_for_clf.predict(X_train)
print(f'Predictions on training data made')

# Finding the accuracy score of the training predictions
accuracy_train = accuracy_score(y_train, predict_train)
print(f'Accuracy score of training predictions: {round(accuracy_train, 3)}')

# Printing the classification report of the training predictions
report_train = classification_report(y_train, predict_train, digits=3)
print(f'Classification report of training predictions:')
print(report_train)

# Making predictions on the testing data
predict_test = ran_for_clf.predict(X_test)
print(f'Predictions on testing data made')

# Finding the accuracy score of the testing predictions
accuracy_test = accuracy_score(y_test, predict_test)
print(f'Accuracy score of testing predictions: {round(accuracy_test, 3)}')

# Printing the classification report of the testing predictions
report_test = classification_report(y_test, predict_test, digits=3)
print(f'Classification report of testing predictions:')
print(report_test)

# In[ ]:


from Notebook.DSNotebook.Note
bookExecutor import NotebookExecutor
nb = NotebookExecutor()
df_order_sales_data = nb.get_data('27911694761892217', '@SYS.USERID', 'True', {}, ['27911694761942588'])
df_order_sales_data

# In[ ]:


df_order_sales_data.columns

# In[ ]:



import pandas as pd
df = df_order_sales_data[['App','Branch_Id','Commission','Day_Name','Holiday','Order_Item','Order_Type','Payment_Mode','Price','Quantity','Restaurant_Name','Rating','Tax','Total_Amount']]

# In[ ]:



one_hot_encoded_data = pd.get_dummies(df, columns = ['App','Branch_Id','Commission','Day_Name','Holiday','Order_Type','Payment_Mode','Price','Quantity','Restaurant_Name','Rating','Tax','Total_Amount'],drop_first=True)
one_hot_encoded_data

# In[ ]:


import pandas as pd
from time import perf_counter as get_time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the necessary variables here
_data   = one_hot_encoded_data    # pd.DataFrame: Full data to process
_target = 'Order_Item'    # string: Column name of the target variable

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
ran_for_clf = RandomForestClassifier()
time_now = get_time()
ran_for_clf.fit(X_train, y_train);
print(f'Model {ran_for_clf} trained')
print(f'Seconds elapsed: {round(get_time() - time_now, 3)}')

# Making predictions on the training data
predict_train = ran_for_clf.predict(X_train)
print(f'Predictions on training data made')

# Finding the accuracy score of the training predictions
accuracy_train = accuracy_score(y_train, predict_train)
print(f'Accuracy score of training predictions: {round(accuracy_train, 3)}')

# Printing the classification report of the training predictions
report_train = classification_report(y_train, predict_train, digits=3)
print(f'Classification report of training predictions:')
print(report_train)

# Making predictions on the testing data
predict_test = ran_for_clf.predict(X_test)
print(f'Predictions on testing data made')

# Finding the accuracy score of the testing predictions
accuracy_test = accuracy_score(y_test, predict_test)
print(f'Accuracy score of testing predictions: {round(accuracy_test, 3)}')

# Printing the classification report of the testing predictions
report_test = classification_report(y_test, predict_test, digits=3)
print(f'Classification report of testing predictions:')
print(report_test)

# In[ ]:


rf_classifier

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
saved_model = nb.save_model(model = ran_for_clf, modelName = 'workflow6', modelType = 'ml', X = None, y = None, estimator_type='')
#X and y are training datasets to get explainer dashboard.
#estimator_type is to specify algorithm type i.e., classification and regression.
#Only 'ml’ models with tabular data as input will support in Explainer Dashboard.
#Choose modelType = 'ml' for machine learning models, modelType = 'cv' for computer vision models and modelType = 'dp' for data transformation pickle files. 
#Provide ‘column_headers’ as a parameter if they have to be saved in the model.
#If using custom layer in keras, use native save functionality from keras.

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
loaded_model = nb.load_saved_model('27911694782782475')


# In[ ]:


df[0:1]

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
saved_model = nb.save_model(model = rf_classifier, modelName = 'order', modelType = 'ml', X = None, y = None, estimator_type='classification')
#X and y are training datasets to get explainer dashboard.
#estimator_type is to specify algorithm type i.e., classification and regression.
#Only 'ml’ models with tabular data as input will support in Explainer Dashboard.
#Choose modelType = 'ml' for machine learning models, modelType = 'cv' for computer vision models and modelType = 'dp' for data transformation pickle files. 
#Provide ‘column_headers’ as a parameter if they have to be saved in the model.
#If using custom layer in keras, use native save functionality from keras.

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
saved_model = nb.save_model(model = model, modelName = 'model', modelType = 'ml', X = None, y = None, estimator_type='classification')
#X and y are training datasets to get explainer dashboard.
#estimator_type is to specify algorithm type i.e., classification and regression.
#Only 'ml’ models with tabular data as input will support in Explainer Dashboard.
#Choose modelType = 'ml' for machine learning models, modelType = 'cv' for computer vision models and modelType = 'dp' for data transformation pickle files. 
#Provide ‘column_headers’ as a parameter if they have to be saved in the model.
#If using custom layer in keras, use native save functionality from keras.
