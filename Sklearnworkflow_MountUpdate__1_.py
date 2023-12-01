#!/usr/bin/env python
# coding: utf-8
# as
# # Model fitting & training

# In[ ]:


#sklearn train model
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = MultinomialNB()
model.fit(X_train, Y_train);

# # Model Save

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
saved_model = nb.save_model(model = model, modelName = 'Model_newTenant_V2', modelType = 'ml', X = X_train, y = Y_train, estimator_type='classification')
#X and y are training datasets to get explainer dashboard.
#estimator_type is to specify algorithm type i.e., classification and regression.
#Only 'ml’ models with tabular data as input will support in Explainer Dashboard.
#Choose modelType = 'ml' for machine learning models, modelType = 'cv' for computer vision models and modelType = 'dp' for data transformation pickle files. 
#Provide ‘column_headers’ as a parameter if they have to be saved in the model.
#If using custom layer in keras, use native save functionality from keras.

# # Model Load

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
loaded_model = nb.load_saved_model('17171692773469511')

# In[ ]:


X_test_copy = X_test.copy()

# # Model Predict

# In[ ]:


nb.predict(model = loaded_model, dataframe = X_test_copy, modeltype='ml') 
 #Choose modeltype 'ml' for machine learning models and 'cv' for computer vision model 
 #ex: For machine learning model nb.predict(model = model, modeltype = 'ml', dataframe = df) 
 #ex: For computer vision keras model nb.predict(model = model, modeltype = 'cv', imgs = imgs, imgsize = (28, 28), dim = 1, class_names = class_names) 
 #and for pytorch model(model = model, modeltype = 'cv', imgs = imgs, class_names = class_names) 
 #Note: incase any error in prediction user squeezed image data in keras

# In[ ]:


X_test_copy = X_test.copy()

# In[ ]:


Y_pred = nb.predict(model = loaded_model, dataframe = X_test_copy, modeltype='ml') 

# In[ ]:


Y_pred.head()

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred.predictions)

# # Sandbox file read

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
df_Test_Data = nb.get_data('17171692681942349', '@SYS.USERID', 'True', {}, [])
df_Test_Data

# # Artifacts file save

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
#File extension should be with .csv/.json/.txt
nb.save_artifact(dataframe = df_Test_Data, name = 'df_Test_Data.txt')

# # Artifacts saved file read

# In[ ]:


@SYS.ARTIFACT_PATH+'df_Test_Data.txt'

# In[ ]:


print(open(@SYS.ARTIFACT_PATH+'df_Test_Data.txt').read())

# # Reading uploaded file in forder structure

# In[ ]:


@SYS.DATASANDBOX_PATH + '1231421441/Data/Folder_V1/churn_data_new.csv'

# In[ ]:


pd.read_csv(@SYS.DATASANDBOX_PATH + '1231421441/Data/Folder_V1/churn_data_new.csv')

# # Utility file read

# In[ ]:


from Utiity_script_V15 import Person
Future = Person("Shyam", "29")
print(Future.name)
print(Future.age)

# # Data Transformation save

# In[ ]:


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pickle import dump
# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=1)
# define scaler
scaler = MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(X_train);
# transform the training dataset
X_train_scaled = scaler.transform(X_train)
from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
saved_model = nb.save_model(model = scaler, modelName = 'ScalerTransform', modelType = 'dp', X = None, y = None, estimator_type='')
#X and y are training datasets to get explainer dashboard.
#estimator_type is to specify algorithm type i.e., classification and regression.
#Only 'ml’ models with tabular data as input will support in Explainer Dashboard.
#Choose modelType = 'ml' for machine learning models, modelType = 'cv' for computer vision models and modelType = 'dp' for data transformation pickle files. 
#Provide ‘column_headers’ as a parameter if they have to be saved in the model.
#If using custom layer in keras, use native save functionality from keras.

# # Transformation load

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
loaded_model = nb.load_model('17171692773771012')

# # Transforming traing data

# In[ ]:


loaded_model.transform(X_train)
