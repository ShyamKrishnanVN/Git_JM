#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#sklearn train model
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = MultinomialNB()
model.fit(X_train, Y_train);

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
saved_model = nb.save_model(model = model, modelName = 'Model_Test_V2', modelType = 'ml', X = None, y = None, estimator_type='')
#X and y are training datasets to get explainer dashboard.
#estimator_type is to specify algorithm type i.e., classification and regression.
#Only 'ml’ models with tabular data as input will support in Explainer Dashboard.
#Choose modelType = 'ml' for machine learning models, modelType = 'cv' for computer vision models and modelType = 'dp' for data transformation pickle files. 
#Provide ‘column_headers’ as a parameter if they have to be saved in the model.
#If using custom layer in keras, use native save functionality from keras.

# In[ ]:


from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
nb = NotebookExecutor()
loaded_model = nb.load_saved_model('11561711013856078')

# In[ ]:


nb.predict(model = loaded_model, dataframe = X_test, modeltype='ml') 
 #Choose modeltype 'ml' for machine learning models and 'cv' for computer vision model 
 #ex: For machine learning model nb.predict(model = model, modeltype = 'ml', dataframe = df) 
 #ex: For computer vision keras model nb.predict(model = model, modeltype = 'cv', imgs = imgs, imgsize = (28, 28), dim = 1, class_names = class_names) 
 #and for pytorch model(model = model, modeltype = 'cv', imgs = imgs, class_names = class_names) 
 #Note: incase any error in prediction user squeezed image data in keras
