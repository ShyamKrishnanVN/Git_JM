def input_fun():
    from Notebook.DSNotebook.NotebookExecutor import NotebookExecutor
    nb = NotebookExecutor()
    df_Time_Series_Test = nb.get_data('17171689931251954', '@SYS.USERID', 'True', {}, [], None, sparkSession)
    return df_Time_Series_Test.show(10)

# In[8]:


input_fun()
