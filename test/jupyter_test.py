#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML

def mask_value(value):
    return HTML(f'<input type="password" value="{value}" readonly style="border:none; background: none; width: auto;">')

# Example usage:
secret_value = "my_secret_value"
ENV_SECRET = mask_value(secret_value)
ENV_SECRET

# In[8]:


sec = @ENV.SECRET_VALUE

# In[6]:


def foo():
    return (@ENV.MYSQL_DM_USERNAME)

# In[14]:


@ENV.MYSQL_DM_USERNAME.__repr__()

# In[13]:


def new(@ENV.MYSQL_DM_PASSWORD):
    return df

# In[1]:


code = '''
_var1 = 'This is a sample startup file'
_var2 = 1+2
import pickle, os
_var3 = os.getcwd()
'''
with open(@SYS.DATASANDBOX_PATH + '8290341/Data/startup.py', 'w') as f:
    f.write(code)

# In[2]:


xx = @SYS.DATASANDBOX_PATH + '8290341/Data/startup.py'
