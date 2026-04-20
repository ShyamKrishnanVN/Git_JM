#!/usr/bin/env python
# coding: utf-8

# In[1]:


@ENV.ATHINA_ID
@ENV.ATHINA_KEY
@ENV.ATHINA_REGION
@ENV.ATHINA_DBNAME 


# In[2]:


import boto3
from pyathena import connect
import pandas as pd

def connect_to_athena_V1(database, s3_staging_dir, region_name, aws_access_key_id, aws_secret_access_key):
    # Create connection
    conn = connect(region_name=region_name,
                   aws_access_key_id=aws_access_key_id,
                   aws_secret_access_key=aws_secret_access_key,
                   s3_staging_dir=s3_staging_dir,
                   schema_name=database)

    return conn.cursor()

# Example usage
# database = 'pipeline_athena'  # Replace with your database name
# s3_staging_dir = 's3://your-bucket-name/'  # Replace with your S3 bucket name
# region_name = 'us-west-2'  # Replace with your region
# aws_access_key_id = 'AKIARV7DJ7DWPA75Y2LZ'  # Replace with your AWS access key ID
# aws_secret_access_key = 'sRiFReagleyhbM0im4EorlrNabnk9+USiTh+JgE/'  # Replace with your AWS secret access key

cursor = connect_to_athena_V1(s3_staging_dir = 's3://athenabdb/parquet/', region_name = @ENV.ATHINA_REGION, aws_access_key_id = @ENV.ATHINA_ID, aws_secret_access_key = @ENV.ATHINA_KEY, database=@ENV.ATHINA_DBNAME)


# In[ ]:


# Define your SQL query
query = "SELECT * FROM smith_table limit 10"  # Replace your_table_name with the actual table name

# Execute the query
# cursor = conn.cursor()
cursor.execute(query)

# Fetch the results
result_set = cursor.fetchall()
for row in result_set:
    print(row)


# In[ ]:


def new(a):
    return a
new(@ENV.ATHINA_REGION)


# In[ ]:


pqr_V1 = @ENV.ATHINA_REGION
pqr_V1


# In[ ]:


SYS = @SYS.ARTIFACT_PATH+'SYS_test.txt'
SYS

