from io import BytesIO
from time import perf_counter 
import logging 
logging.basicConfig() 
logger = logging.getLogger('MTA')
logging_level = logging.DEBUG
logger.setLevel(logging_level)
import boto3
from botocore.exceptions import ClientError
<<<<<<< HEAD
import pandas as pd
=======
import pandas
>>>>>>> 6c187ccc1a553b1acae2e641c3fc9dcb6d2afa44
#Test now 
def()
def get_athena_connection(REGION_NAME,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY):
   
    athena_client = boto3.client(
        'athena', 
        region_name=REGION_NAME, 
        aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    return athena_client

def run_athena_query(query, REGION_NAME,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,ATHENA_DATABASE,ATHENA_STATUS_OUTPUT_LOCATION,
                     S3_TARGET_BUCKET,S3):
    print('Starting Athena query run')
    logger.debug('Connecting to Athena')
    ATHENA_CLIENT = get_athena_connection(REGION_NAME,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY)
    logger.debug(f'Starting Athena query execution')
    logger.debug(f'Athena query:\n{query}')

    response = ATHENA_CLIENT.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': ATHENA_DATABASE},
        ResultConfiguration={'OutputLocation': ATHENA_STATUS_OUTPUT_LOCATION}
    )

    # Get the query execution ID
    query_execution_id = response['QueryExecutionId']

    # Wait for the query to complete
    while True:
        query_status = ATHENA_CLIENT.get_query_execution(QueryExecutionId=query_execution_id)
        status = query_status['QueryExecution']['Status']['State']

        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        # Retrieve and return the query results if the query succeeded
    if status == 'SUCCEEDED':
#         results = ATHENA_CLIENT.get_query_results(QueryExecutionId=query_execution_id)
#         results = format_athena_results(results)
        s3_response_object = S3.get_object(Bucket=S3_TARGET_BUCKET, Key=f"{ATHENA_STATUS_OUTPUT_LOCATION.split('/')[-2]}/{query_execution_id}.csv")
        object_content = s3_response_object['Body'].read()
        results = pd.read_csv(BytesIO(object_content))
        logger.debug(f'Athena query ran successfully')
        return results
    elif status == 'FAILED':
        raise Exception('Query from Athena failed')
