import requests
import json
import os
import datetime
import tarfile
import shutil
import logging
import aiohttp
logger = logging.getLogger()

def save_artifact(data, name, spacekey, experimentId, UserId):
    try:
        with open( name, "w+") as file:
            file.write(json.dumps(data))
    except Exception as e:
        logger.error(f'ERROR (Save_artifacts) =====> {e}')
        raise Exception(": Please provide Name with extension eg:(.csv,.json,.txt)" + str(e))
    spacekey = str(spacekey)
    head, tail = os.path.split(name)

    res = createNoteBookModel_apiCall(experimentId, UserId, spacekey, 3, tail)  # type = 3 for automl output json
    res = res.json()
    if res["panotebook"]["success"] == False:
        return {"success": False, "Message": "Artifact already exists, please give a new name."}
    modelId = res["panotebook"]["notebookModel"]['id']
    try:
        artifact_path = f'{os.getenv("SANDBOX_PATH")}/{experimentId}/automl/'
        if not os.path.exists(artifact_path):
            os.makedirs(artifact_path)
        json.dump(data, open(artifact_path + name, 'w'))
        # files = [('file', (tail, open( tail, 'rb'), 'application/octet-stream'))]
        # res2 = uploadNotebookModelChunkByChunk_apicall(experimentId, UserId, spacekey, tail, 3, files, modelId, head)
        # res2 = res2.json()
        # if res2['success'] != True:
        #     return {"success": False, "Message": "Error while uploading artifact"}
        return {"success": True, "Message": "Saved."}
    except Exception as e:
        print(e)


def save_native(model_path, spacekey, experimentId, UserId, modelType="am", dirr = "false", model_class = "ag", Model_Name= "", model_list=[]):
    metadata = {"modelType": modelType,"class":model_class, "model_name":Model_Name}
    metadata = json.dumps(metadata)
    spacekey = str(spacekey)
    head, modelname = os.path.split(model_path)
    res = createNoteBookModel_apiCall(experimentId, UserId, spacekey, 1, modelname, metadata,
                                      modelType)  # type = 1 for models
    res = res.json()
    if res["panotebook"]["success"] == False:
        logger.error(f'ERROR WHILE SAVING MODEL (Service response): {res}')
        logger.error(f'ERROR WHILE SAVING MODEL (model name): {modelname}')
        return {"success": False, "Message": res["panotebook"].get("message", 'Model already exists, please give a new name.')}
    else:
        modelId = res["panotebook"]["notebookModel"]['id']

        transform_path = 'transform.pkl'
        if dirr == "false":
            path = model_path
            if not os.path.exists(path):
                os.makedirs(path)
            file = [('file', (modelname, open(path, 'rb'), 'application/octet-stream'))
                    ]
            res = uploadNotebookModelChunkByChunk_apicall(experimentId, UserId, spacekey, modelname, 1, file, modelId, head="")
            res = res.json()
            if res['success'] != True:
                deleteNotebookModel(modelId, UserId, spacekey)
                return {"success": False, "Message": "Error while uploading model"}
        else:
            tar_file = f'{Model_Name}.tar.gz'
            tar = tarfile.open(tar_file, 'w:gz')

            # for item in model_list:
            #     dir_path = dirr + item
            models_path = f'{dirr}models/'
            for subdir, dirs, files in os.walk(models_path):
                try:
                    for file in files:
                        fullpath = (os.path.join(subdir, file))
                        tar.add(fullpath, fullpath.replace(models_path, ''))
                except Exception as e:
                    print(e)
            dataset_path = f'{dirr}utils/data/'
            for subdir, dirs, files in os.walk(dataset_path):
                try:
                    for file in files:
                        fullpath = (os.path.join(subdir, file))
                        tar.add(fullpath, fullpath.replace(subdir, ''))
                except Exception as e:
                    print(e)
            tar.add(transform_path)
            tar.close()
        try:
            tar_file_path = f'{os.getenv("SANDBOX_PATH")}/{experimentId}/model/{modelId}/'
            if not os.path.exists(tar_file_path):
                os.makedirs(tar_file_path)
            shutil.move(tar_file, tar_file_path)
            tar = tarfile.open(tar_file_path + tar_file, 'r:gz')
            tar.extractall(tar_file_path)
            tar.close()
            return {"success": True, "Message": "Model Uploaded Successfully"}
        except Exception as e:
            return {"success": False, "Message": "Error while uploading model"}

def createNoteBookModel_apiCall(projectId, UserId, Spacekey, type, name="", metaData="", modelType="", model_id="", notebookId=None, parentId=None):
    base_url = str(os.environ['PL_BASE_LAYER'])
    url = base_url + "/BizVizEP/services/rest/notebook/createNoteBookModel"
    date = datetime.datetime.now().timestamp() * 1000
    data = json.dumps({
        "panotebook": {
            "paNotebook": {
                "id": notebookId
            },
            "notebookModel": {
                "id": model_id,
                "projectId": projectId,
                "parentId":parentId,
                "spaceKey": Spacekey,
                "status": 1,
                "type": type,
                "description": "",
                "modelName": name,
                "data": "",
                "createdBy": UserId,
                "updatedBy": UserId,
                "createdDate": date,
                "isApiEnabled": 0,
                "modelType": modelType,
                "metaData": metaData
            }
        }
    })
    headers = {
        "SHARDKEY": Spacekey,
        "USERID": str(UserId),
        "Content-Type": "application/json"
    }
    res = requests.post(url=url, data=data, headers=headers)
    return res

def uploadNotebookModelChunkByChunk_apicall(projectId,UserId, Spacekey,name,type,files, modelId,head):
    print("start upload")
    base_url =  str(os.environ['DATA_SANDBOX_HOST'])
    url = base_url + "/cxf/files/uploadNotebookModelChunkByChunk"
    payload={
        'data': '{"iscellHeader":"true","delimiter":",","action":"save","name":"'+name+'"}',
        'spacekey': Spacekey,
        'userId': UserId,
        'chunk': '0',
        'chunks': '1',
        'name': name,
        'projectId': projectId,
        'description': '',
        'modelId': modelId,
        'type': type,
        'subdir':head,
        'compId':''
    }
    headers = {
    'userID': str(UserId),
    'spacekey': Spacekey,
    }
    res = requests.post(url = url, data = payload, headers=headers, files= files)
    print(res)
    print("end upload")
    return res  

def deleteNotebookModel(modelId, userId, spaceKey):
    base_url =  str(os.environ['DATA_SANDBOX_HOST'])
    url = base_url + "/BizVizEP/services/rest/notebook/deleteNoteBookModel"

    payload = json.dumps({
        "panotebook": {
            "notebookModel": {
                "spaceKey": str(spaceKey),
                "id": modelId
            },
            "user": {
                "id": userId
            }
        }
    })
    headers = {
        'SHARDKEY': str(spaceKey),
        'USERID': str(userId),
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

def getSettingsConfs(UserID, spacekey):
    headers = {
        "SHARDKEY": str(spacekey),
        "USERID": str(UserID),
        "Content-Type": "application/json"
    }
    data = '{ "settings": { "isActive": 0, "active": 1,  "type": "157" }}'
    rse = requests.post(
        os.environ["PL_BASE_LAYER"] +
        "/BizVizEP/services/rest/bizvizsettings/getSettingsConfs",
        headers=headers, data=data)
    return rse.json()


def getSandboxDatasetById(UserID, serviceID, prep_ids, spacekey):
    headers = {
        "SHARDKEY": str(spacekey),
        "USERID": str(UserID),
        "Content-Type": "application/json"
    }

    data = json.dumps({
        "sandbox_dataset": {
            "id": serviceID,
            "spaceKey": str(spacekey),
            "prepJson": json.dumps(prep_ids)
        }
    })

    rse = requests.post(
        os.environ["PL_BASE_LAYER"] +
        "/BizVizEP/services/rest/sandboxDataset/getSandboxDatasetById",
        headers=headers, data=data)

    logger.info(f'getSandboxDatasetById (Request-Body) =====> {rse.request.body}')
    logger.info(f'getSandboxDatasetById (Response) =====> {rse.text}')
    print(f'getSandboxDatasetById (Response) =====> {rse.text}')
    return rse.json()


def getqueryservicebyserviceid(UserID, serviceID, prep_ids, spacekey):
    headers = {
        "SHARDKEY": str(spacekey),
        "USERID": str(UserID),
        "Content-Type": "application/json"
    }

    data = json.dumps({
        "serviceId": serviceID,
        "spacekey": spacekey,
        "userid": UserID,
        "prepIds": prep_ids
    })

    rse = requests.post(
        os.environ["PL_BASE_LAYER"] +
        "/BizVizEP/services/rest/QueryServiceManager/getqueryservicebyserviceid",
        headers=headers, data=data)

    return rse.json()


def getEnvVars(UserID, env_lst, spacekey, namespace=''):
    headers = {
        "SHARDKEY": str(spacekey),
        "USERID": str(UserID),
        "Content-Type": "application/json"
    }
    env_data = {
        "envs": env_lst,
        "namespace": str(namespace) if namespace else str(os.getenv("PL_NAMESPACE",""))
    }
    env_data = json.dumps(env_data)
    env_rse = requests.post(
        str(os.getenv('MDS_BASE_URL')) + "/api/v1/getEnvVars", headers=headers,
        data=env_data)
    return env_rse.json()


def vieweditqueryservice(UserID, serviceID, spacekey):
    headers = {
        "SHARDKEY": str(spacekey),
        "USERID": str(UserID),
        "Content-Type": "application/json"
    }

    data = json.dumps({
        "queryServices": {
            "queryService": {
                "serviceId": serviceID,
                "spaceKey": spacekey
            }
        }
    })

    res = requests.post(
        os.environ["PL_BASE_LAYER"] +
        "/BizVizEP/services/rest/QueryServiceManager/vieweditqueryservice",
        headers=headers, data=data)

    return res.json()

def getNoteBookModelById(spacekey, userid, modelId):
    base_url = str(os.environ['PL_BASE_LAYER'])
    url = base_url + "/BizVizEP/services/rest/notebook/getNoteBookModelById"
    payload = json.dumps({
        "panotebook": {
            "notebookModel": {
                "id": modelId
            }
        }
    })
    headers = {
        'SHARDKEY': str(spacekey),
        'USERID': str(userid),
        'Content-Type': 'application/json'
    }
    response = requests.post(url=url, headers=headers, data=payload)
    response = response.json()
    return response

def save_autogluon_models(spacekey, experimentId, UserId, modelType="am", model_class = "ag", model_name="", parentId="", experiment_name="", target_column=""):
    metadata = {"modelType": modelType,"class":model_class, "model_name":model_name, 'target_column':target_column}
    metadata = json.dumps(metadata)
    spacekey = str(spacekey)
    res = createNoteBookModel_apiCall(projectId=experimentId, UserId=UserId, Spacekey=spacekey, type=1,
                                        name=experiment_name, metaData=metadata, modelType=modelType, parentId=parentId)  # type = 1 for models
    try:
        res = res.json()
        if res["panotebook"]["success"] == False:
            logger.error(f'ERROR WHILE SAVING MODEL (Service response): {res}')
            logger.error(f'ERROR WHILE SAVING MODEL (model name): {experiment_name}')
            raise Exception(res["panotebook"].get("message", 'Model already exists, please give a new name.'))
        else:
            uuid = res["panotebook"]["notebookModel"]['uuid']
            modelId = res["panotebook"]["notebookModel"]['id']
            return modelId, uuid
    except Exception as e:
        logger.error(f"Error while saving models =====> {e}")
        raise e

def update_model_metadata(projectID, userID, spacekey, model_id, metadata):
    data = getNoteBookModelById(spacekey, userID, model_id)
    notebookModelData = data.get('panotebook').get('notebookModel')
    if notebookModelData:
        metaData = json.loads(notebookModelData.get("metaData", "{}"))
    else:
        raise Exception()
        
    metaData.update(metadata)
    res = createNoteBookModel_apiCall(projectID, userID, spacekey, 1, metaData=json.dumps(metaData), model_id=model_id)
    res = res.json()
    if res["panotebook"]["success"] == False:
        raise Exception ("Failed to update 'meta_data'") from None

def sendStatus(spacekey, userid, experimentId, status, msg = "none"):
    url = str(os.getenv('PL_BASE_LAYER'))
    url = url + "/BizVizEP/services/rest/notebook/createAutoML"
    headers = {
        'SHARDKEY': spacekey, 'USERID': userid,  "Content-Type": "application/json"

    }
    payload = json.dumps({
        "panotebook":{
        "autoML": {
        "id": int(experimentId),
        "status": status,
        "logs": msg
                }
        }
    })
    response = requests.request("POST", url, headers=headers, data=payload)
    logger.info(f"Sent experiment status =====> {response.json()}")
    return

async def load_explainer(model_id, project_id, user_id, spacekey, experimentId=None, limit=100):
    url = f"{os.environ['BVZEXPLAINABILITY_GENERATOR_URL']}/loadDashboard"
    payload = json.dumps({
        "data": {
            "experimentId": experimentId,
            "modelId": int(model_id),
            "dataset": {
                "limit": limit
            }
        }
    })
    headers = {
        "PROJECT_ID":str(project_id),
        "CREATED_BY":str(user_id),
        "SPACE_KEY":str(spacekey),
        "Content-Type": "application/json"
    }

    session = aiohttp.ClientSession()
    response = await session.post(url, headers=headers, data=payload)
    return response

def get_automl_by_id(experimentId, spacekey, createdBy):
    base_url = str(os.environ['PL_BASE_LAYER'])
    url = base_url + "/BizVizEP/services/rest/notebook/getAutoMLById"
    payload = json.dumps({
        "panotebook": {
            "autoML": {
                "id": int(experimentId)
            }
        }
    })
    headers = {
        'SHARDKEY': str(spacekey),
        'USERID': str(createdBy),
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()
