from pymongo import MongoClient

def create_mongo_connection():
    uri='mongodb+srv://root:root@cluster0.acekv.mongodb.net/test?authSource=admin&replicaSet=atlas-svib4e-shard-0&readPreference=primary&ssl=true'
    client=MongoClient(uri)
    return client
