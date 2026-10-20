import pymongo
from pymongo import MongoClient

try:
    # Connect to MongoDB Atlas cluster
    uri = "mongodb+srv://pdurand365_db_user:1SwMqFq3et9M3iNV@ufc-fight-analyzer.coxtmry.mongodb.net/"
    client = MongoClient(
        uri,
        server_api=pymongo.server_api.ServerApi(
            version="1",
            strict=True,
            deprecation_errors=True
        )
    )
    # Verify connection
    client.admin.command("ping")
    print("Connected to MongoDB")






    client.close()
    
except Exception as e:
    raise Exception("The following error occurred: ", e)