import datetime
from pymongo import MongoClient
from datetime import datetime
import os

def connect_mongo_db():
    client = MongoClient("mongodb+srv://root:admin@cluster0.tblvtp2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client['data']

    return db

def send_data_to_db(first_name, last_name, start_time, data):
    # Create session in "sessions" collection (Only first time)
    # Send "data" to "hrv" collection (time series)
    db = connect_mongo_db()

    sessions_collection = db['sessions']
    hrv_collection = db['hrv']

    sessionID =  str(int(start_time)) + "_" + first_name.lower() + "_" + last_name.lower()
    start_time_readable = datetime.fromtimestamp(start_time)
    start_time_string = datetime.fromisoformat(str(start_time_readable))

    session_object = {
        "sessionID": sessionID,
        "first_name": first_name,
        "last_name": last_name,
        "start_time": start_time_readable,
        "start_time_string": start_time_string.strftime("%B %d, %Y, %H:%M:%S")
    }

    # Insert a new session if it doesn't exist
    if sessions_collection.find_one({"sessionID": sessionID}) is None:
        # The document does not exist, so insert it
        result = sessions_collection.insert_one(session_object)
        print("Inserted new session: ", result)

    # Don't post anything if data is empty
    if len(data) == 0:
        return
    
    hrv_object = {
        "sessionID": sessionID,
        "timestamp": start_time_readable,
        "data": data,
    }

    result = hrv_collection.insert_one(hrv_object)
    print("Inserted new hrv data for sessionID", sessionID)


    # print("ID:", str(int(start_time)) + "_" + first_name.lower() + "_" + last_name.lower())
    # print("Sending data to database:", data)
