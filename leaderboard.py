from flask import Flask, render_template
from pymongo import MongoClient
import os

app = Flask(__name__)

def connect_mongo_db():
    client = MongoClient("mongodb+srv://root:admin@cluster0.tblvtp2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client['data']
    return db

def get_average_hrv():
    db = connect_mongo_db()
    hrv_collection = db['hrv']

    # Aggregating average HRV per session
    pipeline = [
        {"$unwind": "$data"},  # Unwind the data array
        {"$group": {
            "_id": "$sessionID", 
            "averageHRV": {"$avg": "$data"}  # Calculate the average of the unwound data
        }},
        {"$sort": {"averageHRV": 1}}  # Sorting by averageHRV in ascending order
    ]
    result = list(hrv_collection.aggregate(pipeline))

    # Joining with sessions to get user names
    sessions_collection = db['sessions']
    leaderboard = []
    for entry in result:
        session = sessions_collection.find_one({"sessionID": entry["_id"]})
        if session:
            leaderboard.append({
                "name": session["first_name"] + " " + session["last_name"],
                "averageHRV": entry["averageHRV"]
            })

    return leaderboard


@app.route('/')
def leaderboard():
    data = get_average_hrv()
    return render_template('leaderboard.html', leaderboard=data)

if __name__ == '__main__':
    app.run(debug=True)
