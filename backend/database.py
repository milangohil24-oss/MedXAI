import os
from pymongo import MongoClient

# Fetch MONGO_URI from Render environment variables, or fall back to your Atlas cluster URI
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://milangohil24_db_user:1pbGJWqjFaMUkoOH@cluster0.y1644ws.mongodb.net/medxai_db?retryWrites=true&w=majority"
)

client = MongoClient(MONGO_URI)
db = client["medxai_db"]

users_collection = db["users"]
analyses_collection = db["analyses"]
reports_collection = db["reports"]


def get_db():
    return db