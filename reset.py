from google.cloud import storage
from firebase import firebase
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./sanfer.json"
from firebase_admin import credentials

cred = credentials.Certificate("./sanfer.json")
from firebase_admin import firestore
import firebase_admin
from firebase_admin import db


firebase_admin.initialize_app(
    cred, {"databaseURL": "https://depression-bot-sujmkv.firebaseio.com"}
)

db = firestore.client()


doc_ref = db.collection("user1").document("session1")
doc_ref.set(
    {
        "sentiment": {"positive": float(0), "neutral": float(0), "negative": float(0),},
        "emotion": {
            "angry": float(0),
            "bored": float(0),
            "excited": float(0),
            "fear": float(0),
            "happy": float(0),
            "sad": float(0),
        },
    },
    merge=True,
)

