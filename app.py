import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from textblob import TextBlob
import json
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
import nltk
from paralleldots import set_api_key, get_api_key
from paralleldots import (
    similarity,
    ner,
    sentiment,
    keywords,
    intent,
    emotion,
    abuse,
    batch_intent,
    batch_emotion,
    batch_abuse,
    batch_ner,
    batch_sentiment,
    batch_keywords,
    batch_phrase_extractor,
    set_api_key,
    taxonomy,
    phrase_extractor,
    sarcasm,
    batch_sarcasm,
    custom_classifier,
    multilang_keywords,
    batch_taxonomy,
    facial_emotion_url,
    object_recognizer_url,
    object_recognizer,
    facial_emotion,
    target_sentiment,
    batch_target_sentiment,
)

# firebase
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

app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))
db = firestore.client()

n = 0


@app.route("/")
def home():
    return render_template("bot.html")


@app.route("/predict/", methods=["POST"])
def predict():
    # response = request.get_json()
    # blob = TextBlob(response["message"])
    # for sentence in blob.sentences:
    #     result = sentence.sentiment.polarity
    # print(result)
    global n
    n += 1
    set_api_key("tEgayBzxTpAZZNppX62n8niYYoHeTna20DqQw8S9TQU")
    response = request.get_json()
    response = response["message"]

    senti = sentiment(response)
    emot = emotion(response)
    # ab = abuse(response)
    # print(emot)
    negative = (senti["sentiment"])["negative"]
    # print(negative)
    # print((ab["abuse"])["Abusive"])
    # print(ab)
    # from collections import Counter

    # lower_text = response.lower()
    # cleaned = lower_text.translate(str.maketrans("", "", string.punctuation))
    # tokens = word_tokenize(cleaned)
    # nltk.download
    # from nltk.corpus import stopwords

    # stop_words = set(stopwords.words("english"))

    # final_words = []

    # for word in tokens:
    #     if word not in stop_words:
    #         final_words.append(word)

    # from nltk.stem import PorterStemmer

    # ps = PorterStemmer()
    # stemmed_words = []
    # for word in final_words:
    #     stemmed_words.append(ps.stem(word))
    # # print(stemmed_words)

    # from nltk.stem import WordNetLemmatizer

    # lemmatizer = WordNetLemmatizer()
    # lemma_words = []
    # for word in final_words:
    #     lemma_words.append(lemmatizer.lemmatize(word, "a"))
    # print(lemma_words)

    # emotion_list = []
    # with open("emotions.txt", "r") as file:
    #     for line in file:
    #         clear_line = (
    #             line.replace("\n", "").replace(",", "").replace("'", "").strip()
    #         )
    # word, emotion = clear_line.split(":")
    # if word in stemmed_words:
    #     emotion_list.append(emotion)
    # print(emotion_list)
    # count = 0
    # for word in emotion_list:
    #     if word == " sad":
    #         count = count + 1
    # w = Counter(emotion_list)
    # # print(w.keys().sad)
    # plt.bar(w.keys(), w.values())

    # set
    # doc_ref = db.collection('cbt').document('user1')
    # doc_ref.set({
    #     'negative': (result['sentiment'])['negative'],
    #     'neutral': (result['sentiment'])['neutral'],
    #     'positive': (result['sentiment'])['positive'],

    # })
    users_ref = db.collection(u"user1")
    docs = users_ref.stream()

    for doc in docs:
        # print(u"{}=> {}".format(doc.id, doc.to_dict()))
        if doc.id == "data":
            data = doc.to_dict()
            # print(data)
        elif doc.id == "session" + str(data["sessions"] + 1):
            session = doc.to_dict()
            # print(session)
        else:
            result = doc.to_dict()
            # print(result)

    positive_init = (session["sentiment"])["positive"]
    positive = (senti["sentiment"])["positive"]
    average_positive = (positive_init + positive) / 2
    neutral_init = (session["sentiment"])["neutral"]
    neutral = (senti["sentiment"])["neutral"]
    average_neutral = (neutral_init + neutral) / 2
    negative_init = (session["sentiment"])["negative"]
    negative = (senti["sentiment"])["negative"]
    average_negative = (negative_init + negative) / 2

    # abuse_init = (session["abuse"])["abusive"]
    # abu = (ab["abuse"])["abusive"]
    # average_abusive = (abuse_init + abu) / n
    # hate_init = (session["abuse"])["hate_speech"]
    # hate = (ab["abuse"])["hate_speech"]
    # average_hate = (hate_init + hate) / n
    # neither_init = (session["abuse"])["neither"]
    # neither = (ab["abuse"])["neither"]
    # average_neither = (neither + neither_init) / n

    angry_init = (session["emotion"])["angry"]
    angry = (emot["emotion"])["Angry"]
    average_angry = (angry_init + angry) / 2

    bored_init = (session["emotion"])["bored"]
    bored = (emot["emotion"])["Bored"]
    average_bored = (bored_init + bored) / 2

    excited_init = (session["emotion"])["excited"]
    excited = (emot["emotion"])["Excited"]
    average_excited = (excited_init + excited) / 2

    fear_init = (session["emotion"])["fear"]
    fear = (emot["emotion"])["Fear"]
    average_fear = (angry_init + fear) / 2

    happy_init = (session["emotion"])["happy"]
    happy = (emot["emotion"])["Happy"]
    average_happy = (happy_init + happy) / 2

    sad_init = (session["emotion"])["sad"]
    sad = (emot["emotion"])["Sad"]
    average_sad = (sad_init + sad) / 2

    doc_ref = db.collection("user1").document("session" + str(data["sessions"] + 1))

    if n <= 4:
        doc_ref.set(
            {
                "sentiment": {
                    "positive": average_positive,
                    "neutral": average_neutral,
                    "negative": average_negative,
                },
                "emotion": {
                    "angry": average_angry,
                    "bored": average_bored,
                    "excited": average_excited,
                    "fear": average_fear,
                    "happy": average_happy,
                    "sad": average_sad,
                }
                #######################################RESET CODE
                # "sentiment": {
                #     "positive": float(0),
                #     "neutral": float(0),
                #     "negative": float(0),
                # },
                # "emotion": {
                #     "angry": float(0),
                #     "bored": float(0),
                #     "excited": float(0),
                #     "fear": float(0),
                #     "happy": float(0),
                #     "sad": float(0),
                # },
            },
            merge=True,
        )
        print(n)
    else:
        doc_ref.set(
            {
                "sentiment": {
                    "positive": average_positive,
                    "neutral": average_neutral,
                    "negative": average_negative,
                },
                "emotion": {
                    "angry": average_angry,
                    "bored": average_bored,
                    "excited": average_excited,
                    "fear": average_fear,
                    "happy": average_happy,
                    "sad": average_sad,
                }
                #######################################RESET CODE
                # "sentiment": {
                #     "positive": float(0),
                #     "neutral": float(0),
                #     "negative": float(0),
                # },
                # "emotion": {
                #     "angry": float(0),
                #     "bored": float(0),
                #     "excited": float(0),
                #     "fear": float(0),
                #     "happy": float(0),
                #     "sad": float(0),
                # },
            },
            merge=True,
        )
        res_ref = db.collection("user1").document("result")
        res_ref.set(
            {
                "sentiment": {
                    "positive": average_positive,
                    "neutral": average_neutral,
                    "negative": average_negative,
                },
                "emotion": {
                    "angry": average_angry,
                    "bored": average_bored,
                    "excited": average_excited,
                    "fear": average_fear,
                    "happy": average_happy,
                    "sad": average_sad,
                }
                #######################################RESET CODE
                # "sentiment": {
                #     "positive": float(0),
                #     "neutral": float(0),
                #     "negative": float(0),
                # },
                # "emotion": {
                #     "angry": float(0),
                #     "bored": float(0),
                #     "excited": float(0),
                #     "fear": float(0),
                #     "happy": float(0),
                #     "sad": float(0),
                # },
            },
            merge=True,
        )
        n = 0
        ses_ref = db.collection("user1").document("data")
        ses_ref.set({"sessions": int(data["sessions"] + 1)}, merge=True)
        new_ref = db.collection("user1").document("session" + str(data["sessions"] + 2))
        new_ref.set(
            {
                "sentiment": {
                    "positive": float(0),
                    "neutral": float(0),
                    "negative": float(0),
                },
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

    # print(sentiment)

    # print(count)

    # from twilio.rest import Client
    # account_sid = 'AC0025439265e463bd42eec06fdce38f16'
    # auth_token = '94a4bfe4931f0792ea0679350eac1f01'
    # if count >= 2:
    #     client = Client(account_sid, auth_token)
    #     message = client.messages.create(
    #                         from_='whatsapp:+14155238886',
    #                         body='Your friend is depressed, he needs help.',
    #                         to='whatsapp:+919689920287'
    #                     )
    #     print(message.sid)

    return jsonify("1")


@app.after_request
def add_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    return response


if __name__ == "__main__":
    app.run(debug=True)
