import os
import requests
import spacy
import logging
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from openai import OpenAIError
import openai

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

# Load API keys securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY is not set!")
    raise ValueError("OPENAI_API_KEY is not set!")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
logger.debug("OPENAI_API_KEY loaded and set in environment.")

# Load embeddings
try:
    embeddings = OpenAIEmbeddings()
    logger.info("OpenAI embeddings loaded successfully.")
except Exception as e:
    logger.exception("Failed to load OpenAI embeddings.")
    embeddings = None

# Load FAISS index safely
index_path = "faiss_index"
faiss_index = None
if embeddings and os.path.exists(index_path):
    try:
        faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully from %s.", index_path)
    except Exception as e:
        logger.exception("Error loading FAISS index.")
else:
    logger.warning("FAISS index not found or embeddings not available. Some queries may not work.")

# Set up ChatGPT model
if ping_openai():
    try:
        chat_model = ChatOpenAI(temperature=0)
        logger.info("Chat model initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize Chat model.")
else:
    logger.warning("Skipping ChatGPT initialization due to OpenAI ping failure.")

# Load chat history
chat_history = []
logger.debug("Chat history initialized.")

# Weather Agent
class WeatherAgent:
    def __init__(self):
        try:
            self.api_key = os.getenv("WEATHER_API_KEY", "your-fallback-api-key")
            self.base_url = "https://api.open-meteo.com/v1/forecast?"
            self.nlp = spacy.load("en_core_web_sm")
            self.geolocator = Nominatim(user_agent="weather-agent")
            logger.info("WeatherAgent initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize WeatherAgent.")
            self.nlp = None
            self.geolocator = None

    def get_weather(self, utterance):
        logger.debug("Processing utterance: %s", utterance)

        if not self.nlp or not self.geolocator:
            logger.warning("WeatherAgent dependencies not fully initialized.")
            return "Wetteragent ist momentan nicht verfügbar."

        try:
            city = self.get_city_from_utterance(utterance)
            logger.debug("Extracted city: %s", city)
            if not city:
                return "Ich konnte den Ort leider nicht erkennen."

            latitude, longitude = self.get_coordinates(city)
            logger.debug("Coordinates for %s: (%s, %s)", city, latitude, longitude)
            if not latitude or not longitude:
                return f"Ich konnte die Koordinaten für {city} nicht ermitteln."

            url = f"{self.base_url}latitude={latitude}&longitude={longitude}&current_weather=true"
            response = requests.get(url)
            logger.debug("Weather API request URL: %s", url)

            if response.status_code != 200:
                logger.error("Weather API returned status %s", response.status_code)
                return f"Wetterdaten konnten nicht abgerufen werden (Status: {response.status_code})."

            temp = response.json()["current_weather"]["temperature"]
            return f"Die aktuelle Temperatur in {city} beträgt {temp}°C."

        except Exception as e:
            logger.exception("Error in get_weather.")
            return "Beim Abrufen der Wetterdaten ist ein Fehler aufgetreten."

    def get_city_from_utterance(self, utterance):
        try:
            doc = self.nlp(utterance)
            for ent in doc.ents:
                if ent.label_ == "GPE":  # Geo-political entity (like a city)
                    logger.debug("Found city in utterance: %s", ent.text)
                    return ent.text
            logger.debug("No GPE entity found in utterance.")
        except Exception as e:
            logger.exception("Error extracting city from utterance.")
        return None

    def get_coordinates(self, city_name):
        try:
            location = self.geolocator.geocode(city_name)
            if location:
                logger.debug("Geolocation found: %s -> (%s, %s)", city_name, location.latitude, location.longitude)
                return location.latitude, location.longitude
            else:
                logger.warning("No geolocation result for %s", city_name)
        except Exception as e:
            logger.exception("Error in geolocation lookup.")
        return None, None

def ping_openai():
    try:
        openai.Model.list()
        logger.info("Successfully connected to OpenAI API. Proxy is working.")
        return True
    except OpenAIError as e:
        logger.exception("OpenAI API ping failed. Check proxy or API key.")
        return False

# Flask App Initialization
app = Flask(__name__)

@app.route("/")
def index():
    logger.debug("Rendering index.html")
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("query", "")
    logger.debug("Received query: %s", question)

    try:
        if not question:
            logger.warning("Empty question received.")
            return jsonify({"response": "Bitte stellen Sie eine Frage."})

        if not chat_model:
            logger.warning("Chat model not initialized.")
            return jsonify({"response": "Das Sprachmodell ist momentan nicht verfügbar."})

        if not faiss_index:
            logger.warning("FAISS index not available.")
            return jsonify({"response": "Die Wissensdatenbank ist momentan nicht verfügbar."})

        chain = ConversationalRetrievalChain.from_llm(chat_model, retriever=faiss_index.as_retriever())
        result = chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        logger.info("Query processed successfully.")
        return jsonify({"response": result["answer"]})
    except Exception as e:
        logger.exception("Failed to handle query.")
        return jsonify({"response": "Bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten."})

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(debug=True)
