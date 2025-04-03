import os
import requests
import spacy
from flask import Flask, request, jsonify, render_template_string
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from geopy.geocoders import Nominatim

# Load API keys securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set!")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings
embeddings = OpenAIEmbeddings()

# Load FAISS index safely
index_path = "faiss_index"
faiss_index = None
if os.path.exists(index_path):
    faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully.")
else:
    print("⚠️ FAISS index not found! Some queries may not work.")

# Set up ChatGPT model
chat_model = ChatOpenAI(temperature=0)

# Load chat history (ensure it is defined)
chat_history = []

# Weather Agent
class WeatherAgent:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY", "your-fallback-api-key")
        self.base_url = "https://api.open-meteo.com/v1/forecast?"
        self.nlp = spacy.load("en_core_web_sm")

    def get_weather(self, utterance):
        city = self.get_city_from_utterance(utterance)
        if not city:
            return "Location not found."

        latitude, longitude = self.get_coordinates(city)
        if not latitude or not longitude:
            return "Location not found."

        response = requests.get(f"{self.base_url}latitude={latitude}&longitude={longitude}&current_weather=true")
        if response.status_code != 200:
            return f"Error retrieving weather data: {response.status_code}"

        temp = response.json()["current_weather"]["temperature"]
        return f"Die aktuelle Temperatur in {city} beträgt {temp}°C."

    def get_coordinates(self, city_name):
        """Fetch latitude and longitude"""
        geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={self.api_key}"
        response = requests.get(geocode_url)
        if response.status_code == 200 and response.json():
            return response.json()[0]["lat"], response.json()[0]["lon"]
        return None, None

    def get_city_from_utterance(self, utterance):
        """Extract city name from user query"""
        doc = self.nlp(utterance)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                return ent.text
        return None

# Retrieval-based QA chain
if faiss_index:
    car_qa_chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=faiss_index.as_retriever())

def get_car_manual_response(query):
    """Process car manual queries"""
    if faiss_index:
        return car_qa_chain.run(question=query, chat_history=chat_history)
    return "Car manual database is unavailable."

# Multi-Agent System
class MultiAgentSystem:
    def __init__(self, weather_agent, car_manual_agent):
        self.weather_agent = weather_agent
        self.car_manual_agent = car_manual_agent

        selection_prompt_template = (
            "You are a highly intelligent assistant capable of answering a wide variety of questions."
            "Given the following list of agents and their capabilities, determine which agent should handle the user query: "
            "1. Weather Agent: Can provide current weather information for any city. "
            "2. Car Manual Agent: Can provide details and explanations about car models and their manuals. "
            "User Query: {query} "
            "Choose the appropriate agent to handle this query. Respond with the agent's name ('Weather Agent' or 'Car Manual Agent')"
        )

        selection_prompt = PromptTemplate(template=selection_prompt_template, input_variables=["query"])
        self.selection_chain = LLMChain(llm=chat_model, prompt=selection_prompt)

    def process_query(self, query):
        selection_agent_answer = self.selection_chain.run(query=query)
        while "|" not in selection_agent_answer:
            selection_agent_answer = self.selection_chain.run(query=query)
        
        agent_name, query_new = selection_agent_answer.split("|", 1)

        if agent_name.strip() == "Weather Agent":
            return self.weather_agent.get_weather(query_new.strip())
        elif agent_name.strip() == "Car Manual Agent":
            return get_car_manual_response(query_new.strip())
        return "No fitting Agent found"

# Initialize agents and system
weather_agent = WeatherAgent()
multi_agent_system = MultiAgentSystem(weather_agent=weather_agent, car_manual_agent=None)

# Flask API
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Interface</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        input { width: 300px; padding: 10px; margin-bottom: 10px; }
        button { padding: 10px; cursor: pointer; }
        #response { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h2>Enter Your Query</h2>
    <input type="text" id="queryInput" placeholder="Type your query here...">
    <button onclick="sendQuery()">Submit</button>
    <p id="response"></p>

    <script>
        function sendQuery() {
            let query = document.getElementById("queryInput").value;
            fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("response").innerText = "Response: " + data.response;
            })
            .catch(error => {
                document.getElementById("response").innerText = "Error: " + error;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    response = multi_agent_system.process_query(query)
    return jsonify({"query": query, "response": response})

if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 8080))  # OpenShift uses dynamic ports
    app.run(host='0.0.0.0', port=PORT)
