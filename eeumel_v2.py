import os
import requests
import spacy
import logging
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from openai import OpenAIError

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

def print_directory_tree(start_path, prefix=""):
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

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
base_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(base_dir, "faiss_index")
faiss_index = None

if embeddings and os.path.exists(index_path):
    try:
        faiss_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully from %s.", index_path)
    except Exception as e:
        print(f"Path '{index_path}' not found. Showing current directory tree for debugging:")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print_directory_tree(current_dir)
        logger.exception("Error loading FAISS index.")
else:
    logger.warning("FAISS index not found or embeddings not available. Some queries may not work.")
    logger.warning(base_dir)

# Set up ChatGPT model
try:
    from langchain.chat_models import ChatOpenAI  # Assuming this is what you're using
    chat_model = ChatOpenAI(temperature=0)
    logger.info("Chat model initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize Chat model.")

# Load chat history
chat_history = []
logger.debug("Chat history initialized.")

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
    PORT = int(os.environ.get("PORT", 8080))  # OpenShift uses dynamic ports
    logger.info("Starting Flask app on Port " + str(PORT) + "...")
    app.run(host='0.0.0.0', port=PORT)
