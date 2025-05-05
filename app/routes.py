from flask import Blueprint, render_template, request, jsonify
from app.services.chat_model import get_chat_model
from app.services.embeddings import get_faiss_index
from langchain.chains import ConversationalRetrievalChain

bp = Blueprint("main", __name__)
chat_model = get_chat_model()
faiss_index = get_faiss_index()
chat_history = []

@bp.route("/")
def index():
    return render_template("index.html")

@bp.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("query", "")

    if not question:
        return jsonify({"response": "Bitte stellen Sie eine Frage."})

    if not chat_model:
        return jsonify({"response": "Das Sprachmodell ist momentan nicht verfügbar."})

    if not faiss_index:
        return jsonify({"response": "Die Wissensdatenbank ist momentan nicht verfügbar."})

    try:
        chain = ConversationalRetrievalChain.from_llm(chat_model, retriever=faiss_index.as_retriever())
        result = chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        return jsonify({"response": result["answer"]})
    except Exception:
        return jsonify({"response": "Bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten."})
