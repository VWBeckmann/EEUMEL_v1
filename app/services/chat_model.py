import logging
from langchain_community.chat_models import ChatOpenAI

def get_chat_model():
    try:
        return ChatOpenAI(temperature=0)
    except Exception as e:
        logging.exception("Failed to initialize Chat model.")
        return None
