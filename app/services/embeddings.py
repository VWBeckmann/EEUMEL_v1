import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .utils import print_directory_tree

def get_faiss_index():
    try:
        embeddings = OpenAIEmbeddings()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(base_dir, "../../faiss_index")
        
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print(f"Path '{index_path}' not found. Directory tree:")
            print_directory_tree(os.path.dirname(os.path.abspath(__file__)))
            logging.warning("FAISS index not found.")
            return None
    except Exception:
        logging.exception("Failed to load FAISS index.")
        return None
