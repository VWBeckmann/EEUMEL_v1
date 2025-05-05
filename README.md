# EEUMEL

This is EEUMEL_v1. A cleaned up structure of EEUMEL

# Structure

EEUMEL_v1/
├── app/
│   ├── __init__.py              # Initializes the Flask app with logging and environment config
│   ├── routes.py                # Main application routes (e.g., index, /query)
│   ├── services/                # Service layer (logic for models, embeddings, utils)
│   │   ├── __init__.py
│   │   ├── chat_model.py        # Loads and returns the ChatOpenAI model
│   │   ├── embeddings.py        # Loads OpenAI embeddings and FAISS index
│   │   ├── geolocation.py       # (Optional) Geolocation support using geopy
│   │   └── utils.py             # Utility functions (e.g., print_directory_tree)
├── templates/
│   └── index.html               # HTML template rendered on the home page
├── .env                         # Environment variables (e.g., OPENAI_API_KEY)
├── requirements.txt             # Python package dependencies
├── run.py                       # Application entry point
└── README.txt                   # This file
