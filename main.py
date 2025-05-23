from app import create_app

app = create_app()

if __name__ == "__main__":
    import os
    PORT = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=PORT)
