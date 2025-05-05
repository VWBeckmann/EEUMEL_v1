
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
