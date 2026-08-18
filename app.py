import os
import uvicorn
import requests
import json

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre"
    )


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""

    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    """Get current weather for an Indian city."""

    indian_cities = [
        "hyderabad",
        "delhi",
        "new delhi",
        "mumbai",
        "bangalore",
        "bengaluru",
        "chennai",
        "kolkata",
        "pune",
        "ahmedabad",
        "jaipur",
        "lucknow",
        "kanpur",
        "nagpur",
        "indore",
        "bhopal",
        "visakhapatnam",
        "vizag",
        "vijayawada",
        "warangal",
        "tirupati",
        "goa",
        "surat",
        "patna",
        "ranchi",
        "kochi",
        "thiruvananthapuram",
        "mysore",
        "mysuru"
    ]

    city_lower = city.lower().strip()

    if city_lower not in indian_cities:
        return "Invalid input check properly"

    try:

        # Geocoding API
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        ).json()

        if "results" not in geo_response:
            return "Invalid input check properly"

        location = geo_response["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # Weather API
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        ).json()

        current = weather_response["current"]

        result = {
            "resolved_city": location["name"],
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except Exception:
        return "Invalid input check properly"


# ============================================================
# 2. TOOLS LIST
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 3. API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 4. MODEL
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 5. AGENT
# ============================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,

    system_prompt=(
        "You are an AI agent restricted to Indian weather "
        "and Indian cinema. "

        "You can also convert Celsius to Fahrenheit. "

        "Do not answer questions outside these capabilities."
    )
)


# ============================================================
# 6. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Message for the agent"
    )


# ============================================================
# 7. VALIDATION
# ============================================================

def is_valid_input(user_input: str) -> bool:

    text = user_input.lower().strip()

    # --------------------------------------------------------
    # Indian cities
    # --------------------------------------------------------

    indian_locations = [
        "india",
        "hyderabad",
        "delhi",
        "new delhi",
        "mumbai",
        "bangalore",
        "bengaluru",
        "chennai",
        "kolkata",
        "pune",
        "ahmedabad",
        "jaipur",
        "lucknow",
        "kanpur",
        "nagpur",
        "indore",
        "bhopal",
        "visakhapatnam",
        "vizag",
        "vijayawada",
        "warangal",
        "tirupati",
        "goa",
        "surat",
        "patna",
        "ranchi",
        "kochi",
        "thiruvananthapuram",
        "mysore",
        "mysuru"
    ]

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather_keywords = [
        "weather",
        "temperature",
        "rain",
        "rainfall",
        "forecast",
        "climate",
        "humidity",
        "wind",
        "hot",
        "cold",
        "sunny",
        "cloudy",
        "storm"
    ]

    # --------------------------------------------------------
    # Movies
    # --------------------------------------------------------

    movie_keywords = [
        "movie",
        "movies",
        "film",
        "films",
        "cinema",
        "bollywood",
        "tollywood",
        "kollywood",
        "actor",
        "actress",
        "director",
        "producer",
        "indian movie",
        "indian movies",
        "indian cinema",

        "rrr",
        "bahubali",
        "baahubali",
        "vikram",
        "3 idiots",
        "hera pheri",
        "munna bhai",
        "cargo",
        "mr india"
    ]

    # --------------------------------------------------------
    # Temperature conversion
    # --------------------------------------------------------

    conversion_keywords = [
        "celsius",
        "fahrenheit",
        "convert temperature",
        "temperature conversion"
    ]

    # --------------------------------------------------------
    # Check Indian weather
    # --------------------------------------------------------

    has_weather = any(
        keyword in text
        for keyword in weather_keywords
    )

    has_indian_location = any(
        location in text
        for location in indian_locations
    )

    if has_weather and has_indian_location:
        return True

    # --------------------------------------------------------
    # Check movies
    # --------------------------------------------------------

    if any(
        keyword in text
        for keyword in movie_keywords
    ):
        return True

    # --------------------------------------------------------
    # Check temperature conversion
    # --------------------------------------------------------

    if any(
        keyword in text
        for keyword in conversion_keywords
    ):
        return True

    # --------------------------------------------------------
    # Everything else is INVALID
    # --------------------------------------------------------

    return False


# ============================================================
# 8. EXTRACT RESPONSE
# ============================================================

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict):

                if "messages" in value:
                    messages = value["messages"]
                    break

    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            str(last_message)
        )

        return content

    return "Invalid input check properly"


# ============================================================
# 9. VALIDATE FIRST, THEN CALL AGENT
# ============================================================

def run_agent(x):

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    # ========================================================
    # IMPORTANT:
    # Invalid input NEVER reaches Gemini
    # ========================================================

    if not is_valid_input(user_input):

        return "Invalid input check properly"

    # ========================================================
    # Valid input → send to agent
    # ========================================================

    agent_input = {
        "messages": [
            ("user", user_input)
        ]
    }

    result = agent.invoke(agent_input)

    return extract_text_response(result)


# ============================================================
# 10. CREATE RUNNABLE
# ============================================================

formatted_agent_chain = RunnableLambda(
    run_agent
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 11. FASTAPI
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent"
)


# ============================================================
# 12. LANGSERVE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 13. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
