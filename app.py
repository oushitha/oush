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


@tool
def search_movies(genre: str) -> str:
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "drama": "12th Fail, Taare Zameen Par, Dangal",
        "romance": "Jab We Met, Sita Ramam, Rockstar",
        "thriller": "Drishyam, Andhadhun, Kahaani"
    }

    return movies.get(
        genre.lower(),
        "No Indian movies found for that genre."
    )


@tool
def change__to_f(temp_c: float) -> float:
    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        ).json()
    except Exception:
        return f"Could not find weather data for city: {city}"

    if "results" not in geo_response:
        return f"Could not find weather data for city: {city}"

    location = geo_response["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    country = location.get("country", "")

    if country.lower() != "india":
        return "Weather information is available only for Indian cities."

    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius"
    }

    try:
        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        ).json()["current"]
    except Exception:
        return f"Could not retrieve weather information for {city}."

    result = {
        "resolved_city": location["name"],
        "temperature_celsius": weather_response["temperature_2m"],
        "weather_code": weather_response["weather_code"]
    }

    return json.dumps(result)


tools = [
    get_weather,
    search_movies,
    change__to_f
]


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are an Indian weather and Indian cinema assistant. "
        "You are ONLY allowed to answer questions related to "
        "weather in India, Indian cities weather, Indian movies, "
        "Indian cinema, Indian actors, Indian actresses, Indian "
        "movie genres, and Indian film recommendations. "
        "If the user asks anything outside these topics, respond "
        "with exactly this sentence: "
        "'I am not authorized to answer questions outside of Indian weather and cinema.' "
        "Do not answer general knowledge questions. "
        "Do not answer programming questions. "
        "Do not answer personal questions. "
        "Do not answer questions about countries other than India."
    )
)


class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the agent"
    )


def is_allowed_question(user_input: str) -> bool:
    text = user_input.lower().strip()

    weather_keywords = [
        "weather",
        "temperature",
        "forecast",
        "rain",
        "raining",
        "climate",
        "humidity",
        "hot",
        "cold",
        "wind",
        "sunny",
        "cloudy",
        "storm",
        "heat",
        "degrees",
        "celsius",
        "fahrenheit"
    ]

    indian_cities = [
        "hyderabad",
        "delhi",
        "mumbai",
        "chennai",
        "bangalore",
        "bengaluru",
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
        "vijayawada",
        "warangal",
        "tirupati",
        "kochi",
        "goa",
        "patna",
        "surat",
        "varanasi",
        "agra",
        "amritsar",
        "nashik",
        "noida",
        "gurgaon",
        "gurugram"
    ]

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
        "movie recommendation",
        "film recommendation",
        "movie genre",
        "action movie",
        "comedy movie",
        "romance movie",
        "thriller movie",
        "sci-fi movie",
        "drama movie"
    ]

    indian_movies = [
        "rrr",
        "baahubali",
        "bahubali",
        "3 idiots",
        "hera pheri",
        "munna bhai",
        "dangal",
        "12th fail",
        "drishyam",
        "andhadhun",
        "kahaani",
        "jab we met",
        "rockstar",
        "sita ramam",
        "cargo",
        "mr. india",
        "vikram"
    ]

    if any(word in text for word in weather_keywords):
        return True

    if any(city in text for city in indian_cities):
        if any(word in text for word in weather_keywords):
            return True

    if any(word in text for word in movie_keywords):
        return True

    if any(movie in text for movie in indian_movies):
        return True

    return False


def format_for_agent(x):
    user_input = x["input"] if isinstance(x, dict) else x.input

    if not is_allowed_question(user_input):
        return {
            "messages": [
                (
                    "assistant",
                    "I am not authorized to answer questions outside of Indian weather and cinema."
                )
            ],
            "blocked": True
        }

    return {
        "messages": [
            ("user", user_input)
        ],
        "blocked": False
    }


def run_agent(x):
    if x.get("blocked", False):
        return {
            "messages": x["messages"]
        }

    return agent.invoke({
        "messages": x["messages"]
    })


def extract_text_response(agent_output):
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)

        if content is not None:
            return str(content)

        return str(last)

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | RunnableLambda(run_agent)
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


app = FastAPI(
    title="Indian Weather and Cinema Agent"
)


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


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
