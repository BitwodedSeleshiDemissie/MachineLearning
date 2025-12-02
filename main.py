from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool(
    "get_weather",
    description="Return weather information for a given city",
    return_direct=False,
)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    response.raise_for_status()
    return response.json()


@tool("locate_user", description="Look up a user's city based on the context")
def locate_user(user_id: str) -> str:
    match user_id:
        case "ABC123":
            return "vienna"
        case "XYZ456":
            return "london"
        case "HJKL111":
            return "paris"
        case _:
            return "Unknown"


model = init_chat_model("gpt-4o-mini", temperature=0.3)
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant who is humorous",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'what is the weather like in vienna?'}
    ],
    'context': Context(user_id='ABC123')
}, config=config)

print(response['structured_response'])
print(response['structured_response'].summary)
# Fixed typo: temprature_celsi -> temperature_celsius
print(response['structured_response'].temperature_celsius)
