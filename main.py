from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load variables from .env (expects OPENAI_API_KEY)
load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

response = model.invoke("Tell me why math is great!")
print(response.content)
