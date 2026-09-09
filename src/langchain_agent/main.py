from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

model = ChatOpenRouter(
    model="openai/gpt-oss-120b",
    temperature=0,
)

agent = create_agent(
    model=model,
    system_prompt="Tu es un assistant utile.",
)


def main() -> None:
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Bonjour, qui es-tu ?"}
            ]
        }
    )
    print(result["messages"][-1].content)