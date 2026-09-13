from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@tool("ConvertToEuros", description="Convert an amount from a given currency to euros.")
def convert_to_euros(amount: float, currency: str) -> str:
    """Convert the given amount from the specified currency to euros."""
    rates = {"USD": 0.92, "GBP": 1.17, "CAD": 0.68}
    rate = rates.get(currency.upper(), 1.0)
    return f"{amount} {currency.upper()} = {round(amount * rate, 2)} EUR"


model = init_chat_model(
    model="openrouter:openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=2048,
)

agent = create_agent(
    model=model,
    tools=[convert_to_euros],
    system_prompt="Tu es un assistant utile.",
)


def main() -> None:
    config = {"configurable": {"thread_id": "demo"}}

    stream = agent.stream(
        {
            "messages": [
                {"role": "user", "content": "Combien font 200 dollars américains en euros ?"}
            ]
        },
        config=config,
        stream_mode="messages",
    )

    for message, _metadata in stream:
        for block in message.content_blocks:
            if block.get("type") == "text":
                print(block["text"], end="", flush=True)


if __name__ == "__main__":
    main()