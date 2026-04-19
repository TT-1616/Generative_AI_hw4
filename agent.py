"""Math agent that solves questions using tools in a ReAct loop."""

import json
import re
import sys
import time

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from calculator import calculate

load_dotenv()

# Configure your model below. Examples:
#   "google-gla:gemini-2.5-flash"       (needs GOOGLE_API_KEY)
#   "openai:gpt-4o-mini"                (needs OPENAI_API_KEY)
#   "anthropic:claude-sonnet-4-6"    (needs ANTHROPIC_API_KEY)
MODEL = "google-gla:gemini-2.5-flash-lite"

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step. "
        "Use the calculator tool for arithmetic. "
        "Use the product_lookup tool when a question mentions products from the catalog. "
        "When a question asks for a total, difference, comparison, or best deal involving multiple products, "
        "call product_lookup separately for each product, then use the calculator tool for the arithmetic. "
        "Do not ask the user to choose a product when the product names are already in the question. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Examples: "847 * 293", "10000 * (1.07 ** 5)", "23 % 4"
    """
    return calculate(expression)


@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name.

    Use this when a question asks about product prices from the catalog.
    Call this once for each product when a question compares products,
    asks for the price difference, or asks for a total cost.
    """
    with open("products.json") as f:
        products = json.load(f)

    if product_name in products:
        return str(products[product_name])

    normalized_name = product_name.strip().lower()
    for name, price in products.items():
        if name.lower() == normalized_name:
            return str(price)

    if normalized_name.endswith("s"):
        singular_name = normalized_name[:-1]
        for name, price in products.items():
            if name.lower() == singular_name:
                return str(price)

    available_products = ", ".join(products.keys())
    return f"Product not found. Available products: {available_products}"


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def main():
    questions = load_questions()
    if len(sys.argv) > 1:
        question_number = int(sys.argv[1])
        questions_to_run = [(question_number, questions[question_number - 1])]
    else:
        questions_to_run = list(enumerate(questions, 1))

    for i, question in questions_to_run:
        print(f"## Question {i}")
        print(f"> {question}\n")

        while True:
            try:
                result = agent.run_sync(question)
                break
            except ModelHTTPError as e:
                if e.status_code != 429:
                    raise

                match = re.search(r"retry in ([\d.]+)s", str(e), re.IGNORECASE)
                wait_seconds = float(match.group(1)) + 1 if match else 30
                print(f"Rate limit reached. Waiting {wait_seconds:.1f} seconds...")
                time.sleep(wait_seconds)

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** `{part.tool_name}({part.args})`")
                elif kind == "tool-return":
                    print(f"- **Result:** `{part.content}`")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")


if __name__ == "__main__":
    main()
