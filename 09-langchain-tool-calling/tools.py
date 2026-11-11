"""
Step 1 — Define tools.

A "tool" is just a normal Python function with a @tool decorator.
LangChain reads the function name, docstring, and type hints to build
a JSON schema that gets sent to the model, so the model knows the tool
exists and what arguments to call it with.

Rules that matter:
- ALWAYS write a clear docstring. The model uses it to decide whether
  to call the tool — a vague docstring = the model ignoring your tool.
- Type-hint every argument. No type hint = no schema = tool call fails
  or the model guesses wrong.
"""

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 1)'.
    Only use this for math. Supports + - * / ** ( )."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city name.
    Returns a short text description."""
    # Mocked so this repo runs with zero API keys. Swap this for a real
    # call (e.g. Open-Meteo, free, no key required) when you're ready.
    fake_data = {
        "karachi": "32°C, humid, partly cloudy",
        "lahore": "29°C, clear skies",
        "islamabad": "24°C, light rain",
    }
    return fake_data.get(city.lower(), f"No data for '{city}'. Try Karachi, Lahore, or Islamabad.")


@tool
def word_count(text: str) -> str:
    """Count the number of words in a given piece of text."""
    return str(len(text.split()))


# This is the list you'll pass into bind_tools() / create_agent() later.
all_tools = [calculator, get_weather, word_count]

if __name__ == "__main__":
    # Tools are still plain functions — you can call them directly.
    # .invoke() is the Runnable-standard way to call a tool manually.
    print(calculator.invoke({"expression": "12 * (4 + 1)"}))
    print(get_weather.invoke({"city": "Karachi"}))
    print(word_count.invoke({"text": "LangChain tool calling is not that scary"}))
