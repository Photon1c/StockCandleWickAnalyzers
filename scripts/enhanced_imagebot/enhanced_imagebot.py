# Enhanced ImageBot utilities
import os
import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image
import tinytroupe
from tinytroupe.agent import TinyPerson, FilesAndWebGroundingFaculty
from tinytroupe.environment import TinyWorld

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Work around openai==1.3.x expecting httpx <0.28 (where `proxies` kwarg exists).
# By providing our own http_client, we avoid OpenAI constructing one with the
# removed `proxies` argument in httpx>=0.28.
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment (.env)")

# Provide an explicit httpx.Client to avoid openai constructing one with deprecated args
client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client())
screenshot_dir = r"F:\StillWater\Apps\Pecuniary\smartinvestor\tools\FalconEye\screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

memory_file = "agent_memory.json"

# Initialize agents with grounding faculty
grounding_faculty = FilesAndWebGroundingFaculty()

# Initialize agents
leslie = TinyPerson("Leslie")
leslie.define("age", 75)
leslie.define("occupation", "Baker, Mechanic, Accountant")
leslie.define("personality", {"traits": [
    "Patient and analytical.",
    "Enjoys explaining solutions to problems with numbers.",
    "Friendly and a good active listener."
]})
leslie.add_mental_faculties([grounding_faculty])

ashley = TinyPerson("Ashley")
ashley.define("age", 21)
ashley.define("occupation", "Model, Chemist, Gymnast")
ashley.define("personality", {"traits": [
    "Curious and analytical.",
    "Loves cooking healthy food.",
    "Very playful and energetic."
]})
ashley.add_mental_faculties([grounding_faculty])

# Create world
environment = TinyWorld("Trading Room", [leslie, ashley])
environment.make_everyone_accessible()

# Function to query OpenAI's GPT-4o
def get_gpt_response(prompt):
    try:
        completions = client.chat.completions.create(
            model="gpt-4",  # Changed from gpt-4o to gpt-4
            messages=[{"role": "system", "content": "You are an expert stock analyst guiding two AI personas."},
                    {"role": "user", "content": prompt}]
        )
        return completions.choices[0].message.content
    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return f"Error: {str(e)}"

# Function to capture TradingView chart
def capture_tradingview_chart(symbol):
    try:
        def build_options(use_new_headless: bool) -> Options:
            o = Options()
            o.add_argument("--window-size=1920x1080")
            o.add_argument("--headless=new" if use_new_headless else "--headless")
            o.add_argument("--no-sandbox")
            o.add_argument("--disable-gpu")
            o.add_argument("--disable-dev-shm-usage")
            o.add_argument("--log-level=3")
            return o

        try:
            driver = webdriver.Chrome(options=build_options(True))
        except Exception:
            driver = webdriver.Chrome(options=build_options(False))
        driver.get(f"https://www.tradingview.com/chart/?symbol={symbol}")
        time.sleep(5)
        screenshot_path = os.path.join(screenshot_dir, f"{symbol}_chart.png")
        driver.save_screenshot(screenshot_path)
        driver.quit()
        return screenshot_path
    except Exception as e:
        print(f"Error capturing chart: {e}")
        return None

# Function to analyze chart using GPT-4 Vision
def analyze_chart_with_gpt4o(image_path):
    try:
        with open(image_path, "rb") as image_file:
            # Encode the image as base64
            import base64
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
        response = client.chat.completions.create(
            model="gpt-4.1",  # Using GPT-4 Vision model
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the stock chart image and provide insights."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe trends, volatility, and possible price action."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing chart: {e}")
        return f"Error analyzing chart: {str(e)}"

def _serialize_entry(entry):
    try:
        # Prefer explicit text/content fields
        for key in ("text", "content", "message", "value"):
            if hasattr(entry, key):
                val = getattr(entry, key)
                if isinstance(val, (str, int, float, bool)) or val is None:
                    return val
                try:
                    return json.loads(json.dumps(val))
                except Exception:
                    return str(val)
        # Common to_dict/to_json style
        for key in ("to_dict", "model_dump"):
            if hasattr(entry, key) and callable(getattr(entry, key)):
                try:
                    d = getattr(entry, key)()
                    return json.loads(json.dumps(d))
                except Exception:
                    pass
        if hasattr(entry, "to_json") and callable(getattr(entry, "to_json")):
            try:
                j = getattr(entry, "to_json")()
                if isinstance(j, str):
                    return json.loads(j)
                return j
            except Exception:
                pass
    except Exception:
        pass
    # Fallback: string representation
    try:
        return str(entry)
    except Exception:
        return "<unserializable>"


def _serialize_agent_memory(agent):
    entries = getattr(agent, "episodic_memory", [])
    out = []
    for e in entries:
        out.append(_serialize_entry(e))
    return out


# Function to save agent memory (JSON-serializable)
def save_memory():
    memory_data = {
        "Leslie": _serialize_agent_memory(leslie),
        "Ashley": _serialize_agent_memory(ashley),
    }
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=4, ensure_ascii=False)

def _restore_agent_memory(agent, items):
    # Push items back in a safe way using agent.think if available
    if not items:
        return
    for item in items:
        text = None
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("message") or str(item)
        else:
            text = str(item)
        if hasattr(agent, "think") and callable(getattr(agent, "think")):
            try:
                agent.think(text)
            except Exception:
                pass


# Function to load agent memory
def load_memory():
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
        _restore_agent_memory(leslie, memory_data.get("Leslie"))
        _restore_agent_memory(ashley, memory_data.get("Ashley"))

# Execute stock chart analysis
def analyze_stock_chart(symbol):
    print("\n=== Capturing TradingView Chart ===")
    chart_path = capture_tradingview_chart(symbol)
    if not chart_path:
        return "Chart capture failed."
    
    print("\n=== Agents Reading Image ===")
    # Instead of reading the document directly, we'll have the agents think about the chart
    leslie.think(f"Analyzing the chart for {symbol} at {chart_path}")
    ashley.think(f"Examining the technical patterns in {symbol}'s chart at {chart_path}")
    
    print("\n=== AI Analysis ===")
    ai_analysis = analyze_chart_with_gpt4o(chart_path)
    print(f"AI Analysis: {ai_analysis}")
    
    print("\n=== Agents Responding ===")
    leslie_response = get_gpt_response(f"Leslie, provide an expert opinion on the AI's analysis: {ai_analysis}")
    ashley_response = get_gpt_response(f"Ashley, respond to Leslie's thoughts: {leslie_response}")
    
    # Have agents process the responses through thinking
    leslie.think(leslie_response)
    ashley.think(ashley_response)
    
    # Run a few rounds of interaction
    environment.run(4)
    
    save_memory()
    return ai_analysis

# Load memory before running
environment.run(2)
load_memory()

if __name__ == "__main__":
    # Run for a sample stock
    analyze_stock_chart("SPY")
