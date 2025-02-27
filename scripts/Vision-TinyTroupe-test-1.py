#Pileine image analyzis from OpenAI to swarms to TinyTroupe, more debugging is pending.
import json
import sys
import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe.examples import *
from tinytroupe.factory import TinyPersonFactory
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
sys.path.append('..')
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from imagekitio import ImageKit

load_dotenv()

symbol = "NVDA"

client = OpenAI()

screenshot_dir = r"F:\StillWater\Apps\Pecuniary\smartinvestor\tools\FalconEye\screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# ImageKit Credentials
imagekit = ImageKit(
    private_key=os.getenv("IMAGEKIT_PRIVATE_KEY"),
    public_key=os.getenv("IMAGEKIT_PUBLIC_KEY"),
    url_endpoint=os.getenv("IMAGEKIT_URL_ENDPOINT")
)

# Function to Upload Image to ImageKit
def upload_image_to_imagekit(image_path):
    """Uploads an image to ImageKit and returns the HTTPS URL."""
    try:
        if not os.path.exists(image_path):
            return "Error: Image file not found."

        print(f"Uploading {image_path} to ImageKit...")

        with open(image_path, "rb") as img_file:
            upload = imagekit.upload(
                file=img_file,
                file_name=os.path.basename(image_path)
            )

        print("Upload response:", upload.response_metadata.raw)
        return upload.url
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None
        
# AI Analysis Function
def generate_ai_analysis_from_image(image_url):
    """Uses OpenAI's vision model to analyze a TradingView chart image via HTTPS URL."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a stock market expert analyzing candlestick charts."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze the stock chart in this image and provide insights on the perceived directional momentum sentiment, trends, volatility, and possible price action in a neat table report. Limit your response to 450 tokens and indicate the perceived sentiment."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating analysis: {e}"        

# Function to Capture TradingView Chart
def capture_tradingview_chart(symbol):
    """Captures and converts a TradingView chart screenshot to a proper PNG."""
    try:
        print("Initializing WebDriver...")
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        tradingview_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
        print(f"Opening TradingView URL: {tradingview_url}")
        driver.get(tradingview_url)
        time.sleep(5)

        raw_screenshot_path = os.path.join(screenshot_dir, f"{symbol}_chart_raw.png")
        proper_screenshot_path = os.path.join(screenshot_dir, f"{symbol}_chart.png")

        # Take a screenshot (raw format)
        print("Taking screenshot...")
        driver.save_screenshot(raw_screenshot_path)
        driver.quit()

        # Convert screenshot to proper PNG format
        print("Converting screenshot to a valid PNG format...")
        img = Image.open(raw_screenshot_path)
        img = img.convert("RGB")  # Ensures proper color format
        img.save(proper_screenshot_path, "PNG")

        # Remove the raw screenshot
        os.remove(raw_screenshot_path)

        if os.path.exists(proper_screenshot_path):
            print(f"Chart successfully saved: {proper_screenshot_path}")
            return proper_screenshot_path
        else:
            print("Screenshot failed, file not found!")
            return None

    except Exception as e:
        print(f"Error capturing TradingView chart: {e}")
        return None

def translate_openai_response(openai_response):
    """
    Translates the verbose OpenAI response into a concise string for TinyTroupe.
    
    Args:
        openai_response: The raw response from OpenAI's API.
        
    Returns:
        str: A simplified string containing key insights.
    """
    try:
        # Check if the response is already a string
        if isinstance(openai_response, str):
            return openai_response
        
        # Extract the content from the OpenAI response
        if hasattr(openai_response, 'choices') and len(openai_response.choices) > 0:
            message = openai_response.choices[0].message
            if hasattr(message, 'content'):
                return message.content.strip()
        
        # Fallback: Return the raw response as a string
        return str(openai_response)
    except Exception as e:
        print(f"Error translating OpenAI response: {e}")
        return "Error: Unable to translate response."

# Simulate OpenAI API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/Photon1c/PecuniaryScriptsLibrary/raw/main/inputs/logo.png?raw=true",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

# Translate the OpenAI response
translated_string = translate_openai_response(response)




def send_to_tinytroupe(translated_string):
    """
    Sends the translated string to TinyTroupe agents for commentary.
    
    Args:
        translated_string (str): The simplified string from the translator.
        
    Returns:
        str: Commentary generated by TinyTroupe agents.
    """
    try:
        # Simulate TinyTroupe processing logic
        # Replace this with actual TinyTroupe API calls
        commentary = f"TinyTroupe Commentary: {translated_string}"
        return commentary
    except Exception as e:
        print(f"Error sending to TinyTroupe: {e}")
        return "Error: Failed to generate commentary."



def full_workflow(ticker):
    """
    Runs the complete workflow: Capture chart, analyze, translate, and send to TinyTroupe.
    
    Args:
        ticker (str): The stock ticker symbol.
        
    Returns:
        str: Final commentary from TinyTroupe agents.
    """
    print("\n=== Capturing TradingView Chart ===")
    chart_path = capture_tradingview_chart(ticker)
    if not chart_path:
        print("Error: TradingView chart capture failed.")
        return
    
    print("\nUploading image to ImageKit...")
    image_url = upload_image_to_imagekit(chart_path)
    if not image_url or "Error" in image_url:
        print(f"Image upload failed: {image_url}")
        return
    
    print(f"Image uploaded successfully: {image_url}")
    print("\nRunning AI analysis on the uploaded chart...")
    ai_report = generate_ai_analysis_from_image(image_url)
    
    # Translate the OpenAI response
    print("\nTranslating OpenAI response...")
    translated_string = translate_openai_response(ai_report)
    print(f"Translated String: {translated_string}")
    
    # Send the translated string to TinyTroupe
    print("\nSending to TinyTroupe for commentary...")
    tinytroupe_commentary = send_to_tinytroupe(translated_string)
    print(f"\n=== TinyTroupe Commentary ===\n{tinytroupe_commentary}")
    
    # Clean up local files
    if os.path.exists(chart_path):
        os.remove(chart_path)
        print(f"Deleted local image: {chart_path}")
    
    return tinytroupe_commentary

# Run the workflow for a sample stock
full_workflow(symbol)
