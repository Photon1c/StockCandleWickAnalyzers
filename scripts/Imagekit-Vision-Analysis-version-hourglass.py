#Define a stock ticker, take a TradingView screenshot, upload it to Imagekit.io, perform AI analysis and generate a report.
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from imagekitio import ImageKit
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

client = OpenAI()
screenshot_dir = r"F:\StillWater\Apps\Pecuniary\smartinvestor\tools\FalconEye\screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# ImageKit Credentials
imagekit = ImageKit(
    private_key=os.getenv("IMAGEKIT_PRIVATE_KEY"),
    public_key=os.getenv("IMAGEKIT_PUBLIC_KEY"),
    url_endpoint=os.getenv("IMAGEKIT_URL_ENDPOINT")
)

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
                    {"type": "text", "text": "Analyze the stock chart in this image and provide insights on trends, volatility, and possible price action."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating analysis: {e}"

# **Full Execution Function**
def analyze_stock_chart(ticker):
    """Runs the complete process: Capture TradingView chart, upload image, and get AI analysis."""
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

    if os.path.exists(chart_path):
        os.remove(chart_path)
        print(f"Deleted local image: {chart_path}")

    print("\n=== AI-Generated TradingView Chart Analysis ===\n")
    print(ai_report)

    return ai_report

# **Run for a sample stock**
analyze_stock_chart("PLTR")
