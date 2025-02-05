# Analyze uploaded stock candle-wick chart
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

client = OpenAI()

analysis_report = []

def generate_ai_analysis(data_summary):
    """Generate detailed analysis using OpenAI API."""
    prompt = f"""
    Analyze the following stock data summary and provide insights on its price action:
    {data_summary}

    Include observations about:
    - Price volatility
    - Trends in upper and lower wicks
    - What the wick scores indicate about market behavior
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with expertise in stock market analysis."},
                {"role": "user", "content": prompt}
            ]
        )
    
        # Access the message content correctly
        ai_analysis = response.choices[0].message.content.strip()
        return ai_analysis
    except Exception as e:
        return f"Error generating analysis: {e}"



def wick_analysis(ticker, start_date, end_date):
    # Download historical data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if data.empty:
        return "No data available for the given date range."

    # Ensure all required columns exist and flatten the column index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)  # Flatten the MultiIndex to single level

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        return "The required columns are missing from the data."

    # Drop rows with NaN or non-numeric values and convert to float
    data = data[required_columns].dropna().astype(float)

    # Ensure the index is a datetime object
    data.index = pd.to_datetime(data.index)

    # Calculate wick sizes and body sizes
    high_price = data['High']
    low_price = data['Low']
    open_price = data['Open']
    close_price = data['Close']

    data['Upper_Wick'] = high_price - close_price.where(close_price > open_price, open_price)
    data['Lower_Wick'] = close_price.where(close_price < open_price, open_price) - low_price
    data['Body_Size'] = (close_price - open_price).abs()
    data['Total_Candle'] = high_price - low_price
    data['Wick_Score'] = data['Upper_Wick'] + data['Lower_Wick']

    # Summary statistics
    upper_wick_avg = data['Upper_Wick'].mean()
    lower_wick_avg = data['Lower_Wick'].mean()
    body_size_avg = data['Body_Size'].mean()
    total_candle_avg = data['Total_Candle'].mean()
    wick_score_avg = data['Wick_Score'].mean()

    # Prepare summary for AI analysis
    data_summary = f"""
    Average Upper Wick: {upper_wick_avg:.2f}
    Average Lower Wick: {lower_wick_avg:.2f}
    Average Body Size: {body_size_avg:.2f}
    Average Total Candle: {total_candle_avg:.2f}
    Average Wick Score: {wick_score_avg:.2f}
    """

    # Generate AI analysis
    ai_analysis = generate_ai_analysis(data_summary)

    # Generate a report
    report = f"""
    Wick Analysis Report for {ticker} ({start_date} to {end_date})
    --------------------------------------------------------------
    {data_summary}

    AI-Generated Insights:
    {ai_analysis}
    """

    # Save detailed table to a CSV file
    output_filename = f"{ticker}_wick_analysis.csv"
    data[['Upper_Wick', 'Lower_Wick', 'Body_Size', 'Total_Candle', 'Wick_Score']].to_csv(output_filename)
    report += f"\nA detailed wick analysis table has been saved as '{output_filename}'."
    plot_candlestick_chart(ticker, data)
    # Create a candlestick chart
    chart_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    plt.figure(figsize=(10, 6))
    try:
        mpf.plot(
            chart_data,
            type="candle",
            title=f"{ticker} Candlestick Chart ({start_date} to {end_date})",
            ylabel="Price",
            volume=True,
            style="yahoo"
        )
    except ValueError as e:
        print(f"Error during plotting: {e}")
        return "Chart plotting failed. Please check the cleaned data."

    chart_filename = f"{ticker}_candlestick_chart.jpg"
    #plt.savefig(chart_filename)
    plt.show()

    report += f"\nThe candlestick chart has been saved as '{chart_filename}'."

    return report


def plot_candlestick_chart(ticker, data):
    """Generate and display a Plotly candlestick chart."""
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',  # Green for bullish candles
        decreasing_line_color='red',   # Red for bearish candles
        increasing_fillcolor='rgba(0, 255, 0, 0.5)',  # Semi-transparent green for bullish
        decreasing_fillcolor='rgba(255, 0, 0, 0.5)'   # Semi-transparent red for bearish
    ))

    # Set title and labels
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False  # Disable range slider for clarity
    )
    
    fig.show()


ticker = input("Enter the stock ticker (e.g., INTU): ").strip().upper()
start_date = "2024-11-15" #input("Enter the start date (YYYY-MM-DD): ").strip()
end_date = "2025-01-23"#input("Enter the end date (YYYY-MM-DD): ").strip()

report = wick_analysis(ticker, start_date, end_date)
print(report)
# Check /output directory later for sample outputs :)
