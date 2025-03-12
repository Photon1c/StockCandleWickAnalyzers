# Stock Candle Wick Analyzers

# Update 3.12.25 - Quick Reference Table (Created By GPT-4o)

| Feature          | Meaning                       | Market Sentiment            |
|------------------|-------------------------------|-----------------------------|
| Hollow Candle    | Close > Open                  | Bullish (Buying Pressure)   |
| Filled Candle    | Close < Open                  | Bearish (Selling Pressure)  |
| Long Body        | Strong Momentum               | Confident (Bullish/Bearish) |
| Short Body       | Weak Momentum                 | Indecision, Consolidation   |
| Long Upper Wick  | Rejected Higher Prices        | Resistance, Bearish Signal  |
| Long Lower Wick  | Rejected Lower Prices         | Support, Bullish Signal     |
| Short or No Wick | Price stability within period | Controlled or Consolidated  |


# Update 2.20.25 - Imagekit Vision Report Generator Ready 👓💹🎊

The [Imagekit Vision Analysis](scripts/Imagekit-Vision-Analysis-version-hourglass.py) script is now available. Here is some sample [output.](outputs/Imagekit-Vision-version-hourglass-sample-output.txt)

# Wick Machine Vision Analysis 🚧 - Check back for new scripts!

This repository is a collection of minimal scripts that perform the following intended steps:

1)Create a stock candle chart of a defined stock.  
2)Upload this chart to imagekit.io.  
3)Using LLM vision models, analyze the uploaded images-- this repository uses OpenAI's GPT-4o.  

The fatman script uses LLM chat completion, while the imagekit test is running into write permission issues that will be updated as time permits.
> [!NOTE]
>As of 2.21.25 this issue has been resolved, but the imagekit test script will still be archived for further studying.  

Check back later for optimized scripts, since this project is still in the debugging phase :construction: :building_construction: :construction_worker:   

Currently in the task pipeline:  

- [x] Successfully upload image to imagekit.io and analyze with LLM. ✔️
- [x] Generate sample output text files. ✔️
- [ ] Integrate image analysis with streamlined list iterator and report builder. 🗞️📰⏳
- [ ] Create scheduler and email agent to further automate reports. 🕐⏳
- [ ] Create persistent monitor listener script to test commentary agent. 👂⌛

> [!NOTE]
> The results obtained from these scripts are not financial advice.  

| Requirements  | Purpose |
| ------------- | ------------- |
| OpenAI API Key  | Processing the image query  |
| Imagekit.io API Key  | Storing the image for retrieval  |  

![Flow Diagram](media/aistockvision.png)
![Cover Image](media/coverimage.png)

