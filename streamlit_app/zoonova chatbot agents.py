import pandas as pd
import streamlit as st
from streamlit_chat import message
from io import StringIO
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from itertools import islice
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain_core.tools import tool
from langchain.tools import Tool
import os
from dotenv import load_dotenv
import time

load_dotenv()

# azure_api_key = os.getenv("AZURE_API_KEY")
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# azure_deployment = os.getenv("DEPLOYMENT_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
df = pd.read_csv("ChatGPTData2.csv", memory_map=True)
chat_model1 = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Setting page title and header
st.set_page_config(page_title="Zoonova chatbot", page_icon=":robot_face:")
st.markdown(
    "<h1 style='text-align: center;'>Zoonova chatbot</h1>", unsafe_allow_html=True
)


# Initialise session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if "model" not in st.session_state:
    st.session_state["model"] = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history")


# Sidebar - let user clear the current conversation
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state["number_tokens"] = []

    st.session_state["total_tokens"] = []
    st.session_state["model"] = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history")


def as_list(arr: list[str]):
    for key in arr:
        if key not in Data_dict:
            raise Exception(f'"{key}" is not a valid column!')
    return arr


def to_rank(arr: list[str]):
    temp: list[str] = []
    for s in arr:
        key = s + " Rank"
        if key in Data_dict:
            temp.append(key)
    return temp


Data_dict = [
    "10d Alpha",
    "10d Delta",
    "10d Price",
    "1d Alpha",
    "1d Delta",
    "1d Price",
    "1y Alpha",
    "1y Delta",
    "1y Price",
    "2m Alpha",
    "2m Delta",
    "2m Price",
    "2w Alpha",
    "2w Delta",
    "2w Price",
    "3d Alpha",
    "3d Delta",
    "3d Price",
    "3m Alpha",
    "3m Delta",
    "3m Price",
    "3w Alpha",
    "3w Delta",
    "3w Price",
    "5d Alpha",
    "5d Delta",
    "5d Price",
    "6m Alpha",
    "6m Delta",
    "6m Price",
    "6w Alpha",
    "6w Delta",
    "6w Price",
    "9m Alpha",
    "9m Delta",
    "9m Price",
    "Accumulation/Distribution Index",
    "Accumulation/Distribution Index Rank",
    "Alpha Long (>7w)",
    "Alpha Long (>7w) Rank",
    "Alpha Mid (3w - 6w)",
    "Alpha Mid (3w - 6w) Rank",
    "Alpha Short (<2w)",
    "Alpha Short (<2w) Rank",
    "Aroon Down",
    "Aroon Down Rank",
    "Aroon Indicator",
    "Aroon Indicator Rank",
    "Aroon Up",
    "Aroon Up Rank",
    "Average Directional Movement Index",
    "Average Directional Movement Index Negative",
    "Average Directional Movement Index Negative Rank",
    "Average Directional Movement Index Positive",
    "Average Directional Movement Index Positive Rank",
    "Average Directional Movement Index Rank",
    "Average True Rng",
    "Average True Rng Rank",
    "Awesome Oscillator",
    "Awesome Oscillator Rank",
    "Bearish Engulfing",
    "Bearish Harami",
    "Bearish Long-Term (1w) Trend Anomaly",
    "Bearish Short-Term (3d) Trend Anomaly",
    "Blackrock Index Alpha",
    "Blackrock Index Alpha Rank",
    "Bollinger Bands Lower",
    "Bollinger Bands Lower Indicator",
    "Bollinger Bands Lower Rank",
    "Bollinger Bands Middle",
    "Bollinger Bands Middle Rank",
    "Bollinger Bands Price",
    "Bollinger Bands Price Rank",
    "Bollinger Bands Upper",
    "Bollinger Bands Upper Indicator",
    "Bollinger Bands Upper Rank",
    "Bollinger Bands Width",
    "Bollinger Bands Width Rank",
    "Bond Market Index Alpha",
    "Bond Market Index Alpha Rank",
    "Bullish Engulfing",
    "Bullish Harami",
    "Bullish Long-Term (1w) Trend Anomaly",
    "Bullish Short-Term (3d) Trend Anomaly",
    "Chaikin Money Flow",
    "Chaikin Money Flow Rank",
    "Chaikin Oscillator",
    "Chaikin Oscillator Rank",
    "Chande Momentum Oscillator",
    "Chande Momentum Oscillator Rank",
    "Close Date",
    "Commodity Channel Index",
    "Commodity Channel Index Rank",
    "Company",
    "Convergence/Divergence Moving Average",
    "Convergence/Divergence Moving Average Rank",
    "Dark Cloud Cover",
    "Detr Price Oscillator",
    "Detr Price Oscillator Rank",
    "Detrended Price Oscillator",
    "Detrended Price Oscillator Rank",
    "Doji",
    "Doji Star",
    "Donchain Channel Lower",
    "Donchain Channel Lower Rank",
    "Donchain Channel Middle",
    "Donchain Channel Middle Rank",
    "Donchain Channel Price",
    "Donchain Channel Price Rank",
    "Donchain Channel Upper",
    "Donchain Channel Upper Rank",
    "Donchain Channel Width",
    "Donchain Channel Width Rank",
    "Double Exponential Moving Average",
    "Double Exponential Moving Average Rank",
    "Dow Jones Index Alpha",
    "Dow Jones Index Alpha Rank",
    "Dragonfly Doji",
    "Em",
    "Em Rank",
    "Energy Index",
    "Energy Index Alpha",
    "Energy Index Alpha Rank",
    "Energy Index Rank",
    "Envelopes Lower",
    "Envelopes Lower Rank",
    "Envelopes Upper",
    "Envelopes Upper Rank",
    "Exp Moving Average Fast",
    "Exp Moving Average Fast Rank",
    "Exp Moving Average Slow",
    "Exp Moving Average Slow Rank",
    "Exponential Moving Average",
    "Exponential Moving Average Rank",
    "Fibonacci Retracement (rl=0.0)",
    "Fibonacci Retracement (rl=0.0) Rank",
    "Fibonacci Retracement (rl=100.0)",
    "Fibonacci Retracement (rl=100.0) Rank",
    "Fibonacci Retracement (rl=38.2)",
    "Fibonacci Retracement (rl=38.2) Rank",
    "Fibonacci Retracement (rl=50.0)",
    "Fibonacci Retracement (rl=50.0) Rank",
    "Fibonacci Retracement (rl=61.8)",
    "Fibonacci Retracement (rl=61.8) Rank",
    "Force Index",
    "Force Index Rank",
    "Forecast Oscillator",
    "Forecast Oscillator Rank",
    "Gravestone Doji",
    "Hammer",
    "Hanging Man",
    "Hull Moving Average",
    "Hull Moving Average Rank",
    "Ichimoku A",
    "Ichimoku A Rank",
    "Ichimoku B",
    "Ichimoku B Rank",
    "Ichimoku Base",
    "Ichimoku Base Rank",
    "Ichimoku Conv",
    "Ichimoku Conv Rank",
    "Intraday Movement Index",
    "Intraday Movement Index Rank",
    "Inverted Hammer",
    "Kaufman Adp Moving Average",
    "Kaufman Adp Moving Average Rank",
    "Keltner Channel Lower",
    "Keltner Channel Lower Indicator",
    "Keltner Channel Lower Rank",
    "Keltner Channel Middle",
    "Keltner Channel Middle Rank",
    "Keltner Channel Price",
    "Keltner Channel Price Rank",
    "Keltner Channel Upper",
    "Keltner Channel Upper Indicator",
    "Keltner Channel Upper Rank",
    "Keltner Channel Width",
    "Keltner Channel Width Rank",
    "Klinger Oscillator",
    "Klinger Oscillator Rank",
    "Know Sure Thing",
    "Know Sure Thing Difference",
    "Know Sure Thing Difference Rank",
    "Know Sure Thing Rank",
    "Know Sure Thing Signal",
    "Know Sure Thing Signal Rank",
    "Linear Regression Indicator",
    "Linear Regression Indicator Rank",
    "Linear Regression Slope",
    "Linear Regression Slope Rank",
    "Market Facilitation Index",
    "Market Facilitation Index Rank",
    "Mass Index",
    "Mass Index Rank",
    "Median Price",
    "Median Price Rank",
    "Momentum",
    "Momentum Rank",
    "Money Flow Index",
    "Money Flow Index Rank",
    "Morning Star",
    "Morning Star Doji",
    "Moving Average Convergence/Divergence",
    "Moving Average Convergence/Divergence Difference",
    "Moving Average Convergence/Divergence Difference Rank",
    "Moving Average Convergence/Divergence Rank",
    "Moving Average Convergence/Divergence Signal",
    "Moving Average Convergence/Divergence Signal Rank",
    "NASDAQ Index Alpha",
    "NASDAQ Index Alpha Rank",
    "Negative Volume Index",
    "Negative Volume Index Rank",
    "On-Balance Volume",
    "On-Balance Volume Rank",
    "Parabolic Stop & Reverse Down",
    "Parabolic Stop & Reverse Down Indicator",
    "Parabolic Stop & Reverse Down Rank",
    "Parabolic Stop & Reverse Up",
    "Parabolic Stop & Reverse Up Indicator",
    "Parabolic Stop & Reverse Up Rank",
    "Percent Price Oscillator",
    "Percent Price Oscillator Historical",
    "Percent Price Oscillator Historical Rank",
    "Percent Price Oscillator Rank",
    "Percent Price Oscillator Signal",
    "Percent Price Oscillator Signal Rank",
    "Performance",
    "Performance Rank",
    "Piercing Pattern",
    "Positive Volatility Index",
    "Positive Volatility Index Rank",
    "Previous Close",
    "Price Change Anomaly",
    "Price Channel Lower",
    "Price Channel Lower Rank",
    "Price Channel Upper",
    "Price Channel Upper Rank",
    "Price Channels Lower",
    "Price Channels Lower Rank",
    "Price Channels Upper",
    "Price Channels Upper Rank",
    "Price Level Shift Anomaly",
    "Price Oscillator",
    "Price Oscillator Rank",
    "Price Trend",
    "Price Trend Rank",
    "Probability of 1m Alpha + 3%",
    "Probability of 1w Alpha + 3%",
    "Probability of 1y Alpha + 10%",
    "Probability of 3m Alpha + 3%",
    "Probability of 6m Alpha + 10%",
    "Probability of 9m Alpha + 10%",
    "Projection Bands Lower",
    "Projection Bands Lower Rank",
    "Projection Bands Upper",
    "Projection Bands Upper Rank",
    "Projection Oscillator",
    "Projection Oscillator Rank",
    "Projection Oscillator Trigger",
    "Projection Oscillator Trigger Rank",
    "QStick",
    "QStick Rank",
    "Rain Drop",
    "Rain Drop Doji",
    "Range Indicator",
    "Range Indicator Rank",
    "Rate of Change",
    "Rate of Change Rank",
    "Relative Strength Index",
    "Relative Strength Index Rank",
    "Relative Volatility Index",
    "Relative Volatility Index Rank",
    "S&P 500 Index Alpha",
    "S&P 500 Index Alpha Rank",
    "Schaff Trn Cyc",
    "Schaff Trn Cyc Rank",
    "Sentiment",
    "Shooting Star",
    "Simple Moving Average Em",
    "Simple Moving Average Em Rank",
    "Simple Moving Average Fast",
    "Simple Moving Average Fast Rank",
    "Simple Moving Average Slow",
    "Simple Moving Average Slow Rank",
    "Smoothed Moving Average",
    "Smoothed Moving Average Rank",
    "Standard Deviation",
    "Standard Deviation Rank",
    "Star",
    "Stochastic",
    "Stochastic Momentum Index",
    "Stochastic Momentum Index Rank",
    "Stochastic Oscillator Fast D",
    "Stochastic Oscillator Fast D Rank",
    "Stochastic Oscillator Fast K",
    "Stochastic Oscillator Fast K Rank",
    "Stochastic Oscillator Slow D",
    "Stochastic Oscillator Slow D Rank",
    "Stochastic Oscillator Slow K",
    "Stochastic Oscillator Slow K Rank",
    "Stochastic Rank",
    "Stochastic Relative Strength Index",
    "Stochastic Relative Strength Index D",
    "Stochastic Relative Strength Index D Rank",
    "Stochastic Relative Strength Index K",
    "Stochastic Relative Strength Index K Rank",
    "Stochastic Relative Strength Index Rank",
    "Stochastic Signal",
    "Stochastic Signal Rank",
    "Stock",
    "Stock Market Index Alpha",
    "Stock Market Index Alpha Rank",
    "Swing Index",
    "Swing Index Rank",
    "Time Series Forecast",
    "Time Series Forecast Rank",
    "Time Series Moving Average",
    "Time Series Moving Average Rank",
    "Tiple Exp Average",
    "Tiple Exp Average Rank",
    "Triangular Moving Average",
    "Triangular Moving Average Rank",
    "Triple Exponential Moving Average",
    "Triple Exponential Moving Average Rank",
    "True Strength Index",
    "True Strength Index Rank",
    "Typical Price",
    "Typical Price Rank",
    "Ulcer Index",
    "Ulcer Index Rank",
    "Ultimate Oscillator",
    "Ultimate Oscillator Rank",
    "Variable Moving Average",
    "Variable Moving Average Rank",
    "Vertical Horizontal Filter",
    "Vertical Horizontal Filter Rank",
    "Vis Ichimoku A",
    "Vis Ichimoku A Rank",
    "Vis Ichimoku B",
    "Vis Ichimoku B Rank",
    "Volatility",
    "Volatility Chaikins",
    "Volatility Chaikins Rank",
    "Volume Oscillator",
    "Volume Oscillator Rank",
    "Volume Rate of Change",
    "Volume Rate of Change Rank",
    "Vortex Difference",
    "Vortex Difference Rank",
    "Vortex Negative",
    "Vortex Negative Rank",
    "Vortex Positive",
    "Vortex Positive Rank",
    "Weighted Average Price",
    "Weighted Average Price Rank",
    "Weighted Close",
    "Weighted Close Rank",
    "Weighted Moving Average",
    "Weighted Moving Average Rank",
    "Wilders Smoothing",
    "Wilders Smoothing Rank",
    "Williams Accumulation Distribution",
    "Williams Accumulation Distribution Rank",
    "William's R",
    "William's R Rank",
]
Data_dict_indicators = as_list(
    [
        "Sentiment",
        "Momentum",
        "Price Oscillator",
        "Relative Strength Index",
        "William's R",
        "Price Channels Lower",
        "Price Channels Upper",
        "Convergence/Divergence Moving Average",
        "Double Exponential Moving Average",
        "Exponential Moving Average",
        "Hull Moving Average",
        "Smoothed Moving Average",
        "Weighted Moving Average",
        "Awesome Oscillator",
        "Kaufman Adp Moving Average",
        "Percent Price Oscillator",
        "Percent Price Oscillator Historical",
        "Percent Price Oscillator Signal",
        "Rate of Change",
        "Stochastic",
        "Stochastic Relative Strength Index",
        "Stochastic Relative Strength Index D",
        "Stochastic Relative Strength Index K",
        "Stochastic Signal",
        "True Strength Index",
        "Ultimate Oscillator",
        "Average Directional Movement Index",
        "Average Directional Movement Index Negative",
        "Average Directional Movement Index Positive",
        "Aroon Down",
        "Aroon Indicator",
        "Aroon Up",
        "Commodity Channel Index",
        "Detr Price Oscillator",
        "Exp Moving Average Fast",
        "Exp Moving Average Slow",
        "Ichimoku A",
        "Ichimoku B",
        "Ichimoku Base",
        "Ichimoku Conv",
        "Know Sure Thing",
        "Know Sure Thing Difference",
        "Know Sure Thing Signal",
        "Moving Average Convergence/Divergence",
        "Moving Average Convergence/Divergence Difference",
        "Moving Average Convergence/Divergence Signal",
        "Mass Index",
        "Parabolic Stop & Reverse Down",
        "Parabolic Stop & Reverse Down Indicator",
        "Parabolic Stop & Reverse Up",
        "Parabolic Stop & Reverse Up Indicator",
        "Simple Moving Average Fast",
        "Simple Moving Average Slow",
        "Schaff Trn Cyc",
        "Tiple Exp Average",
        "Vis Ichimoku A",
        "Vis Ichimoku B",
        "Vortex Difference",
        "Vortex Negative",
        "Vortex Positive",
        "Average True Rng",
        "Bollinger Bands Upper",
        "Bollinger Bands Upper Indicator",
        "Bollinger Bands Lower",
        "Bollinger Bands Lower Indicator",
        "Bollinger Bands Middle",
        "Bollinger Bands Price",
        "Bollinger Bands Width",
        "Donchain Channel Upper",
        "Donchain Channel Lower",
        "Donchain Channel Middle",
        "Donchain Channel Price",
        "Donchain Channel Width",
        "Keltner Channel Middle",
        "Keltner Channel Upper",
        "Keltner Channel Upper Indicator",
        "Keltner Channel Lower",
        "Keltner Channel Lower Indicator",
        "Keltner Channel Price",
        "Keltner Channel Width",
        "Ulcer Index",
        "Accumulation/Distribution Index",
        "Chaikin Money Flow",
        "Em",
        "Force Index",
        "Money Flow Index",
        "Negative Volume Index",
        "On-Balance Volume",
        "Simple Moving Average Em",
        "Price Trend",
        "Weighted Average Price",
        "Chaikin Oscillator",
        "Chande Momentum Oscillator",
        "Price Channel Lower",
        "Price Channel Upper",
        "Detrended Price Oscillator",
        "Envelopes Lower",
        "Envelopes Upper",
        "Stochastic Oscillator Fast D",
        "Stochastic Oscillator Fast K",
        "Fibonacci Retracement (rl=0.0)",
        "Fibonacci Retracement (rl=38.2)",
        "Fibonacci Retracement (rl=50.0)",
        "Fibonacci Retracement (rl=61.8)",
        "Fibonacci Retracement (rl=100.0)",
        "Forecast Oscillator",
        "Intraday Movement Index",
        "Klinger Oscillator",
        "Linear Regression Indicator",
        "Linear Regression Slope",
        "Market Facilitation Index",
        "Median Price",
        "Performance",
        "Positive Volatility Index",
        "Projection Oscillator",
        "Projection Oscillator Trigger",
        "Projection Bands Lower",
        "Projection Bands Upper",
        "QStick",
        "Range Indicator",
        "Relative Volatility Index",
        "Stochastic Oscillator Slow D",
        "Stochastic Oscillator Slow K",
        "Stochastic Momentum Index",
        "Standard Deviation",
        "Swing Index",
        "Triangular Moving Average",
        "Triple Exponential Moving Average",
        "Time Series Forecast",
        "Time Series Moving Average",
        "Typical Price",
        "Variable Moving Average",
        "Vertical Horizontal Filter",
        "Volatility Chaikins",
        "Volume Oscillator",
        "Volume Rate of Change",
        "Weighted Close",
        "Wilders Smoothing",
        "Williams Accumulation Distribution",
    ]
)
Data_dict_indicators_rank = to_rank(Data_dict_indicators)
Data_dict_Prediction = as_list(
    [
        "1d Price",
        "1d Delta",
        "1d Alpha",
        "3d Price",
        "3d Delta",
        "3d Alpha",
        "5d Price",
        "5d Delta",
        "5d Alpha",
        "10d Price",
        "10d Delta",
        "10d Alpha",
        "2w Price",
        "2w Delta",
        "2w Alpha",
        "3w Price",
        "3w Delta",
        "3w Alpha",
        "6w Price",
        "6w Delta",
        "6w Alpha",
        "2m Price",
        "2m Delta",
        "2m Alpha",
        "3m Price",
        "3m Delta",
        "3m Alpha",
        "6m Price",
        "6m Delta",
        "6m Alpha",
        "9m Price",
        "9m Delta",
        "9m Alpha",
        "1y Price",
        "1y Delta",
        "1y Alpha",
        "Probability of 1w Alpha + 3%",
        "Probability of 1m Alpha + 3%",
        "Probability of 3m Alpha + 3%",
        "Probability of 6m Alpha + 10%",
        "Probability of 9m Alpha + 10%",
        "Probability of 1y Alpha + 10%",
    ]
)
Data_dict_Volatility_Cluster = as_list(
    ["Volatility", "Alpha Short (<2w)", "Alpha Mid (3w - 6w)", "Alpha Long (>7w)"]
)
Data_dict_Volatility_Cluster_Rank = to_rank(Data_dict_Volatility_Cluster)
Data_dict_anomalies = as_list(
    [
        "Price Level Shift Anomaly",
        "Price Change Anomaly",
        "Bullish Short-Term (3d) Trend Anomaly",
        "Bearish Short-Term (3d) Trend Anomaly",
        "Bullish Long-Term (1w) Trend Anomaly",
        "Bearish Long-Term (1w) Trend Anomaly",
    ]
)
Data_dict_Index_Cluster = as_list(
    [
        "Blackrock Index Alpha",
        "Bond Market Index Alpha",
        "Dow Jones Index Alpha",
        "NASDAQ Index Alpha",
        "S&P 500 Index Alpha",
        "Stock Market Index Alpha",
        "Energy Index Alpha",
        "Energy Index",
    ]
)
Data_dict_Index_Cluster_Rank = to_rank(Data_dict_Index_Cluster)
Data_dict_Candlestick = as_list(
    [
        "Bearish Engulfing",
        "Bearish Harami",
        "Bullish Engulfing",
        "Bullish Harami",
        "Dark Cloud Cover",
        "Doji",
        "Dragonfly Doji",
        "Gravestone Doji",
        "Hammer",
        "Inverted Hammer",
        "Hanging Man",
        "Morning Star",
        "Morning Star Doji",
        "Piercing Pattern",
        "Rain Drop",
        "Rain Drop Doji",
        "Shooting Star",
        "Star",
        "Doji Star",
    ]
)
Data_dict_probabilities = as_list(
    [
        "Probability of 1w Alpha + 3%",
        "Probability of 1m Alpha + 3%",
        "Probability of 3m Alpha + 3%",
        "Probability of 6m Alpha + 10%",
        "Probability of 9m Alpha + 10%",
        "Probability of 1y Alpha + 10%",
    ]
)
GPT_PROMPT = 'You are an expert in Python pandas DataFrame ("DF") and DF queries. You will be creating a query of a DF from natural language text.'


def search_ddg(user_input):
    """Searches the internet using duckduckgo search for stocks information related to user's query"""
    max_links = 8
    results = []
    with DDGS() as ddgs:
        ddgs_gen = ddgs.text(user_input, max_results=max_links)
        for r in islice(ddgs_gen, max_links):
            results.append(r)
        titles = []
        bodies = []
        links = []
        for result in results:
            title = result["title"]
            body = result["body"]
            link = result["href"]
            if link.startswith("https://duckduckgo.com"):
                continue  # Skip DuckDuckGo ads
            else:
                titles.append(title)
                bodies.append(body)
                links.append(link)
    return [titles, bodies, links]


# searching inside the data frame:
@tool
def search_df(user_input):
    """
    Converts natural language query into a pandas dataframe query. Only works well when the user query asks for any stock information.
    """
    INSTRUCTIONS = f"""Convert the given natural language text {user_input} into a query for a DataFrame (DF) containing columns as listed in {Data_dict}. Return ONLY the converted query with NO explanation or additional information and omit "df = query."
    Be precise with the query, ensuring it is accurate.
    The query should ONLY contain the dataframe query and must not contain line breaks or begin with 'python'.
    The query must include the columns 'Stock' and 'Company'.
    Do NOT use variable names for the query; instead, convert any variables to CSV format (e.g., ['val1', 'val2']).
    Only query DF columns listed in {Data_dict}.
    If one or more stocks are specified, use case-insensitive matching: First, try an exact match against symbols in 'Stock' and, if not found, next try a partial match against company names in 'Company'.
    If {user_input} contains "low," "medium," or "high" in lowercase, convert them to uppercase (e.g., LOW, MEDIUM, or HIGH).
    If the word "technical indicators" is in {user_input}, use all columns in {Data_dict_indicators} for the query.
    If {user_input} references any column in {Data_dict_indicators_rank}, use {Data_dict_indicators_rank} and {user_input} to replace the Column Name Rank and RANK in the following query and return it: df.loc[df['Column Name Rank'] == 'RANK', ['Stock', 'Company', 'Column Name Rank']].
    If {user_input} does not contain "cluster," skip to [END_IDX].
        If "cluster" does not follow "volatility" in {user_input}, skip to [END_VOL].
        If "rank" follows "cluster," use all columns in {Data_dict_Index_Cluster_Rank} for the query; otherwise, use all columns in {Data_dict_Index_Cluster} for the query.
    [END_VOL]
        If "cluster" does not follow "index" in {user_input}, skip to [END_IDX].
        If "cluster" always follows "alpha" in {user_input}, skip to [END_IDX].
        If "rank" follows "cluster," use all columns in {Data_dict_Index_Cluster_Rank} for the query; otherwise, use all columns in {Data_dict_Index_Cluster} for the query.
    [END_IDX]
    If {user_input} references "candlestick" or "candlestick pattern," include any referenced columns in {Data_dict_Candlestick} for the query; otherwise, use all columns in {Data_dict_Candlestick} for the query.
    If "price," "prediction," or "price prediction" is in {user_input}, use all columns in {Data_dict_Prediction} for the query.
    If {user_input} does not contain "anomaly" or "anomalies," skip to [END_ANOM].
        If {user_input} references any column in {Data_dict_anomalies}, include those columns for the query; otherwise, use all columns in {Data_dict_anomalies} for the query.
    [END_ANOM]
    If {user_input} references "all stocks," return all rows for the query, sorting all stocks and without using .head().
    If {user_input} specifies the number of rows to return, select only the required number of rows.
    If {user_input} contains "probability," refer to {Data_dict_probabilities} to convert {user_input} into a DF query and use the exact keyword from {Data_dict_probabilities}.
    If {user_input} includes "Rank" or "Probability," do not remove them from the DF query and refer to the provided dictionary.
    Always prioritize selecting the columns before applying any sorting.
    """

    # message = "I will use variables which are placed in braces, e.g., {var}. A string variable is assigned like {string_var}=\"string\"; a list variable is assigned like {list_var}=[\"val1\",\"val2\",...] and represents one or more column names in the source DF." + f"""
    message = f"""
        Data_dict="{Data_dict}"
        Data_dict_anomalies={Data_dict_anomalies}
        Data_dict_Candlestick={Data_dict_Candlestick}
        Data_dict_Index_Cluster={Data_dict_Index_Cluster}
        Data_dict_Index_Cluster_Rank={Data_dict_Index_Cluster_Rank}
        Data_dict_indicators={Data_dict_indicators}
        Data_dict_indicators_rank={Data_dict_indicators_rank}
        Data_dict_Prediction={Data_dict_Prediction}
        Data_dict_probabilities={Data_dict_probabilities}
        Data_dict_Volatility_Cluster={Data_dict_Volatility_Cluster}
        Data_dict_Volatility_Cluster_Rank={Data_dict_Volatility_Cluster_Rank}
        user_input="{{user_input}}"

        {INSTRUCTIONS}

        Human: {{user_input}}
        Chatbot:
        """
    prompt = PromptTemplate(input_variables=["user_input"], template=message)

    llm_chain = LLMChain(
        llm=st.session_state["model"],
        prompt=prompt,
        verbose=True,
        # memory=memory1,
    )
    try:
        response = llm_chain.predict(user_input=user_input)
        response = response.replace("DF", "df")
        response = eval(response)
    except Exception as e:
        print(e, "EXCEPTION")
        response = pd.DataFrame()

    print(response)
    # chat_model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    # # memory = ConversationBufferMemory()

    # messages = [
    #     SystemMessage(
    #         content= GPT_PROMPT
    #     ),
    #     HumanMessage(content=message),
    # ]

    # resp = chat_model.invoke(messages)
    # # resp = resp.content.split('\n')
    # resp = resp.content.replace('`', '')
    # response = eval(resp)
    # print(response, 'RESPONSE')

    st.session_state["messages"].append({"role": "assistant", "content": response})
    ddg_results = search_ddg(user_input)
    return response, ddg_results


@tool
def general_info(user_input):
    """Returns a response from llm for queries that are not directly related to stock data and donot require searching the entire dataframe."""

    # if not df:
    #     prompt = """Using the given user input and the resultant dataframe, give general information about the stock and the terms mentioned in the given user input.
    #     User Input: {user_input}
    #     Your answer should be a paragraph of not more than 10 sentences.
    #     If there are any technical jargon present in user's query, explain it and relate it to the stocks data.
    #     You should sound like a person giving out general information to the user and answering his query.
    #     Start the answer direclty without greetings or similar messages.
    #     Dont mention that you are AI.

    #     {chat_history}
    #     Human: {user_input}
    #     Chatbot:"""
    # else:

    # print('before')
    # df = str(df.head(10))
    # print('after')

    prompt = """Using the given user input and the resultant dataframe, give general information about the stock and the terms mentioned in the given user input.
    User Input: {user_input}
    Your answer should be a paragraph of not more than 10 sentences.
    If there are any technical jargon present in user's query, explain it and relate it to the stocks data.
    You should sound like a person giving out general information to the user and answering his query.
    Start the answer direclty without greetings or similar messages.
    Dont mention that you are AI.

    {chat_history}
    Human: {user_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"], template=prompt
    )

    llm_chain = LLMChain(
        llm=st.session_state["model"],
        prompt=prompt,
        verbose=True,
        # memory=st.session_state['memory'],
    )
    response = llm_chain.predict(
        chat_history=st.session_state["messages"][:4], user_input=user_input
    )
    print(response)
    # Return the response content

    st.session_state["messages"].append({"role": "assistant", "content": response})

    return response


# dataframe_tool = Tool(name="DataFrame Search", func=search_df, description="Searches the stock dataframe.")
# internet_tool = Tool(name="Internet Search", func=search_ddg, description="Searches the internet for relevant information.")


# convert string to df
def string_to_dataframe(s):
    try:
        data = StringIO(s)
        df = pd.read_csv(data, sep="\t")
        return df
    except Exception as e:
        return None


tools = [search_df, general_info]
llm_with_tools = st.session_state["model"].bind_tools(tools)

# agent_prompt = PromptTemplate(
#     input_variables=["query", "tool_descriptions"],
#     template="""
#     The user has asked: {query}
#     Based on the query, you have the following tools at your disposal:
#     {tool_descriptions}

#     Choose the appropriate tool to handle the user's query. If the query is about stock data, use the relevant tools. For general information, provide a response using the LLM.
#     """
# )

# Create the agent
# class StockAgent(LLMSingleActionAgent):
#     def __init__(self, llm, tools, prompt):
#         super().__init__(llm=llm, prompt=prompt)
#         self.tools = {tool.name: tool for tool in tools}

#     def _call(self, query):
#         # Check the intent of the query
#         if "stock" in query.lower():
#             if any(stock in query.upper() for stock in df['Stock'].values):
#                 # Use dataframe and internet tools for stock-specific queries
#                 dataframe_result = self.tools['DataFrame Search'](query)
#                 internet_result = self.tools['Internet Search'](query)
#                 return f"DataFrame Result:\n{dataframe_result}\n\nInternet Result:\n{internet_result}"
#             else:
#                 # Use only the internet tool for general stock-related queries
#                 internet_result = self.tools['Internet Search'](query)
#                 return f"Internet Result:\n{internet_result}"
#         else:
#             # Use the LLM for general queries
#             return self.llm(query)

# Initialize the agent
# stock_agent = StockAgent(llm=general_info, tools=[dataframe_tool, internet_tool], prompt=agent_prompt)

# Agent executor to handle user queries
# agent_executor = AgentExecutor(agent=stock_agent)


# Function to generate the response
def generate_response(user_input):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    system = """You have to analyze the user query and decide the use of tools appropraitely.
    run the search_df function when user asks about any kind of stock information.
    otherwise run the general_info function.

    User: {input}"""
    prompt = ChatPromptTemplate.from_template(system)
    chain = prompt | llm_with_tools
    response = chain.invoke({"input": user_input})
    if response.tool_calls:
        selected_tool = response.tool_calls[0]["name"].lower()
        if selected_tool == "search_df":
            df_results, ddg_results = search_df(user_input)
            gen_info = general_info(user_input)
        else:
            gen_info = general_info(user_input)
            return gen_info
    # for r in response:
    #     if r['name']=='search_df':
    #         print(1)
    #         df_results, ddg_results = search_df(user_input)
    #         print('df', df_results)
    #         print(type(df_results))
    #         gen_info = general_info(user_input, df_results)
    #         print('gen')

    #     elif r['name']=='general_info':
    #         print(2)
    #         gen_info = general_info(user_input, [])

    # response = agent_executor(user_input)
    # return response
    # return jsonify({"response": response})
    # start = time.time()
    # df_results = search_df(user_input)
    # print('df: ',time.time()-start)
    # start = time.time()
    # ddg_results = search_ddg(user_input)
    # print('ddg: ',time.time()-start)
    # start = time.time()

    # gen_info = general_info(user_input, df_results)
    # print('gen: ',time.time()-start)

    if not gen_info:
        gen_info = ""

    # df_results = df_results[0]
    # ddg_results = df_results[1]
    # # Check if df_results is an instance of pd.DataFrame
    is_df_results_dataframe = isinstance(df_results, pd.core.frame.DataFrame)
    is_df_results_series = isinstance(df_results, pd.Series)
    # print(type(df_results))

    if is_df_results_series and not df_results.empty:
        df_results = df_results.to_frame()

    if is_df_results_dataframe and not df_results.empty and ddg_results:
        # Display both DataFrame and DuckDuckGo results
        return [df_results, gen_info, ddg_results]
    elif is_df_results_dataframe and not df_results.empty:
        # Display only DataFrame results
        return [df_results, gen_info]
    elif isinstance(df_results, str) and ddg_results:
        # Display the string result returned by search_df as DataFrame
        string_df = string_to_dataframe(df_results)
        if string_df is not None:
            return [string_df, gen_info, ddg_results]
        else:
            # If the conversion to DataFrame failed, return the original string
            return [df_results, gen_info, ddg_results]
    elif ddg_results:
        # Display only DuckDuckGo results
        return [gen_info, ddg_results]
    else:
        return "No results found"


# Container for chat history
response_container = st.container()
# Container for text box
container = st.container()

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        # get response
        output = generate_response(user_input)
        st.session_state["past"].append(user_input)

        # Store the output
        st.session_state["generated"].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            output = st.session_state["generated"][i]
            if isinstance(output, list):
                for result in output:
                    if isinstance(result, str):
                        output_without_SC = result.replace("$", "").replace("%", "")
                        st.write(output_without_SC)
                    elif isinstance(result, list):
                        title = result[0]
                        body = result[1]
                        link = result[2]
                        for i in range(8):
                            st.markdown(f"#### {title[i]}")
                            st.markdown(body[i])
                            st.markdown(f"[{link[i]}]({link[i]})")
                    else:
                        st.write(result)  # Display non-string results as is
            elif isinstance(output, pd.DataFrame):
                st.dataframe(output)
            elif isinstance(output, str):
                st.write(output)
            else:
                st.write(f"Unexpected output type: {type(output)}")
