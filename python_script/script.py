import pandas as pd
import openai
from io import StringIO
from duckduckgo_search import ddg
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


# searching inside the data frame:
def search_df(user_input):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    df = pd.read_csv("ChatGPTData2.csv")

    # this will have the column names...
    Data_dict = f"""
    Stock, Company, Close Date, Previous Close, Sentiment, Momentum, Price Oscillator, Relative Strength Index, William's R, Price Channels Lower, Price Channels Upper, Convergence/Divergence Moving Average, Double Exponential Moving Average, Exponential Moving Average, Hull Moving Average, Smoothed Moving Average, Weighted Moving Average, Bearish Engulfing, Bearish Harami, Bullish Engulfing, Bullish Harami, Dark Cloud Cover, Doji, Dragonfly Doji, Gravestone Doji, Hammer, Inverted Hammer, Hanging Man, Morning Star, Morning Star Doji, Piercing Pattern, Rain Drop, Rain Drop Doji, Shooting Star, Star, Doji Star, Awesome Oscillator, Kaufman Adp Moving Average, Percent Price Oscillator, Percent Price Oscillator Historical, Percent Price Oscillator Signal, Rate of Change, Stochastic, Stochastic Relative Strength Index, Stochastic Relative Strength Index D, Stochastic Relative Strength Index K, Stochastic Signal, True Strength Index, Ultimate Oscillator, Average Directional Movement Index, Average Directional Movement Index Negative, Average Directional Movement Index Positive, Aroon Down, Aroon Indicator, Aroon Up, Commodity Channel Index, Detr Price Oscillator, Exp Moving Average Fast, Exp Moving Average Slow, Ichimoku A, Ichimoku B, Ichimoku Base, Ichimoku Conv, Know Sure Thing, Know Sure Thing Difference, Know Sure Thing Signal, Moving Average Convergence/Divergence, Moving Average Convergence/Divergence Difference, Moving Average Convergence/Divergence Signal, Mass Index, Parabolic Stop & Reverse Down, Parabolic Stop & Reverse Down Indicator, Parabolic Stop & Reverse Up, Parabolic Stop & Reverse Up Indicator, Simple Moving Average Fast, Simple Moving Average Slow, Schaff Trn Cyc, Tiple Exp Average, Vis Ichimoku A, Vis Ichimoku B, Vortex Difference, Vortex Negative, Vortex Positive, Average True Rng, Bollinger Bands Upper, Bollinger Bands Upper Indicator, Bollinger Bands Lower, Bollinger Bands Lower Indicator, Bollinger Bands Middle, Bollinger Bands Price, Bollinger Bands Width, Donchain Channel Upper, Donchain Channel Lower, Donchain Channel Middle, Donchain Channel Price, Donchain Channel Width, Keltner Channel Middle, Keltner Channel Upper, Keltner Channel Upper Indicator, Keltner Channel Lower, Keltner Channel Lower Indicator, Keltner Channel Price, Keltner Channel Width, Ulcer Index, Accumulation/Distribution Index, Chaikin Money Flow, Em, Force Index, Money Flow Index, Negative Volume Index, On-Balance Volume, Simple Moving Average Em, Price Trend, Weighted Average Price, Chaikin Oscillator, Chande Momentum Oscillator, Price Channel Lower, Price Channel Upper, Detrended Price Oscillator, Envelopes Lower, Envelopes Upper, Stochastic Oscillator Fast D, Stochastic Oscillator Fast K, Fibonacci Retracement (rl=0.0), Fibonacci Retracement (rl=38.2), Fibonacci Retracement (rl=50.0), Fibonacci Retracement (rl=61.8), Fibonacci Retracement (rl=100.0), Forecast Oscillator, Intraday Movement Index, Klinger Oscillator, Linear Regression Indicator, Linear Regression Slope, Market Facilitation Index, Median Price, Performance, Positive Volatility Index, Projection Oscillator, Projection Oscillator Trigger, Projection Bands Lower, Projection Bands Upper, QStick, Range Indicator, Relative Volatility Index, Stochastic Oscillator Slow D, Stochastic Oscillator Slow K, Stochastic Momentum Index, Standard Deviation, Swing Index, Triangular Moving Average, Triple Exponential Moving Average, Time Series Forecast, Time Series Moving Average, Typical Price, Variable Moving Average, Vertical Horizontal Filter, Volatility Chaikins, Volume Oscillator, Volume Rate of Change, Weighted Close, Wilders Smoothing, Williams Accumulation Distribution, 1d Price, 1d Delta, 1d Alpha, 3d Price, 3d Delta, 3d Alpha, 5d Price, 5d Delta, 5d Alpha, 10d Price, 10d Delta, 10d Alpha, 2w Price, 2w Delta, 2w Alpha, 3w Price, 3w Delta, 3w Alpha, 6w Price, 6w Delta, 6w Alpha, 2m Price, 2m Delta, 2m Alpha, 3m Price, 3m Delta, 3m Alpha, 6m Price, 6m Delta, 6m Alpha, 9m Price, 9m Delta, 9m Alpha, 1y Price, 1y Delta, 1y Alpha, 1w Alpha + 3%, 1m Alpha + 3%, 3m Alpha + 3%, 6m Alpha + 10%, 9m Alpha + 10%, 1y Alpha + 10%, Price Level Shift Anomaly, Price Change Anomaly, Bullish Short-Term (3d) Trend Anomaly, Bearish Short-Term (3d) Trend Anomaly, Bullish Long-Term (1w) Trend Anomaly, Bearish Long-Term (1w) Trend Anomaly, Volatility, Alpha Short (<2w), Alpha Mid (3w - 6w), Alpha Long (>7w), Blackrock Index Alpha, Bond Market Index Alpha, Dow Jones Index Alpha, NASDAQ Index Alpha, S&P 500 Index Alpha, Stock Market Index Alpha, Energy Index Alpha"""
    Data_dict_indicators_rank = f""" 'Momentum Rank', 'Price Oscillator Rank', 'Relative Strength Index Rank', "William's R Rank", 'Price Channels Lower Rank', 'Price Channels Upper Rank', 'Convergence/Divergence Moving Average Rank', 'Double Exponential Moving Average Rank', 'Exponential Moving Average Rank', 'Hull Moving Average Rank', 'Smoothed Moving Average Rank', 'Weighted Moving Average Rank', 'Awesome Oscillator Rank', 'Kaufman Adp Moving Average Rank', 'Percent Price Oscillator Rank', 'Percent Price Oscillator Historical Rank', 'Percent Price Oscillator Signal Rank', 'Rate of Change Rank', 'Stochastic Rank', 'Stochastic Relative Strength Index Rank', 'Stochastic Relative Strength Index D Rank', 'Stochastic Relative Strength Index K Rank', 'Stochastic Signal Rank', 'True Strength Index Rank', 'Ultimate Oscillator Rank', 'Average Directional Movement Index Rank', 'Average Directional Movement Index Negative Rank', 'Average Directional Movement Index Positive Rank', 'Aroon Down Rank', 'Aroon Indicator Rank', 'Aroon Up Rank', 'Commodity Channel Index Rank', 'Detr Price Oscillator Rank', 'Exp Moving Average Fast Rank', 'Exp Moving Average Slow Rank', 'Ichimoku A Rank', 'Ichimoku B Rank', 'Ichimoku Base Rank', 'Ichimoku Conv Rank', 'Know Sure Thing Rank', 'Know Sure Thing Difference Rank', 'Know Sure Thing Signal Rank', 'Moving Average Convergence/Divergence Rank', 'Moving Average Convergence/Divergence Difference Rank', 'Moving Average Convergence/Divergence Signal Rank', 'Mass Index Rank', 'Parabolic Stop & Reverse Down Rank', 'Parabolic Stop & Reverse Up Rank', 'Simple Moving Average Fast Rank', 'Simple Moving Average Slow Rank', 'Schaff Trn Cyc Rank', 'Tiple Exp Average Rank', 'Vis Ichimoku A Rank', 'Vis Ichimoku B Rank', 'Vortex Difference Rank', 'Vortex Negative Rank', 'Vortex Positive Rank', 'Average True Rng Rank', 'Bollinger Bands Upper Rank', 'Bollinger Bands Lower Rank', 'Bollinger Bands Middle Rank', 'Bollinger Bands Price Rank', 'Bollinger Bands Width Rank', 'Donchain Channel Upper Rank', 'Donchain Channel Lower Rank', 'Donchain Channel Middle Rank', 'Donchain Channel Price Rank', 'Donchain Channel Width Rank', 'Keltner Channel Middle Rank', 'Keltner Channel Upper Rank', 'Keltner Channel Lower Rank', 'Keltner Channel Price Rank', 'Keltner Channel Width Rank', 'Ulcer Index Rank', 'Accumulation/Distribution Index Rank', 'Chaikin Money Flow Rank', 'Em Rank', 'Force Index Rank', 'Money Flow Index Rank', 'Negative Volume Index Rank', 'On-Balance Volume Rank', 'Simple Moving Average Em Rank', 'Price Trend Rank', 'Weighted Average Price Rank', 'Chaikin Oscillator Rank', 'Chande Momentum Oscillator Rank', 'Price Channel Lower Rank', 'Price Channel Upper Rank', 'Detrended Price Oscillator Rank', 'Envelopes Lower Rank', 'Envelopes Upper Rank', 'Stochastic Oscillator Fast D Rank', 'Stochastic Oscillator Fast K Rank', 'Fibonacci Retracement (rl=0.0) Rank', 'Fibonacci Retracement (rl=38.2) Rank', 'Fibonacci Retracement (rl=50.0) Rank', 'Fibonacci Retracement (rl=61.8) Rank', 'Fibonacci Retracement (rl=100.0) Rank', 'Forecast Oscillator Rank', 'Intraday Movement Index Rank', 'Klinger Oscillator Rank', 'Linear Regression Indicator Rank', 'Linear Regression Slope Rank', 'Market Facilitation Index Rank', 'Median Price Rank', 'Performance Rank', 'Positive Volatility Index Rank', 'Projection Oscillator Rank', 'Projection Oscillator Trigger Rank', 'Projection Bands Lower Rank', 'Projection Bands Upper Rank', 'QStick Rank', 'Range Indicator Rank', 'Relative Volatility Index Rank', 'Stochastic Oscillator Slow D Rank', 'Stochastic Oscillator Slow K Rank', 'Stochastic Momentum Index Rank', 'Standard Deviation Rank', 'Swing Index Rank', 'Triangular Moving Average Rank', 'Triple Exponential Moving Average Rank', 'Time Series Forecast Rank', 'Time Series Moving Average Rank', 'Typical Price Rank', 'Variable Moving Average Rank', 'Vertical Horizontal Filter Rank', 'Volatility Chaikins Rank', 'Volume Oscillator Rank', 'Volume Rate of Change Rank', 'Weighted Close Rank', 'Wilders Smoothing Rank', 'Williams Accumulation Distribution Rank'
    """
    Data_dict_indicators = f""" 'Sentiment', 'Momentum', 'Price Oscillator', 'Relative Strength Index', "William's R", 'Price Channels Lower', 'Price Channels Upper', 'Convergence/Divergence Moving Average', 'Double Exponential Moving Average', 'Exponential Moving Average', 'Hull Moving Average', 'Smoothed Moving Average', 'Weighted Moving Average', 'Awesome Oscillator', 'Kaufman Adp Moving Average', 'Percent Price Oscillator', 'Percent Price Oscillator Historical', 'Percent Price Oscillator Signal', 'Rate of Change', 'Stochastic', 'Stochastic Relative Strength Index', 'Stochastic Relative Strength Index D', 'Stochastic Relative Strength Index K', 'Stochastic Signal', 'True Strength Index', 'Ultimate Oscillator', 'Average Directional Movement Index', 'Average Directional Movement Index Negative', 'Average Directional Movement Index Positive', 'Aroon Down', 'Aroon Indicator', 'Aroon Up', 'Commodity Channel Index', 'Detr Price Oscillator', 'Exp Moving Average Fast', 'Exp Moving Average Slow', 'Ichimoku A', 'Ichimoku B', 'Ichimoku Base', 'Ichimoku Conv', 'Know Sure Thing', 'Know Sure Thing Difference', 'Know Sure Thing Signal', 'Moving Average Convergence/Divergence', 'Moving Average Convergence/Divergence Difference', 'Moving Average Convergence/Divergence Signal', 'Mass Index', 'Parabolic Stop & Reverse Down', 'Parabolic Stop & Reverse Down Indicator', 'Parabolic Stop & Reverse Up', 'Parabolic Stop & Reverse Up Indicator', 'Simple Moving Average Fast', 'Simple Moving Average Slow', 'Schaff Trn Cyc', 'Tiple Exp Average', 'Vis Ichimoku A', 'Vis Ichimoku B', 'Vortex Difference', 'Vortex Negative', 'Vortex Positive', 'Average True Rng', 'Bollinger Bands Upper', 'Bollinger Bands Upper Indicator', 'Bollinger Bands Lower', 'Bollinger Bands Lower Indicator', 'Bollinger Bands Middle', 'Bollinger Bands Price', 'Bollinger Bands Width', 'Donchain Channel Upper', 'Donchain Channel Lower', 'Donchain Channel Middle', 'Donchain Channel Price', 'Donchain Channel Width', 'Keltner Channel Middle', 'Keltner Channel Upper', 'Keltner Channel Upper Indicator', 'Keltner Channel Lower', 'Keltner Channel Lower Indicator', 'Keltner Channel Price', 'Keltner Channel Width', 'Ulcer Index', 'Accumulation/Distribution Index', 'Chaikin Money Flow', 'Em', 'Force Index', 'Money Flow Index', 'Negative Volume Index', 'On-Balance Volume', 'Simple Moving Average Em', 'Price Trend', 'Weighted Average Price', 'Chaikin Oscillator', 'Chande Momentum Oscillator', 'Price Channel Lower', 'Price Channel Upper', 'Detrended Price Oscillator', 'Envelopes Lower', 'Envelopes Upper', 'Stochastic Oscillator Fast D', 'Stochastic Oscillator Fast K', 'Fibonacci Retracement (rl=0.0)', 'Fibonacci Retracement (rl=38.2)', 'Fibonacci Retracement (rl=50.0)', 'Fibonacci Retracement (rl=61.8)', 'Fibonacci Retracement (rl=100.0)', 'Forecast Oscillator', 'Intraday Movement Index', 'Klinger Oscillator', 'Linear Regression Indicator', 'Linear Regression Slope', 'Market Facilitation Index', 'Median Price', 'Performance', 'Positive Volatility Index', 'Projection Oscillator', 'Projection Oscillator Trigger', 'Projection Bands Lower', 'Projection Bands Upper', 'QStick', 'Range Indicator', 'Relative Volatility Index', 'Stochastic Oscillator Slow D', 'Stochastic Oscillator Slow K', 'Stochastic Momentum Index', 'Standard Deviation', 'Swing Index', 'Triangular Moving Average', 'Triple Exponential Moving Average', 'Time Series Forecast', 'Time Series Moving Average', 'Typical Price', 'Variable Moving Average', 'Vertical Horizontal Filter', 'Volatility Chaikins', 'Volume Oscillator', 'Volume Rate of Change', 'Weighted Close', 'Wilders Smoothing', 'Williams Accumulation Distribution'
    """
    Data_dict_Information = f""" 'Stock', 'Company', 
    """
    Data_dict_dates = f""" 'Close Date', 'Previous Close'
    """
    Data_dict_Volatility_Cluster = f""" 'Volatility', 'Alpha Short (<2w)', 'Alpha Mid (3w - 6w)', 'Alpha Long (>7w)'
    """
    Data_dict_Volatility_Cluster_Rank = f""" 'Alpha Short (<2w) Rank', 'Alpha Mid (3w - 6w) Rank', 'Alpha Long (>7w) Rank'
    """
    Data_dict_anomalies = f""" 'Price Level Shift Anomaly', 'Price Change Anomaly', 'Bullish Short-Term (3d) Trend Anomaly', 'Bearish Short-Term (3d) Trend Anomaly', 'Bullish Long-Term (1w) Trend Anomaly', 'Bearish Long-Term (1w) Trend Anomaly'
    """
    Data_dict_Prediction = f""" '1d Price', '1d Delta', '1d Alpha', '3d Price', '3d Delta', '3d Alpha', '5d Price', '5d Delta', '5d Alpha', '10d Price', '10d Delta', '10d Alpha', '2w Price', '2w Delta', '2w Alpha', '3w Price', '3w Delta', '3w Alpha', '6w Price', '6w Delta', '6w Alpha', '2m Price', '2m Delta', '2m Alpha', '3m Price', '3m Delta', '3m Alpha', '6m Price', '6m Delta', '6m Alpha', '9m Price', '9m Delta', '9m Alpha', '1y Price', '1y Delta', '1y Alpha', '1w Alpha + 3%', '1m Alpha + 3%', '3m Alpha + 3%', '6m Alpha + 10%', '9m Alpha + 10%', '1y Alpha + 10%'
    """
    Data_dict_Index_Cluster_Rank = f""" 'Blackrock Index Alpha Rank', 'Bond Market Index Alpha Rank', 'Dow Jones Index Alpha Rank', 'NASDAQ Index Alpha Rank', 'S&P 500 Index Alpha Rank', 'Stock Market Index Alpha Rank', 'Energy Index Alpha Rank', 'Energy Index Rank'
    """
    Data_dict_Index_Cluster = f""" 'Blackrock Index Alpha', 'Bond Market Index Alpha', 'Dow Jones Index Alpha', 'NASDAQ Index Alpha', 'S&P 500 Index Alpha', 'Stock Market Index Alpha', 'Energy Index Alpha', 'Energy Index'
    """
    Data_dict_Candlestick = f""" ['Bearish Engulfing', 'Bearish Harami', 'Bullish Engulfing', 'Bullish Harami', 'Dark Cloud Cover', 'Doji', 'Dragonfly Doji', 'Gravestone Doji', 'Hammer', 'Inverted Hammer', 'Hanging Man', 'Morning Star', 'Morning Star Doji', 'Piercing Pattern', 'Rain Drop', 'Rain Drop Doji', 'Shooting Star', 'Star', 'Doji Star']
    """
    Data_dict_probabilities = f"""
    'Probability of 1w Alpha + 3%', 'Probability of 1m Alpha + 3%', 'Probability of 3m Alpha + 3%', 'Probability of 6m Alpha + 10%', 'Probability of 9m Alpha + 10%', 'Probability of 1y Alpha + 10%'
    """
    System = f"""
    """

    prompt2 = f""" please convert the following natural language query to a dataframe pandas query and please follow the instruction:
    {user_input}
    instruction:
    1- I want to get only and only the query, do not give me additional text with it. undesired output: df = query.
    2- be precise with the query, and make sure that you get it right.
    3- The query must select all the values of column names mentioned in the {user_input} only, and after that it does sorting or other stuff.
    4- The queries should always list the columns present in {Data_dict_Information}.
    5- please use Stock if you get symbol in {user_input}, and use Compnay if you get the whole company name.
    5- if {user_input} has Low, medium or high in lowercase, please convert them in uppercase to HIGH,LOW or MEDIUM.
    6- if technical indicators word is in {user_input}, you must select all the keywords mentioned in {Data_dict_indicators} to create the DF query.
    7- if clusters word is in {user_input}, use use all the keywords from {Data_dict_Index_Cluster} to create the DF query.
    7- if clusters rank word is in {user_input}, use use all the keywords from {Data_dict_Index_Cluster_Rank} to create the DF query.
    8- if {user_input} have a keyword from {Data_dict_indicators_rank}, use {Data_dict_indicators_rank} and {user_input} to replace the Column Name Rank and RANK in the following query and return it: df.loc[df['Column Name Rank'] == 'RANK', ['Stock', 'Company', 'Column Name Rank']].
    9- if votality clusters word is in {user_input}, use use all the keywords from {Data_dict_Volatility_Cluster} to create the DF query.
    11- if votality rank word is in {user_input}, use use all the keywords from {Data_dict_Volatility_Cluster_Rank} to create the DF query.
    12- If Candlestick or Candlestick pattern word is in {user_input}, use  {Data_dict_Candlestick}  to identify the keywords in {user_input} to create the DF query that return the value either TRUE or FALSE of the given column. 
    13- if {user_input} asked about which stocks have a true candlestick pattern for a given column name from {Data_dict_Candlestick}, this may help you: df.loc[df['column name'] == True, ['Stock', 'Company']].to_string(index=False)
    14- If {user_input} wants to list any true candlestick for a given stock, you must use the following query and refer to {Data_dict_Candlestick} to complete it: df.query("Stock == 'name' & (`first candlestick name` == True | `second candlestick name` == True | etc..)")[['Stock', 'Company', 'Close Date', 'Previous Close', 'Bearish Engulfing', 'Bearish Harami', 'Bullish Engulfing', 'Bullish Harami', 'Dark Cloud Cover', 'Doji', 'Dragonfly Doji', 'Gravestone Doji', 'Hammer', 'Inverted Hammer', 'Hanging Man', 'Morning Star', 'Morning Star Doji', 'Piercing Pattern', 'Rain Drop', 'Rain Drop Doji', 'Shooting Star', 'Star', 'Doji Star']].apply(lambda row: row[row == True].index.tolist(), axis=1).to_string(index=False). 
    15- If price, prediction or price prediction is in {user_input}, use use all the keywords from {Data_dict_Prediction} to create the DF query.
    16- If anomaly or anomalies word is in {user_input}, use all the keywords from {Data_dict_anomalies} to create the DF query.
    17- If {user_input} does not have words similar to candlestick, anomaly, cluster or prediction, The query must select all the values of column names mentioned in the {user_input} only, and after that it does sorting or other stuff.
    19- be careful about the keywords, these column names ( keywords ) should not be modified, use this {Data_dict} to help you identify the keywords.
    20- to remove the index column, please use this at the end of the query: to_string(index=False).
    21- if the word all is in {user_input}, bring all stocks in the dataframe query.
    22- if the {user_input} specified the number of stock to be sorted and the word all is not in {user_input}, bring only the number of stocks required by the {user_input}.
    23- always focus on selecting the columns first before doing any sorting stuff.
    24- if {user_input} has the probability word in , always refer to this dictionary {Data_dict_probabilities} to convert the {user_input} to a DF query and use the exact keyword from {Data_dict_probabilities}.
    25- if all in {user_input}, sort all stocks and do not user .head().
    26- if you get Rank or Probability on {user_input}, don't remove them from the DF query and always refer to the dictionnary.
    """
    max_attempts = 7
    attempts = 0
    while attempts < max_attempts:
        try:
            model = "gpt-3.5-turbo"
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt2},
                    {"role": "user", "content": f"Text: {user_input}"},
                ],
                temperature=0.2,  # adds randomness in answer, default is 1
            )
            resp = response.choices[0].message.content.strip()
            # response = eval(resp)
            return resp

        except Exception as e:
            print("Error inside search_df: {}".format(e))
            error = "Error: {}".format(e)
            attempts += 1

    if attempts == max_attempts:
        return error


# search in duckduckgo
def search_ddg(user_input):
    try:
        max_results: int = 5
        results = ddg(user_input, max_results=max_results)

        if results is None:
            return ["Error: DuckDuckGo Search Result is None"]

        if len(results) == 0:
            return "No good DuckDuckGo Search Result was found"

        search_results = []

        for result in results:
            snippet = result["body"]
            link = result["href"]

            # Remove special characters from the snippet
            snippet = snippet.replace("$", "").replace("%", "")

            search_results.append(f"{snippet}\n{link}\n\n")

        return search_results
    except Exception as e:
        error_message = f"Error in search_ddg function: {e}"
        return [error_message]


# convert string to df
# def string_to_dataframe(s):
#     try:
#         data = StringIO(s)
#         df = pd.read_csv(data, sep='\t')
#         return df
#     except Exception as e:
#         return None


# Function to generate the response
def generate_response(user_input):
    df_results = search_df(user_input)
    ddg_results = search_ddg(user_input)

    # Check if df_results is an instance of pd.DataFrame
    is_df_results_dataframe = isinstance(df_results, pd.DataFrame)

    # Concatenate the elements of the ddg_results list into a single string
    if ddg_results:
        ddg_results = " ".join(ddg_results)

    if is_df_results_dataframe and not df_results.empty and ddg_results:
        # Display both DataFrame and DuckDuckGo results
        return [df_results, ddg_results]
    elif is_df_results_dataframe and not df_results.empty:
        # Display only DataFrame results
        return df_results
    elif isinstance(df_results, str) and ddg_results:
        # Display the string result returned by search_df as DataFrame
        # string_df = string_to_dataframe(df_results)
        if df_results is not None:
            return [df_results, ddg_results]
        else:
            # If the conversion to DataFrame failed, return the original string
            return [df_results, ddg_results]
    elif ddg_results:
        # Display only DuckDuckGo results
        return ddg_results
    else:
        return "No results found"


# Get user input from console
def get_user_input():
    return input("Enter your query: ")


if __name__ == "__main__":
    user_input = get_user_input()
    response = generate_response(user_input)
    print("------------")
    print("The output starts from here")
    print("------------")
    print(response)
