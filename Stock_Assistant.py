import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import google.generativeai as genai

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyCCXqacpCkbyxmC_kF-1bEnyVtkqRVIN7Q"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Stock Data Functions
def get_stock_price(ticker):
    try:
        return str(yf.Ticker(ticker).history(period='1d').iloc[-1].Close)
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"

def calculate_SMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        return str(data.rolling(window=window).mean().iloc[-1])
    except Exception as e:
        return f"Error calculating SMA: {str(e)}"

def calculate_EMA(ticker, window):
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        return str(data.ewm(span=window, adjust=False).mean().iloc[-1])
    except Exception as e:
        return f"Error calculating EMA: {str(e)}"

def calculate_RSI(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        delta = data.diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=14 - 1, adjust=False).mean()
        ema_down = down.ewm(com=14 - 1, adjust=False).mean()
        rs = ema_up / ema_down
        return str(100 - (100 / (1 + rs)).iloc[-1])
    except Exception as e:
        return f"Error calculating RSI: {str(e)}"

def calculate_MACD(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        short_EMA = data.ewm(span=12, adjust=False).mean()
        long_EMA = data.ewm(span=26, adjust=False).mean()
        MACD = short_EMA - long_EMA
        signal = MACD.ewm(span=9, adjust=False).mean()
        MACD_histogram = MACD - signal
        return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'
    except Exception as e:
        return f"Error calculating MACD: {str(e)}"

def plot_stock_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        if data.empty:
            return "No data available for this ticker."
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data.Close)
        plt.title(f'{ticker} Stock Price Over Last Year')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.grid(True)
        plt.savefig('stock.png')
        plt.close()
        return "Graph generated successfully."
    except Exception as e:
        return f"Error generating graph: {str(e)}"

# Function Definitions
functions = [
    {
        'name': 'get_stock_price',
        'description': 'Gets the latest stock price given the ticker symbol of a company.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                }
            },
            'required': ['ticker']
        }
    },
    {
        "name": "calculate_SMA",
        "description": "Calculate the simple moving average for a given stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                   "type": "string",
                   "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)", 
                },
                "window": {
                   "type": "integer",
                   "description": "The timeframe to consider when calculating the SMA"
                }
            },
            "required": ["ticker","window"]
        },
    },
    {
        "name": "calculate_EMA",
        "description": "Calculate the exponential moving average for a given stock ticker and a window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },    
                "window": {
                    "type": "integer",
                    "description": "The timeframe to consider when calculating the EMA"
                }
            },
             "required": ["ticker","window"],
        },
    },
    {
        "name": "calculate_RSI",
        "description": "Calculate the RSI for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type":"string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
            },    
            "required": ["ticker"]
        },
    },
    {
        "name": "calculate_MACD",
        "description": "Calculate the MACD for a given stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
            },        
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_stock_price",
        "description": "Plot the stock price for the last year given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol for a company (e.g., AAPL for Apple)",
                },
            },
            "required": ["ticker"],
        },
    },        
]             

# Available Functions Mapping
available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price
}

# Streamlit UI Setup
st.title('Stock Assistant')

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['parts'][0]['text'])

# User Input
user_input = st.chat_input('Your input:')

# Main Logic
if user_input:
    # Display user message
    with st.chat_message('user'):
        st.write(user_input)

    # Add user message to chat history
    st.session_state['messages'].append({'role': 'user', 'parts': [{'text': user_input}]})

    try:
        # Generate response using Gemini
        chat = model.start_chat(history=[
            {'role': 'user' if msg['role'] == 'user' else 'model', 'parts': [{'text': msg['parts'][0]['text']}]}
            for msg in st.session_state['messages']
        ])
        response = chat.send_message(user_input)

        # Process the response
        if response.text:
            # Check if the response contains a function call
            function_call = None
            for func in functions:
                if func['name'] in response.text:
                    function_call = func
                    break

            if function_call:
                function_name = function_call['name']
                # Extract ticker from the response
                ticker = response.text.split(function_name)[1].split('"')[1]

                # Call the function
                if function_name == 'plot_stock_price':
                    result = plot_stock_price(ticker)
                    if result == "Graph generated successfully.":
                        st.image('stock.png')
                        response_text = f"I've plotted the stock price for {ticker} over the last year."
                    else:
                        response_text = result
                else:
                    function_to_call = available_functions[function_name]
                    if function_name in ['calculate_SMA', 'calculate_EMA']:
                        window = 20  # Default window, you might want to extract this from the response
                        function_response = function_to_call(ticker, window)
                    else:
                        function_response = function_to_call(ticker)
                    response_text = f"Function {function_name} returned: {function_response}"

                # Display function response
                with st.chat_message('assistant'):
                    st.write(response_text)

                # Add function response to chat history
                st.session_state['messages'].append({
                    'role': 'model',
                    'parts': [{'text': response_text}]
                })

                # Generate a second response
                second_response = chat.send_message(f"I've processed your request for {ticker}. Is there anything else you'd like to know?")
                
                # Display second response
                with st.chat_message('assistant'):
                    st.write(second_response.text)

                # Add second response to chat history
                st.session_state['messages'].append({'role': 'model', 'parts': [{'text': second_response.text}]})
            else:
                # Display the response if no function call
                with st.chat_message('assistant'):
                    st.write(response.text)
                
                # Add response to chat history
                st.session_state['messages'].append({'role': 'model', 'parts': [{'text': response.text}]})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")