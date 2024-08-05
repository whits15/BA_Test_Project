import pandas as pd
import io
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
import os
from flask import Flask, render_template, request, send_file
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import openai

#TODO make chat gpt function async NOTE: probably not doing this

#chapters used: 2,3,8,10

# Load environment variable from .env file
load_dotenv()

app = Flask(__name__)


openai.api_key = os.getenv("OPENAI_API_KEY")


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    hist.reset_index(inplace=True)
    return hist


def create_plot(df, ticker):
    if not df.empty:
        fig = px.line(df, x='Date', y='Close', title=f'{ticker} Closing Prices')
        return fig.to_html(full_html=False)
    return None


def create_candlestick_chart(df, ticker):
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        fig.update_layout(title=f'{ticker} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
        return fig.to_html(full_html=False)
    return None


def create_volume_chart(df, ticker):
    if not df.empty:
        fig = px.line(df, x='Date', y='Volume', title=f'{ticker} Volume')
        return fig.to_html(full_html=False)
    return None


def perform_regression(df):
    if not df.empty:
        df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
        X = df[['Date_ordinal']]
        y = df['Close']

        if len(X) > 0 and len(y) > 0:  # Check if there are enough samples
            model = LinearRegression()
            model.fit(X, y)

            df['Trend'] = model.predict(X)

            regression_line = go.Scatter(
                x=df['Date'],
                y=df['Trend'],
                mode='lines',
                name='Trend Line'
            )
            return regression_line
    return None


def analyze_stock(df, ticker):
    if not df.empty:
        recent_data = df.tail(5).to_dict(orient='records')
        prompt = f"Analyze the recent performance and provide recommendations for the stock {ticker} based on the following data and output your response in at most 150 words with recommendation of buy, sell, or hold:\n{recent_data}"

       # response = openai.chat.completions.create(
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a stock market analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        analysis = response.choices[0].message.content
        return analysis
    return "No data available for analysis."


@app.route('/', methods=['GET', 'POST'])
def index():
    selected_ticker = request.form.get('ticker', 'AAPL')  # Default to AAPL if none selected
    df = get_stock_data(selected_ticker)

    if df.empty:
        return render_template('index.html', selected_ticker=selected_ticker, plot=None, candlestick_chart=None,
                               volume_chart=None, analysis="No data available for analysis.", stock_data=[])

    plot = create_plot(df, selected_ticker)
    candlestick_chart = create_candlestick_chart(df, selected_ticker)
    volume_chart = create_volume_chart(df, selected_ticker)

    regression_line = perform_regression(df)
    if regression_line:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(regression_line)
        plot = fig.to_html(full_html=False)

    analysis = analyze_stock(df, selected_ticker)
    stock_data = df

    if stock_data is not None:
        stock_data['Open'] = stock_data['Open'].apply(lambda x: f"{x:.2f}")
        stock_data['Close'] = stock_data['Close'].apply(lambda x: f"{x:.2f}")
        stock_data['High'] = stock_data['High'].apply(lambda x: f"{x:.2f}")
        stock_data['Low'] = stock_data['Low'].apply(lambda x: f"{x:.2f}")
        stock_data = stock_data.to_dict(orient='records')
    else:
        stock_data = []

    return render_template('index.html', selected_ticker=selected_ticker, plot=plot,
                           candlestick_chart=candlestick_chart, volume_chart=volume_chart, analysis=analysis,
                           stock_data=stock_data)

@app.route('/download', methods=['POST'])
def download():
    selected_ticker = request.form.get('ticker')
    if selected_ticker:
        stock_data = yf.download(selected_ticker, period="5y")
        stock_data.reset_index(inplace=True)
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        stock_data.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
        output.seek(0)
        return send_file(output, download_name=f'{selected_ticker}_stock_data.xlsx', as_attachment=True)
    return 'No ticker provided', 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Default to 8000 if PORT not set
    app.run(host='0.0.0.0', port=port)




