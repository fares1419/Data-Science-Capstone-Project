import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

# Load the trained model
model = joblib.load('model.pkl')

# Load the label encoder for trading name
trading_name_encoder = joblib.load('le.pkl')

# Create a Flask app
app = Flask(__name__)

# Define the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the result page and the POST method to receive user input and make a prediction
@app.route('/predict', methods=['POST'])
def result():
    # Get input from user
    trading_name = request.form['trading_name']
    open_value = float(request.form['open'])
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])

    # Encode the trading name input
    trading_name_encoded = trading_name_encoder.transform([trading_name])[0]

    # Create a new dataframe with the input values
    input_df = pd.DataFrame({'trading_name': trading_name_encoded, 'open': open_value, 'year': year, 'month': month, 'day': day}, index=[0])

    # Predict the close value
    predicted_close = model.predict(input_df[['trading_name', 'open', 'year', 'month', 'day']])[0]

    # Render the result template with the predicted close value
    return render_template('result.html', predicted_close=predicted_close)

if __name__ == '__main__':
    app.run(debug=True)
