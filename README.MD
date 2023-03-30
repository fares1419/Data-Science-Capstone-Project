# Stock Price Predictor
This is a machine learning project that predicts the closing stock price of companies listed in the Saudi Stock Exchange (Tadawul). The project is based on historical data obtained from Kaggle, with data from January 2014 to May 2020.

# Project Structure
The project has the following structure:

projcet= ipybnb file for the code(loading,cleaning, visulaize dataset and build model)
model: trained machine learning model.
templates: directory containing the HTML templates for the web application.
app.py: the main Flask application file.
#Machine Learning Model
The project uses a linear Regression model to predict the closing stock price of a company. The model is trained on historical data of multiple companies listed on the Tadawul. The features used in the model include the opening price, closing price, highest price, lowest price, and volume of shares traded for each day.

# Web Application
The project includes a web application built using Flask framework. The web application allows the user to enter the trading name, opening price, year, month, and day of a company and returns the predicted closing price using the trained machine learning model.

# Instructions
to Run the Project:
Clone this repository to your local machine.
Navigate to the project directory in your terminal.
Run the Flask application using the command python app.py.
Open your web browser and navigate to http://127.0.0.1:5000/ to access the home page.
Enter the required details and click the "Predict" button to get the predicted closing stock price.
# Acknowledgments
The dataset used in this project was obtained from Kaggle. Special thanks to the owner of the dataset Salwa Alzahrani.
