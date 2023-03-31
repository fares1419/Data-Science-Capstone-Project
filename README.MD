# Stock Price Predictor
This is a machine learning project that predicts the closing stock price of companies listed in the Saudi Stock Exchange (Tadawul). The project is based on historical data obtained from Kaggle, with data from January 2014 to May 2020.
## Libraries Used
Python (3.7.11).
NumPy (1.19.5): A library used for working with arrays and numerical operations.
pandas (1.3.3): A library used for data manipulation and analysis.

Matplotlib (3.4.3): A library used for creating visualizations and plots.

Scikit-learn (1.0): A library used for machine learning algorithms, including regression, classification, and clustering.

seaborn (0.11.2): A library used for data visualization with improved aesthetics and ease of use compared to Matplotlib.

Jupyter Notebook (6.4.3): An open-source web application used for creating and sharing documents that contain live code, equations, visualizations, and narrative text.

## Files in the Repository
Tadawul_stcks.csv: raw data downloaded from Kaggle

projcet.ipynb: Jupyter notebook for data cleaning and preprocessing.

model.pkl: trained machine learning model.

README.md: readme file containing project description and documentation.

app.py: the main Flask application file.

templates: directory containing the HTML templates for the web application.

templates/home.html: contains the HTML and Jinja code for the home page of the web application. It includes a form for the user to input their data, as well as some basic styling using CSS.

templates/result.html: contains the HTML and Jinja code for displaying the results of the prediction on the user's input data. It includes the predicted value and some additional information, as well as some basic styling using CSS.

# Machine Learning Model
The project uses a Random Forest model to predict the closing stock price of a company. The model is trained on historical data of multiple companies listed on the Tadawul. The features used in the model include the opening price, closing price, highest price, lowest price, and volume of shares traded for each day.

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
