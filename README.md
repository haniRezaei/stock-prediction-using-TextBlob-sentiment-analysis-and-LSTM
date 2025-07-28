# stock-prediction-using-TextBlob-sentiment-analysis-and-LSTM
We have implemented a hybrid stock price prediction model using sentiment analysis (textblob) and LSTM.
we predict DJIA Adjusted Close stock prices using Numerical financial features (Open, High, Low, Close, Volume) and Sentiment analysis features (Polarity and Subjectivity scores from TextBlob) and a Deep learning model(LSTM).
This is a regression task, where the model learns to predict stock price using past stock data and sentiment features.
* Preprocessing & Cleaning
News Data (Combined_News_DJIA.csv):
•	All top 25 daily news headlines are merged into a single string (combined news).
•	Text is cleaned: lowercased, punctuation removed, tokenized, stopwords removed, and lemmatized.
•	Polarity and subjectivity scores are computed using TextBlob.
•	A sentiment label (Positive, Negative, Neutral) is created based on polarity for analysis purposes (not used directly in modeling here).
* Merging and Feature Engineering
•	The cleaned news dataset is merged with the stock dataset using the Date column.
•	Selected features are:
Numerical: Open, High, Low, Close, Volume
Sentiment: Polarity, Subjectivity
Target variable is Adj Close

* Train/Test Split and Scaling
•	Data is split by date:
o	Train: Dates < 2015
o	Test: Dates ≥ 2015
•	Features and target are scaled to [0,1] using MinMaxScaler for better LSTM performance.

* Sequence Creation for LSTM
Since LSTM needs sequences:
•	For each day t, a window of the previous 10 days' features is used to predict the price at t+10.
•	X_train_seq and y_train_seq are created using a sliding window.

* LSTM Model 
our model is a 3-layer LSTM with dropout to reduce overfitting:
LSTM(128) and Dropout(0.1)
LSTM(64) and Dropout(0.1)
LSTM(32) and Dropout(0.1)
Dense(1) , Predict Adj Close price
•	Loss: Mean Squared Error (MSE)
•	Optimizer: Adam
•	Early stopping is used based on validation loss with patience of 20 epochs.
* Evaluation
After training, we evaluate the model:
•	Predict test set using model.predict()
•	Inverse-transform predictions to get real price values
we	Compute,	MSE (mean squared error), RMSE, MAE
o	MAPE (mean absolute percentage error), R² score
the Plot is for	Actual vs. Predicted prices and	Training and validation loss curves

