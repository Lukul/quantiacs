import numpy as np
from numpy import genfromtxt
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
import sys

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
	#Skip the column description
	read = genfromtxt('tickerData/AAPL.txt', delimiter=',', skip_header=1)
	#Remove the date and p column
	ticker_data_appl = read[:, 1:-2]
	volume_data_appl = read[:, -2:-1]

	trading_days = 250
	holding_days = 30

	columns = trading_days * 5

	set = []
	profits = []

	for i in range(ticker_data_appl.shape[0] - trading_days - holding_days):
		training_set_ticker = np.copy(ticker_data_appl[i:i+trading_days])
		close_prize = training_set_ticker[-1][-1]
		training_set_volumne = np.copy(volume_data_appl[i:i+trading_days])

		training_set_ticker /= np.max(training_set_ticker)
		training_set_volumne /= np.max(training_set_volumne)

		training_set = np.append(training_set_ticker, training_set_volumne)

		sell_day_open_prize = ticker_data_appl[i+trading_days+holding_days-1][0]
		profit = sell_day_open_prize / close_prize

		set.append(training_set)
		profits.append(profit)

	set = np.asarray(set)
	profits = np.asarray(profits)

	X_train, X_test, y_train, y_test = train_test_split(set, profits, test_size=0.2, random_state=9)

	makenew = True
	if makenew:
		model = Sequential()
		model.add(Dense(4000, input_shape=(columns,), activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(2500, activation='relu'))
		model.add(Dense(1000, activation='relu'))
		model.add(Dense(500, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(20, activation='relu'))
		model.add(Dense(1))

		model.compile(optimizer='adam', loss='mse')#tensorflow.keras.losses.MeanAbsoluteError()
		model.fit(X_train, y_train, epochs=3, batch_size=32)
		model.save("model")
		loss = model.evaluate(X_test, y_test, verbose=0)
		print(loss)
	else:
		model = tensorflow.keras.models.load_model("model")

	yhat = model.predict(X_test)
	predictions = np.column_stack((yhat.flatten(), y_test))

	predictions_over_10per = predictions[np.where([i > 1.04 for i in predictions[:,0]])]
	negative_results = predictions_over_10per[np.where([i < 1. for i in predictions_over_10per[:,1]])]
	print(predictions)
	print(predictions.shape)
	print(predictions_over_10per.shape)
	print(predictions_over_10per)
	print(negative_results.shape)





