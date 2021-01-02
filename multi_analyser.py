import numpy as np
from numpy import genfromtxt
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
import sys
import os

np.set_printoptions(threshold=sys.maxsize, linewidth=200, suppress=True)

def preprocess(csv_file, lookback, holding):
	# Remove the date and p column
	ticker_data_appl = read[:, 1:-2]
	volume_data_appl = read[:, -2:-1]

	set = []
	profits = []

	for i in range(ticker_data_appl.shape[0] - lookback - holding):
		training_set_ticker = np.copy(ticker_data_appl[i:i + lookback])
		close_prize = training_set_ticker[-1][-1]
		training_set_volumne = np.copy(volume_data_appl[i:i + lookback])

		training_set_ticker /= np.max(training_set_ticker)
		training_set_volumne /= np.max(training_set_volumne)

		training_set = np.append(training_set_ticker, training_set_volumne)

		sell_day_open_prize = ticker_data_appl[i + lookback + holding - 1][0]
		profit = sell_day_open_prize / close_prize

		set.append(training_set)
		profits.append(profit)

	set = np.asarray(set)
	profits = np.asarray(profits)

	return set, profits


if __name__ == '__main__':
	#More markets at https://www.quantiacs.com/For-Quants/Markets.aspx
	stocks = os.listdir('tickerData')
	#stocks = ["AAPL.txt", "ABBV.txt", "ABT.txt", "ACN.txt", "AEP.txt", "AIG.txt"]

	trading_days = 300
	holding_days = 30

	columns = trading_days * 5
	X_multi_stock = []
	y_multi_stock = []
	for stock in stocks:
		# Skip the column description
		print(stock)
		try:
			read = genfromtxt('tickerData/'+stock, delimiter=',', skip_header=1)
			X, y = preprocess(read, trading_days, holding_days)
			X_multi_stock.append(X)
			y_multi_stock.append(y)
		except IndexError:
			print("Skipping " + stock)
			continue

	X_multi_stock = np.concatenate(X_multi_stock)
	y_multi_stock = np.concatenate(y_multi_stock)

	X_train, X_test, y_train, y_test = train_test_split(X_multi_stock, y_multi_stock, test_size=0.33, random_state=11)

	makenew = False
	if makenew:
		model = Sequential()
		model.add(Dense(6000, input_shape=(columns,), activation='relu'))
		model.add(BatchNormalization())
		model.add(Dense(3000, activation='relu'))
		model.add(Dense(2000, activation='relu'))
		model.add(Dense(1000, activation='relu'))
		model.add(Dense(500, activation='relu'))
		model.add(Dense(200, activation='relu'))
		model.add(Dense(20, activation='relu'))
		model.add(Dense(1))

		model.compile(optimizer='adam', loss='mse')
		model.fit(X_train, y_train, epochs=1, batch_size=32)
		model.save("modelmulti")
		loss = model.evaluate(X_test, y_test, verbose=0)
		print(loss)
	else:
		model = tensorflow.keras.models.load_model("modelmulti")

	yhat = model.predict(X_test)
	predictions = np.column_stack((yhat.flatten(), y_test))

	#predictions_over_10per = predictions[np.where([i > 1.04 for i in predictions[:,0]])]
	#negative_results = predictions_over_10per[np.where([i < 1. for i in predictions_over_10per[:,1]])]

	test_size = predictions.shape[0]

	cutoff = np.arange(0.80, 1.3, 0.01)
	means = [np.mean(predictions[np.where([i > j for i in predictions[:,0]]), 1]) for j in cutoff]
	medians = [np.median(predictions[np.where([i > j for i in predictions[:, 0]]), 1]) for j in cutoff]
	sums = [np.sum(predictions[np.where([i > j for i in predictions[:, 0]]), 1] - 1) for j in cutoff]
	ar = [(predictions[np.where([i > j for i in predictions[:, 0]]), 1] - 1) for j in cutoff]
	negative_sum = [np.sum(np.where(row < 0., row, 0)) for row in ar]
	predictions_over_j = [predictions[np.where([i > j for i in predictions[:, 0]])] for j in cutoff]
	predictions_over_j_in_per = [row.shape[0]/test_size for row in predictions_over_j]
	predictions_over_j_negative_in_per = [row[np.where([i < 1. for i in row[:, 1]])].shape[0]/row.shape[0] for row in predictions_over_j]
	print(np.column_stack((cutoff, sums, np.add(sums, negative_sum), means, medians, negative_sum, predictions_over_j_in_per, predictions_over_j_negative_in_per)))




