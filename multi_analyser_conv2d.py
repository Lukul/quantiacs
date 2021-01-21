import numpy as np
from numpy import genfromtxt
import tensorflow.keras
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow import keras
import sys
import os

np.set_printoptions(threshold=sys.maxsize, linewidth=200, suppress=True)

def preprocess(csv_file, lookback, holding):
	# Remove the date and p column
	ticker_data = read[:, 1:-2]
	volume_data = read[:, -2:-1]

	set = []
	profits = []

	for i in range(ticker_data.shape[0] - lookback - holding):
		training_set_ticker = np.copy(ticker_data[i:i + lookback])
		close_prize = training_set_ticker[-1][-1]
		training_set_volumne = np.copy(volume_data[i:i + lookback])

		training_set_ticker /= np.max(training_set_ticker)
		training_set_volumne /= np.max(training_set_volumne)

		training_set = np.column_stack((training_set_ticker.reshape(lookback, 4), training_set_volumne))

		sell_day_open_prize = ticker_data[i + lookback + holding - 1][0]
		profit = sell_day_open_prize / close_prize

		set.append(training_set)
		profits.append(profit)

	set = np.asarray(set)
	profits = np.asarray(profits)

	#start = int(set.shape[0] * 0)
	#end = int(set.shape[0] * 0.8)
	#return set[start:end], profits[start:end]

	return set, profits


if __name__ == '__main__':
	#More markets at https://www.quantiacs.com/For-Quants/Markets.aspx
	#stocks = os.listdir('tickerData')
	#stocks = ["AAPL.txt", "ABBV.txt", "ABT.txt", "ACN.txt", "AEP.txt", "AIG.txt"]
	stocks = ["XOM.txt", "MSFT.txt", "AAPL.txt", "WMT.txt", "BRK.B.txt", "GE.txt", "T.txt", "JNJ.txt", "JPM.txt", "C.txt"]

	trading_days = 100
	holding_days = 20
	columns = 5

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

	makenew = True
	if makenew:
		input_layer = keras.layers.Input((trading_days, 1, columns))

		conv1 = keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, padding="same")(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.ReLU()(conv1)

		conv2 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.ReLU()(conv2)

		conv3 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.ReLU()(conv3)

		flatten = keras.layers.Flatten()(conv3)

		dense1 = Dense(512, activation='relu')(flatten)
		dense2 = Dense(128, activation='relu')(dense1)
		dense3 = Dense(16, activation='relu')(dense2)

		output_layer = keras.layers.Dense(1)(dense3)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(optimizer='adam', loss=keras.losses.Huber(delta=0.1))#'mse')
		model.fit(np.expand_dims(X_train, axis=2), y_train, epochs=30, batch_size=32)
		model.save("modelmulticonv2d")
		loss = model.evaluate(np.expand_dims(X_test, axis=2), y_test, verbose=0)
		print(loss)
	else:
		model = tensorflow.keras.models.load_model("modelmulticonv2d")

	yhat = model.predict(np.expand_dims(X_test, axis=2))
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


	#TODO Lange Brett Prediction: Train on 100. Predict for +1. Use 100 Lookback to Predict 100+1 .. 100+20. ie. the last 20 values of input will be  p redicted by model as well.

