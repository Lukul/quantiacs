import talib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from enum import IntEnum, unique
@unique
class Candle(IntEnum):
	OPEN = 0
	HIGH = 1
	LOW = 2
	CLOSE = 3
	VOLUME = 4
import time
import datetime

def calculate_candle_stick(trades):
	OPEN = trades[0, 1]
	HIGH = np.max(trades[:, 1])
	LOW = np.min(trades[:, 1])
	CLOSE = trades[-1, 1]
	VOLUME = np.sum(trades[:, 2])
	return np.asarray([OPEN, HIGH, LOW, CLOSE, VOLUME])

def calculate_candle_sticks():
	outfile = 'Kraken_Trading_History/XBTUSD.npy'
	try:
		file = np.load(outfile)
	except FileNotFoundError:
		file = np.genfromtxt('Kraken_Trading_History/XBTUSD.npy', delimiter=',')
		np.save(outfile, file)

	candle_sticks = []

	#test1 = datetime.datetime.utcfromtimestamp(file[0][0]).strftime('%Y-%m-%dT%H:%M:%SZ')
	start_time = file[0][0]
	tick_time = 15 * 60 #15minutes * 60 seconds = unix time difference
	i_start = 0
	i_end = 0
	while start_time < file[-1][0]:
		end_time = start_time + tick_time
		#Find the indicies of the tick
		while True:
			if i_end == file.shape[0] or file[i_end][0] > end_time:
				break
			i_end += 1

		if i_start != i_end:
			candle_sticks.append(calculate_candle_stick(file[i_start:i_end]))
		else:
			CLOSE = candle_sticks[-1][3]
			candle_sticks.append(np.asarray([CLOSE, CLOSE, CLOSE, CLOSE, 0]))

		i_start = i_end
		start_time = end_time
	return np.asarray(candle_sticks)

def scale(data, scaler):
	data = np.asarray(data)
	if data.ndim == 1:
		return np.expand_dims(data[-1],axis=0)
		#return scaler.transform(data[-1].reshape(-1, 1)).flatten()
	else:
		return data[:, -1]
		#return scaler.transform(data[:, -1].reshape(-1, 1)).flatten()

def enhance_with_indicators(data):
	set = []

	OPEN = data[:, Candle.OPEN]
	HIGH = data[:, Candle.HIGH]
	LOW = data[:, Candle.LOW]
	CLOSE = data[:, Candle.CLOSE]
	VOLUME = data[:, Candle.VOLUME]

	low_high = talib.BBANDS(CLOSE, timeperiod=14, nbdevup=2, nbdevdn=2, matype=1)
	low_high = np.asarray([low_high[0][-1], low_high[2][-1]]).reshape(-1, 1)
	low_high_scaler = StandardScaler()
	low_high_scaler.fit(low_high)

	one = np.asarray([-1, 1]).reshape(-1, 1)
	one_scaler = StandardScaler()
	one_scaler.fit(one)

	hundred = np.asarray([-100, 100]).reshape(-1, 1)
	hundred_scaler = StandardScaler()
	hundred_scaler.fit(hundred)

	thousand = np.asarray([-1000, 1000]).reshape(-1, 1)
	thousand_scaler = StandardScaler()
	thousand_scaler.fit(thousand)

	million = np.asarray([-1000000, 1000000]).reshape(-1, 1)
	million_scaler = StandardScaler()
	million_scaler.fit(million)

	set.append(scale(OPEN, low_high_scaler))
	set.append(scale(HIGH, low_high_scaler))
	set.append(scale(LOW, low_high_scaler))
	set.append(scale(CLOSE, low_high_scaler))
	#Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price.
	set.append(scale(talib.BBANDS(CLOSE, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0), low_high_scaler)) #121.03399999999903 19719.281591268886
	set.append(scale(talib.BBANDS(CLOSE, timeperiod=14, nbdevup=2, nbdevdn=2, matype=1), low_high_scaler))
	#The DEMA uses two exponential moving averages (EMAs) to eliminate lag, as some traders view lag as a problem.
	set.append(scale(talib.DEMA(CLOSE, timeperiod=10), low_high_scaler)) #122.0 19573.564771355504
	set.append(scale(talib.DEMA(CLOSE, timeperiod=21), low_high_scaler)) #122.0 19546.76082510694
	set.append(scale(talib.DEMA(CLOSE, timeperiod=100), low_high_scaler)) #123.84710711425136 19578.715808186673
	#However, whereas SMA simply calculates an average of price data, EMA applies more weight to data that is more current.
	set.append(scale(talib.EMA(CLOSE, timeperiod=10), low_high_scaler)) #122.0 19499.362560116417
	set.append(scale(talib.EMA(CLOSE, timeperiod=21), low_high_scaler)) #122.0 19433.26416788178
	set.append(scale(talib.EMA(CLOSE, timeperiod=100), low_high_scaler)) #122.11270000000005 19059.124645340504
	#The HTTrendline at a specific bar gives the current Hilbert Transform Trendline as instantaneously measured at that 
	#bar. In its Series form, the Instantaneous Trendline appears much like a Moving Average, but with minimal lag 
	#compared with the lag normally associated with such averages for equivalent periods.
	set.append(scale(talib.HT_TRENDLINE(CLOSE), low_high_scaler)) #122.0 19471.324
	#Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to account for market noise or volatility. 
	#KAMA will closely follow prices when the price swings are relatively small and the noise is low.
	set.append(scale(talib.KAMA(CLOSE, timeperiod=10), low_high_scaler)) #122.0 19397.611724437047
	set.append(scale(talib.KAMA(CLOSE, timeperiod=21), low_high_scaler)) #122.0 19336.434082122203
	set.append(scale(talib.KAMA(CLOSE, timeperiod=100), low_high_scaler)) #123.61 19301.746826077375
	#The MESA adaptive moving average is a trend-following indicator. It adapts to price movements in a very unique way, 
	#based on the rate of change (ROC), as measured by the Hilbert Transform Discriminator.
	set.append(scale(talib.MAMA((HIGH + LOW) / 2., fastlimit=0.5, slowlimit=0.05), low_high_scaler)) #121.04112572694972 19494.294994956996
	set.append(scale(talib.MIDPOINT(CLOSE, timeperiod=5), low_high_scaler)) #122.0 19544.95
	set.append(scale(talib.MIDPRICE(LOW, HIGH, timeperiod=5), low_high_scaler)) #122.0 19562.6
	#The parabolic SAR indicator, developed by J. Welles Wilder Jr., is used by traders to determine trend direction 
	# and potential reversals in price.
	set.append(scale(talib.SAR(HIGH, LOW, acceleration=0.02, maximum=0.2), low_high_scaler)) #122.0 19660.0
	#A simple moving average (SMA) is an arithmetic moving average calculated by adding recent prices and then dividing 
	# that figure by the number of time periods in the calculation average.
	set.append(scale(talib.SMA(CLOSE, timeperiod=5), low_high_scaler)) #122.0 19553.340000000037
	set.append(scale(talib.SMA(CLOSE, timeperiod=25), low_high_scaler)) #122.0 19405.74400000004
	set.append(scale(talib.SMA(CLOSE, timeperiod=50), low_high_scaler)) #122.0 19286.443999999996
	#The Triple Exponential Moving Average (T3) developed by Tim Tillson attempts to offer a moving average with better 
	#smoothing then traditional exponential moving average. It incorporates a smoothing technique which allows it to 
	#plot curves more gradual than ordinary moving averages and with a smaller lag.
	set.append(scale(talib.T3(CLOSE, timeperiod=5, vfactor=0), low_high_scaler)) #122.0 19498.31237177043
	set.append(scale(talib.T3(CLOSE, timeperiod=10, vfactor=0), low_high_scaler)) #122.0 19419.991324685387
	set.append(scale(talib.T3(CLOSE, timeperiod=21, vfactor=0), low_high_scaler)) #122.84310194419339 19306.63501695168
	#The triple exponential moving average was designed to smooth price fluctuations, thereby making it easier to 
	#identify trends without the lag associated with traditional moving averages (MA).
	set.append(scale(talib.TEMA(CLOSE, timeperiod=7), low_high_scaler)) #122.0 19617.222402494965
	set.append(scale(talib.TEMA(CLOSE, timeperiod=15), low_high_scaler)) #122.0 19586.42515855386
	#The Triangular Moving Average is basically a double-smoothed Simple Moving Average that gives more weight to the 
	#middle section of the data interval. The TMA has a significant lag to current prices and is not well-suited to 
	#fast moving markets.
	set.append(scale(talib.TRIMA(CLOSE, timeperiod=5), low_high_scaler)) #122.0 19567.31111092877
	set.append(scale(talib.TRIMA(CLOSE, timeperiod=25), low_high_scaler)) #122.0 19459.8816568341
	set.append(scale(talib.TRIMA(CLOSE, timeperiod=50), low_high_scaler)) #122.0 19359.257076923175
	#The weighted moving average (WMA) is a technical indicator that assigns a greater weighting to the most recent data 
	#points, and less weighting to data points in the distant past. The WMA is obtained by multiplying each number in 
	#the data set by a predetermined weight and summing up the resulting values.
	set.append(scale(talib.WMA(CLOSE, timeperiod=5), low_high_scaler)) #122.0 19567.840000466134
	set.append(scale(talib.WMA(CLOSE, timeperiod=10), low_high_scaler)) #122.0 19527.127272724356
	set.append(scale(talib.WMA(CLOSE, timeperiod=21), low_high_scaler)) #122.0 19479.342424127473
	set.append(scale(talib.WMA(CLOSE, timeperiod=50), low_high_scaler)) #122.0 19355.600000135404
	set.append(scale(talib.WMA(CLOSE, timeperiod=100), low_high_scaler)) #122.21647326732675 19265.66566335264
	set.append(scale(talib.LINEARREG(CLOSE, timeperiod=14), low_high_scaler))  # 122.0 19585.157142857144
	set.append(scale(talib.LINEARREG_INTERCEPT(CLOSE, timeperiod=14), low_high_scaler)) #121.54000000000003 19643.968571428577
	set.append(scale(talib.TSF(CLOSE, timeperiod=14), low_high_scaler)) #122.0 19609.668131868133



	#ADX stands for Average Directional Movement Index and can be used to help measure the overall strength of a trend. 
	#The ADX indicator is an average of expanding price range values.
	set.append(scale(talib.ADX(HIGH, LOW, CLOSE, timeperiod=10), hundred_scaler)) #0.0 99.99999999999963
	set.append(scale(talib.ADX(HIGH, LOW, CLOSE, timeperiod=14), hundred_scaler)) #0.0 99.9999999940751
	set.append(scale(talib.ADX(HIGH, LOW, CLOSE, timeperiod=21), hundred_scaler)) #0.0 99.99998408446837
	#ADXR stands for Average Directional Movement Index Rating. ADXR is a component of the Directional Movement System 
	#developed by Welles Wilder.
	set.append(scale(talib.ADXR(HIGH, LOW, CLOSE, timeperiod=14), hundred_scaler)) #0.0 99.9999999892742
	#The Aroon indicator is a technical indicator that is used to identify trend changes in the price of an asset,
	#as well as the strength of that trend. In essence, the indicator measures the time between highs and the time
	#between lows over a time period. ... The indicator signals when this is happening, and when it isn't
	set.append(scale(talib.AROON(HIGH, LOW), hundred_scaler)) #0.0 100.0
	#The Directional Movement Index, or DMI, is an indicator developed by J. ... An optional third line, called
	#directional movement (DX) shows the difference between the lines. When +DI is above -DI, there is more upward
	#pressure than downward pressure in the price.
	set.append(scale(talib.DX(HIGH, LOW, CLOSE, timeperiod=5), hundred_scaler)) #0.0 100.0
	set.append(scale(talib.DX(HIGH, LOW, CLOSE, timeperiod=21), hundred_scaler)) #0.0 100.0
	set.append(scale(talib.DX(HIGH, LOW, CLOSE, timeperiod=50), hundred_scaler)) #0.0 100.0
	set.append(scale(talib.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=14), hundred_scaler)) #-5.888410733172162e-08 100.00000000707982
	set.append(scale(talib.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=26), hundred_scaler)) #-1.3802451942902055e-09 100.00000001185656
	set.append(scale(talib.MFI(HIGH, LOW, CLOSE, VOLUME, timeperiod=100), hundred_scaler)) #-5.3901183535126315e-08 100.00000000091877
	set.append(scale(talib.MINUS_DI(HIGH, LOW, CLOSE, timeperiod=14), hundred_scaler)) #0.0 99.99999996020233
	set.append(scale(talib.PLUS_DI(HIGH, LOW, CLOSE, timeperiod=14), hundred_scaler)) #0.0 100.0
	set.append(scale(talib.RSI(CLOSE, timeperiod=14), hundred_scaler)) #0.0 100.0
	set.append(scale(talib.STOCH(HIGH, LOW, CLOSE, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0), hundred_scaler)) #-1.0137076363510762e-12 100.00000000000279
	set.append(scale(talib.STOCHF(HIGH, LOW, CLOSE, fastk_period=5, fastd_period=3, fastd_matype=0), hundred_scaler)) #-1.0137076363510762e-12 100.0000000000012
	set.append(scale(talib.STOCHRSI(CLOSE, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0), hundred_scaler)) #-1.2166564052525548e-12 100.00000000000011
	set.append(scale(talib.ULTOSC(HIGH, LOW, CLOSE, timeperiod1=7, timeperiod2=14, timeperiod3=28), hundred_scaler))  # 0.0 100.00000032481957
	set.append(scale(talib.CDL3WHITESOLDIERS(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLDOJI(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLDRAGONFLYDOJI(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLGRAVESTONEDOJI(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLHAMMER(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLHOMINGPIGEON(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLINVERTEDHAMMER(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLLADDERBOTTOM(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLLONGLEGGEDDOJI(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLMATCHINGLOW(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLMORNINGDOJISTAR(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLMORNINGSTAR(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLPIERCING(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLRICKSHAWMAN(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLSTICKSANDWICH(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLTAKURI(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100
	set.append(scale(talib.CDLUNIQUE3RIVER(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #0 100



	#Absolute Price Oscillator crossing above zero is considered bullish, while crossing below zero is bearish. A 
	#positive indicator value indicates an upward movement, while negative readings signal a downward trend.
	set.append(scale(talib.APO(CLOSE, fastperiod = 12, slowperiod = 26, matype = 1), thousand_scaler)) #-536.1910463572985 443.13971636041424
	set.append(scale(VOLUME, thousand_scaler))
	#The Commodity Channel Index (CCI) measures the current price level relative to an average price level over a given
	#period of time. CCI is relatively high when prices are far above their average.
	set.append(scale(talib.CCI(HIGH, LOW, CLOSE), thousand_scaler)) #-466.66671166042244 466.66673951370416
	#Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
	#between two moving averages of a security’s price.
	set.append(scale(talib.MACD(CLOSE, fastperiod=5, slowperiod=14, signalperiod=5), thousand_scaler))  # -536.1910463572985 443.13971636041424
	set.append(scale(talib.MACD(CLOSE, fastperiod=12, slowperiod=26, signalperiod=9), thousand_scaler)) #-536.1910463572985 443.13971636041424
	set.append(scale(talib.MACD(CLOSE, fastperiod=14, slowperiod=50, signalperiod=25), thousand_scaler))  # -536.1910463572985 443.13971636041424
	set.append(scale(talib.ATR(HIGH, LOW, CLOSE), thousand_scaler)) #0.0 672.1715311610562
	set.append(scale(talib.HT_DCPHASE(CLOSE), hundred_scaler)) #-44.99298332517037 314.99654478107004
	set.append(scale(talib.LINEARREG_SLOPE(CLOSE, timeperiod=14), hundred_scaler)) #-222.33604395604476 152.37032967033085
	set.append(scale(talib.STDDEV(CLOSE, timeperiod=5, nbdev=1), thousand_scaler)) #0.0 709.4023698851089



	set.append(scale(talib.MINUS_DM(HIGH, LOW, timeperiod=14), thousand_scaler)) #0.0 2909.3760999618785
	set.append(scale(talib.MOM(CLOSE, timeperiod=10), thousand_scaler)) #-2508.0 1711.2000000000007
	set.append(scale(talib.MOM(CLOSE, timeperiod=25), thousand_scaler))  # -2508.0 1711.2000000000007
	set.append(scale(talib.PLUS_DM(HIGH, LOW, timeperiod=14), thousand_scaler)) #0.0 3697.0008310558483
	set.append(scale(talib.ADOSC(HIGH, LOW, CLOSE, VOLUME), thousand_scaler)) #-1843.4418435977714 1237.4131984846026
	set.append(scale(talib.TRANGE(HIGH, LOW, CLOSE), thousand_scaler)) #0.0 4000.0
	set.append(scale(talib.HT_PHASOR(CLOSE), thousand_scaler)) #-2873.977625168652 3422.2535328187428



	#The Balance of Power indicator measures the market strength of buyers against sellers by assessing the ability of 
	#each side to drive prices to an extreme level. The calculation is: Balance of Power = (Close price – Open price) / 
	#(High price – Low price) The resulting value can be smoothed by a moving average.
	set.append(scale(talib.BOP(OPEN, HIGH, LOW, CLOSE), one_scaler)) #-1.0 1.0
	set.append(scale(talib.ROCP(CLOSE, timeperiod=10), one_scaler)) #-0.30688987999999995 0.46745909457773854
	set.append(scale(talib.ROCR(CLOSE, timeperiod=10), one_scaler)) #0.69311012 1.4674590945777386
	set.append(scale(talib.TRIX(CLOSE, timeperiod=30), one_scaler)) #-0.6044429731575707 0.434667877456385
	set.append(scale(talib.HT_SINE(CLOSE), one_scaler)) #-0.9999999999996187 0.9999999999940317
	set.append(scale(talib.HT_TRENDMODE(CLOSE), one_scaler)) #0 1



	set.append(scale(talib.PPO(CLOSE, fastperiod=12, slowperiod=26, matype=0), hundred_scaler)) #-13.640389725420714 13.383459677599681
	set.append(scale(talib.ROC(CLOSE, timeperiod=10), hundred_scaler)) #-30.688987999999995 46.74590945777386
	set.append(scale(talib.NATR(HIGH, LOW, CLOSE), one_scaler)) #0.0 7.881777549670427
	set.append(scale(talib.HT_DCPERIOD(CLOSE), hundred_scaler)) #6.91050087362864 49.99951053223339
	set.append(scale(talib.CORREL(HIGH, LOW, timeperiod=30), one_scaler)) #-2.4748737341529163 2.2135943621178655



	set.append(scale(talib.AD(HIGH, LOW, CLOSE, VOLUME), million_scaler)) #-3719.2404462314116 199644.25743563366
	set.append(scale(talib.OBV(CLOSE, VOLUME), million_scaler)) #-23849.75116020021 29139.83770172981
	set.append(scale(talib.BETA(HIGH, LOW, timeperiod=5), million_scaler)) #-2971957.111723269 1250567.1172035346
	set.append(scale(talib.VAR(CLOSE, timeperiod=5, nbdev=1), million_scaler)) #-1.4901161193847656e-07 503251.7223986089



	# The Aroon Oscillator is a trend-following indicator that uses aspects of the Aroon Indicator (Aroon Up and Aroon
	# Down) to gauge the strength of a current trend and the likelihood that it will continue. Readings above zero
	# indicate that an uptrend is present, while readings below zero indicate that a downtrend is present.
	set.append(scale(talib.AROONOSC(HIGH, LOW), hundred_scaler))  # -100.0 100.0
	# The Chande Momentum Oscillator (CMO) is a technical momentum indicator developed by Tushar Chande. The CMO
	# indicator is created by calculating the difference between the sum of all recent higher closes and the sum of all
	# recent lower closes and then dividing the result by the sum of all price movement over a given time period.
	set.append(scale(talib.CMO(CLOSE, timeperiod=5), hundred_scaler))  # -100.0 100.0
	set.append(scale(talib.CMO(CLOSE, timeperiod=14), hundred_scaler))  # -99.99999998652726 100.0
	set.append(scale(talib.CMO(CLOSE, timeperiod=25), hundred_scaler))  # -99.99854211548185 100.0
	set.append(scale(talib.CDL3INSIDE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDL3LINESTRIKE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDL3OUTSIDE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLABANDONEDBABY(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLBELTHOLD(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLBREAKAWAY(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLCLOSINGMARUBOZU(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLCOUNTERATTACK(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLDOJISTAR(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLENGULFING(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLGAPSIDESIDEWHITE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLHARAMI(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLHARAMICROSS(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLHIGHWAVE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLKICKING(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLKICKINGBYLENGTH(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLLONGLINE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLMARUBOZU(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLRISEFALL3METHODS(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLSEPARATINGLINES(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLSHORTLINE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLSPINNINGTOP(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLTASUKIGAP(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLTRISTAR(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.CDLXSIDEGAP3METHODS(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 100
	set.append(scale(talib.LINEARREG_ANGLE(CLOSE, timeperiod=14), hundred_scaler)) #-89.74230272272693 89.62397563202859



	set.append(scale(talib.WILLR(HIGH, LOW, CLOSE, timeperiod=14), hundred_scaler))  # -100.00000000000001 0.0
	#The Two Crows is a three-line bearish reversal candlestick pattern. The pattern requires confirmation, that is,
	#the following candles should break a trendline or the nearest support area which may be formed by the first
	#candle's line. If the pattern is not confirmed it may act only as a temporary pause within an uptrend.
	set.append(scale(talib.CDL2CROWS(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDL3BLACKCROWS(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLADVANCEBLOCK(OPEN, HIGH, LOW, CLOSE), hundred_scaler))  # -100 0
	set.append(scale(talib.CDLDARKCLOUDCOVER(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLEVENINGDOJISTAR(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLEVENINGSTAR(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLHANGINGMAN(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLIDENTICAL3CROWS(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLINNECK(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLONNECK(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLSHOOTINGSTAR(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLSTALLEDPATTERN(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0
	set.append(scale(talib.CDLTHRUSTING(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-100 0


	set.append(scale(talib.CDLHIKKAKE(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-200 200
	set.append(scale(talib.CDLHIKKAKEMOD(OPEN, HIGH, LOW, CLOSE), hundred_scaler)) #-200 200


	return np.concatenate(set)


if __name__ == '__main__':
	file = 'Kraken_Trading_History/XBTUSD_15min_candles.npy'
	try:
		candle_sticks = np.load(file)
	except FileNotFoundError:
		candle_sticks = calculate_candle_sticks()
		np.save(file, candle_sticks)

	file = 'Kraken_Trading_History/XBTUSD_15min_train_sets.npy'
	training_set = []
	try:
		training_set = np.load(file)
	except FileNotFoundError:
		lookback = 300
		for i in np.arange(candle_sticks.shape[0] - lookback):
			set = enhance_with_indicators(candle_sticks[i:i+lookback])
			training_set.append(set)
			if i % 1000 == 0 and i != 0:
				print("Step", i)
		training_set = np.asarray(training_set)
		np.save(file, training_set)

	scaling_groups = [ 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
				       2, 2, 3, 2, 2, 2, 1, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
				       3, 2, 2, 2, 2, 3, 3, 3, 2, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1,
				       2, 2, 4, 3, 4, 3, 1, 3, 2, 2, 3, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
				       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
				       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
				       2, 2, 2, 2, 2, 2, 4, 1, 0, 2, 0, 2, 3, 0, 4]

	#TODO use scaling groups to create min max scalers and then 
	for row in training_set:
		row = np.column_stack((scaling_groups, row))
		row.sort(axis=0)
		row = np.split(row[:, 1], np.unique(row[:, 0], return_index=True)[1][1:])
		print(row)

	pearson_matrix = np.corrcoef(np.transpose(training_set))

	fig, ax = plt.subplots()
	im = ax.imshow(pearson_matrix, cmap=plt.get_cmap('seismic'))
	# We want to show all ticks...
	ax.set_xticks(np.arange(169))
	ax.set_yticks(np.arange(169))
	for i, label in enumerate(ax.xaxis.get_ticklabels()):
		label.set_visible(i % 5 == 0)
	for i, label in enumerate(ax.yaxis.get_ticklabels()):
		label.set_visible(i % 5 == 0)

	heatmap = plt.pcolor(pearson_matrix, cmap=plt.get_cmap('seismic'))
	plt.colorbar(heatmap)
	fig.tight_layout()
	plt.show()


