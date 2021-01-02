import numpy as np
import tensorflow

model = tensorflow.keras.models.load_model("modelmulti")
counter = 0
pos = []

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    set = []

    zero_trading_volume_mask = np.zeros(len(settings['markets']))

    for i, stock in enumerate(settings['markets']):
        if i == 0: # CASH
            continue
        ticker_set = np.column_stack((OPEN[:,i], HIGH[:,i], LOW[:,i], CLOSE[:,i]))

        ticker_set /= np.max(ticker_set)

        volume = VOL[:,i]
        volume_set = volume / np.nanmax(np.append(volume, 1.))

        stock_set = np.append(ticker_set, volume_set.astype('float32'))

        if not np.isnan(stock_set).any():
            set.append(stock_set)
        else:
            set.append(np.zeros(stock_set.shape))
            zero_trading_volume_mask[i] = 1

    set = np.asarray(set)

    global model
    #yhat = model.predict(np.expand_dims(set, axis=0)) #For single stock
    yhat = model.predict(set)
    yhat = (np.ma.masked_array(yhat, mask=zero_trading_volume_mask[1:], fill_value=0) - 1)
    yhat[yhat < 0.16] = 0
    yhat = np.insert(yhat.filled(), 0, 0.01) #Insert small cash holding

    global counter
    global pos

    if counter == 0:
        pos = yhat
        counter = 1
    else:
        counter = (counter + 1) % 30

    #print("Day " + str(counter) + " | Invested " + str(pos))

    return pos, settings


##### Do not change this function definition #####
def mySettings():
    '''Define your market list and other settings here.
    The function name "mySettings" should not be changed.
    Default settings are shown below.'''

    # Default competition and evaluation mySettings
    settings = {}

    #settings['markets'] = ['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG']


    #S&P 100 stocks
    settings['markets']=['AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    """
    # Futures Contracts
    settings['markets'] = ['CASH', 'F_AD', 'F_AE', 'F_AH', 'F_AX', 'F_BC', 'F_BG', 'F_BO', 'F_BP', 'F_C',  'F_CA',
                           'F_CC', 'F_CD', 'F_CF', 'F_CL', 'F_CT', 'F_DL', 'F_DM', 'F_DT', 'F_DX', 'F_DZ', 'F_EB',
                           'F_EC', 'F_ED', 'F_ES', 'F_F',  'F_FB', 'F_FC', 'F_FL', 'F_FM', 'F_FP', 'F_FV', 'F_FY',
                           'F_GC', 'F_GD', 'F_GS', 'F_GX', 'F_HG', 'F_HO', 'F_HP', 'F_JY', 'F_KC', 'F_LB', 'F_LC',
                           'F_LN', 'F_LQ', 'F_LR', 'F_LU', 'F_LX', 'F_MD', 'F_MP', 'F_ND', 'F_NG', 'F_NQ', 'F_NR',
                           'F_NY', 'F_O',  'F_OJ', 'F_PA', 'F_PL', 'F_PQ', 'F_RB', 'F_RF', 'F_RP', 'F_RR', 'F_RU',
                           'F_RY', 'F_S',  'F_SB', 'F_SF', 'F_SH', 'F_SI', 'F_SM', 'F_SS', 'F_SX', 'F_TR', 'F_TU',
                           'F_TY', 'F_UB', 'F_US', 'F_UZ', 'F_VF', 'F_VT', 'F_VW', 'F_VX',  'F_W', 'F_XX', 'F_YM',
                           'F_ZQ']
    """

    settings['lookback'] = 300
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['participation'] = 1

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)