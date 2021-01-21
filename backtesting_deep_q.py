import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("modeldeepq2")

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ticker_set = np.column_stack((OPEN, HIGH, LOW, CLOSE))
    ticker_set /= np.max(ticker_set)
    volume_set = VOL / np.nanmax(np.append(VOL, 1.))
    stock_set = np.column_stack((ticker_set.reshape(settings['lookback'], 4), volume_set.astype('float32')))

    global model
    yhat = model(np.expand_dims(stock_set, axis=0), training=False)
    action = tf.argmax(yhat[0]).numpy()

    pos = 0
    if action == 0:
        # short fully
        pos = -1
    elif action == 1:
        # hold
        pos = pos
    elif action == 2:
        # long fully
        pos = 1

    return np.asarray([pos]), settings


##### Do not change this function definition #####
def mySettings():
    '''Define your market list and other settings here.
    The function name "mySettings" should not be changed.
    Default settings are shown below.'''

    # Default competition and evaluation mySettings
    settings = {}

    settings['markets'] = ['AAPL']
    #settings['markets'] = ["CASH", "XOM", "MSFT", "AAPL", "WMT", "BRK.B", "GE", "T", "JNJ", "JPM", "C"]
    #settings['markets'] = ["CASH", "GOOGL", "CAT", "ABBV", "ORCL", "UPS"]
    """
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

    settings['lookback'] = 32
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['participation'] = 1

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)