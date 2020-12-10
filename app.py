import numpy as np
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from datetime import datetime

### Clean Data ###

df = pd.read_csv('/Users/isaacsimons/Downloads/btc_historical.csv')
df['Open'] = df['Open'].str.replace(',', '')
df['High'] = df['High'].str.replace(',', '')
df['Low'] = df['Low'].str.replace(',', '')
df['Close'] = df['Close'].str.replace(',', '')
df['Date'] = df['Date'].str.replace('/','')
del df['Vol.']
del df['Change %']
del df['Price']
df['Date'] = df['Date'].astype(float)
df['High'] = df['High'].astype(float)
df['Open'] = df['Open'].astype(float)
df['Low'] = df['Low'].astype(float)


dfTA = pd.read_csv('/Users/isaacsimons/Downloads/btc_historical.csv')
dfTA['Open'] = dfTA['Open'].str.replace(',', '')
dfTA['High'] = dfTA['High'].str.replace(',', '')
dfTA['Low'] = dfTA['Low'].str.replace(',', '')
dfTA['Close'] = dfTA['Close'].str.replace(',', '')
dfTA['Date'] = dfTA['Date'].str.replace('/','')
del dfTA['Vol.']
del dfTA['Change %']
del dfTA['Price']
dfTA['Date'] = dfTA['Date'].astype(float)
dfTA['High'] = dfTA['High'].astype(float)
dfTA['Open'] = dfTA['Open'].astype(float)
dfTA['Low'] = dfTA['Low'].astype(float)


# In[80]:


### Calculate Technical Analysis Indicators & Create DataFrames ###

# OVERLAP STUDIES INDICATORS #
df['MA'] = ta.SMA(df['Open'],30) # Simple Moving Average
df['DEMA'] = ta.DEMA(df['Open'],30) # Double Exponential Moving Average
df['EMA'] = ta.EMA(df['Open'],30) # Exponential Moving Average
df['HT_TRENDLINE'] = ta.HT_TRENDLINE(df['Open']) # Hilbert Transformation
df['KAMA'] = ta.KAMA(df['Open'],30) # Kaufman Adaptive Moving Average
df['MAMA'], df['FAMA'] = ta.MAMA(df['Open']) # MESA Adapative Moving Average
df['MIDPOINT'] = ta.MIDPOINT(df['Open'],14) # Midpoint Over Period
df['SAR'] = ta.SAR(df['High'],df['Low'],14) # Parabolic SAR
df['SAREXT'] = ta.SAREXT(df['High'],df['Low']) # Parabolic SAR - Extended
df['T3'] = ta.T3(df['Open'],5) # Triple Exponential Moving Average
df['TRIMA'] = ta.TRIMA(df['Open'],30) # Triangular Moving Average
df['BBAND UPPER'], df['BBAND MIDDLE'], df['BBAND LOWER'] = ta.BBANDS(df['Open'],30) # Bollinger Bands

# CYCLE INDICATORS #
df['HT_DCPERIOD'] = ta.HT_DCPERIOD(df['Open']) # Hilbert Transform Dominant Cycle Period
df['INPHASE'], df['QUADRATURE'] = ta.HT_PHASOR(df['Open']) # Hilbert Transform Phasor Components
df['SINE'], df['LEADSINE'] = ta.HT_SINE(df['Open']) # Hilbert Transform SineWave
df['HT_TRENDMODE'] = ta.HT_TRENDMODE(df['Open']) # Hilbert Transform Trend

# PRICE TRANSFORM FUNCTIONS #
df['MED PRICE'] = ta.MEDPRICE(df['High'],df['Low']) # Median Price
df['TYP PRICE'] = ta.TYPPRICE(df['High'],df['Low'],df['Open']) # Typical Price
df['BETA'] = ta.BETA(df['High'],df['Low'],5) # Beta

# STATISTICAL INDICATORS #
df['CORRELATION'] = ta.CORREL(df['High'],df['Low'],30) # Pearson Correlation Coefficient (r)
df['LINEAR REGRESS'] = ta.LINEARREG(df['Open'],14) # Linear Regression
df['ANGLE REGRESS'] = ta.LINEARREG_ANGLE(df['Open'],14) # Linear Regression Angle
df['LINREG INT'] = ta.LINEARREG_INTERCEPT(df['Open'],14) # Linear Regression Intercept
df['LINREG SLP'] = ta.LINEARREG_SLOPE(df['Open'],14) # Linear Regression Slope
df['STD DEV'] = ta.STDDEV(df['Open'],14,nbdev=1) # Standard Deviation
df['TIME SERIES FORECAST'] = ta.TSF(df['Open'],14) # Time Series
df['VARIANCE'] = ta.VAR(df['Open'],14,nbdev=1) # Variance

# VOLATILITY INDICATORS #
df['AVERAGE TRUE RANGE'] = ta.ATR(df['High'],df['Low'],df['Open'],14) # Average True Range
df['NORMALIZED ATR'] = ta.NATR(df['High'],df['Low'],df['Open'],14) # Normalized Average True Range
df['TRUE RANGE'] = ta.TRANGE(df['High'],df['Low'],df['Open']) # True Range

# MATH OPERATOR FUNCTIONS #
df['ADD'] = ta.ADD(df['High'],df['Low']) # High + Low
df['DIVIDE'] = ta.DIV(df['High'],df['Low']) # High / Low
df['MAX'] = ta.MAX(df['Open'],30) # Maximum Value in Period
df['MIN'] = ta.MIN(df['Open'],30) # Minimum Value in Period
df['SUBTRACT'] = ta.SUB(df['High'],df['Low']) # High - Low
df['SUM'] = ta.SUM(df['Open'],30) # Sum of Opens in Period
df['MULTIPLY'] = ta.MULT(df['High'],df['Low']) # High * Low

# MOMENTUM INDICATORS #
df['ADX'] = ta.ADX(df['High'],df['Low'],df['Open']) # Average Directional Movement Index
df['ADXR'] = ta.ADXR(df['High'],df['Low'],df['Open']) # Average Directional Movement Index Rating
df['APO'] = ta.APO(df['Open'],12,26,0) # Absolute Price Oscillator
df['AROON DOWN'], df['AROON UP'] = ta.AROON(df['High'],df['Low'],14) # Aroon Indicator Up / Down
df['AROON OSCILLATOR'] = ta.AROONOSC(df['High'],df['Low'],14) # Aroon Oscillator
df['CCI'] = ta.CCI(df['High'],df['Low'],df['Open'],14) # Commodity Channel Index
df['CMO'] = ta.CMO(df['Open'],14) # Chande Momentum Oscillator
df['DX'] = ta.DX(df['High'],df['Low'],df['Open']) # Directional Movement Index
df['MACD'], df['MACD SIGNAL'], df['MACD HIST'] = ta.MACD(df['Open'],12,26,9) # Moving Average Convergence / Divergence
df['MINUS_DI'] = ta.MINUS_DI(df['High'],df['Low'],df['Open'],14) # Minus Directional Indicator
df['MINUS_DM'] = ta.MINUS_DM(df['High'],df['Low'],14) # Minus Directional Movement Indicator
df['MOM'] = ta.MOM(df['Open'],10) # Momentum Indicator
df['PLUS_DI'] = ta.PLUS_DI(df['High'],df['Low'],df['Open'],14) # Plus Directional Indicator
df['PLUS_DM'] = ta.PLUS_DM(df['High'],df['Low'],14) # Plus Directional Movement Indicator
df['PPO'] = ta.PPO(df['Open'],12,26,0) # Percentage Price Oscillator
df['ROC'] = ta.ROC(df['Open'],10) # Rate of Change Indicator
df['ROCP'] = ta.ROCP(df['Open'],10) # Rate of Change Percentage Indicator
df['ROCR'] = ta.ROCR(df['Open'],10) # Rate of Change Ratio Indicator
df['ROCR 100'] = ta.ROCR100(df['Open'],10) # Rate of Change Ratio Indicator (100 Scale)
df['RSI'] = ta.RSI(df['Open'],14) # Relative Strength Index
df['SLOWK'], df['SLOWD'] = ta.STOCH(df['High'],df['Low'],df['Open'],5,3,0,3,0) # Slow Stochastic Indicator
df['TRIX'] = ta.TRIX(df['Open'],30) # One Day Rate of Change of a Triple Smooth EMA
df['ULT OSC'] = ta.ULTOSC(df['High'],df['Low'],df['Open'],7,14,28) # Ultimate Oscillator
df['WILLR'] = ta.WILLR(df['High'],df['Low'],df['Open'],14) # Willimas' %R

# PATTERN RECOGNITION INDICATORS #
dfTA['2 CROWS'] = ta.CDL2CROWS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['3 BLACK CROWS'] = ta.CDL3BLACKCROWS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['3 INSIDE UP/DOWN'] = ta.CDL3INSIDE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['3 LINE STRIKE'] = ta.CDL3LINESTRIKE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['3 OUTSIDE UP/DOWN'] = ta.CDL3OUTSIDE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['3 SOUTH STARS'] = ta.CDL3STARSINSOUTH(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['3 WHITE SOLDIERS'] = ta.CDL3WHITESOLDIERS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['ABANDONED BABY'] = ta.CDLABANDONEDBABY(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'],0)
dfTA['ADVANCE BLOCK'] = ta.CDLADVANCEBLOCK(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['BELT HOLD'] = ta.CDLBELTHOLD(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['BREAKAWAY'] = ta.CDLBREAKAWAY(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['CLOSING MARUBOZU'] = ta.CDLCLOSINGMARUBOZU(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['CONCEALING BABY SWALLOW'] = ta.CDLCONCEALBABYSWALL(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['COUNTER ATTACK'] = ta.CDLCOUNTERATTACK(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['DARK CLOUD COVER'] = ta.CDLDARKCLOUDCOVER(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'],0)
dfTA['DOJI'] = ta.CDLDOJI(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['DOJI STAR'] = ta.CDLDOJISTAR(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['DRAGONFLY DOJI'] = ta.CDLDRAGONFLYDOJI(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['ENGULFER'] = ta.CDLENGULFING(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['EVENING DOJI STAR'] = ta.CDLEVENINGDOJISTAR(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'],0)
dfTA['EVENING STAR'] = ta.CDLEVENINGSTAR(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'],0)
dfTA['GAP SIDE WHITE'] = ta.CDLGAPSIDESIDEWHITE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['GRAVESTONE DOJI'] = ta.CDLGRAVESTONEDOJI(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['HAMMER'] = ta.CDLHAMMER(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['HANGING MAN'] = ta.CDLHANGINGMAN(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['HARAMI'] = ta.CDLHARAMI(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['HARAMI CROSS'] = ta.CDLHARAMICROSS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['HIGH WAVE'] = ta.CDLHIGHWAVE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['HIKKAKE'] = ta.CDLHIKKAKE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['MODIFIED HIKKAKE'] = ta.CDLHIKKAKEMOD(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['HOMING PIGEON'] = ta.CDLHOMINGPIGEON(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['IDENTICAL 3 CROWS'] = ta.CDLIDENTICAL3CROWS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['IN NECK'] = ta.CDLINNECK(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['INVERTED HAMMER'] = ta.CDLINVERTEDHAMMER(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['KICKING'] = ta.CDLKICKING(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['KICKING BY LENGTH'] = ta.CDLKICKINGBYLENGTH(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['LADDER BOTTOM'] = ta.CDLLADDERBOTTOM(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['LONG LEGGED DOJI'] = ta.CDLLONGLEGGEDDOJI(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['LONG LINE'] = ta.CDLLONGLINE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['MARUBOZU'] = ta.CDLMARUBOZU(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['MATCHING LOW'] = ta.CDLMATCHINGLOW(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['MAT HOLD'] = ta.CDLMATHOLD(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'],0)
dfTA['MORNING DOJI STAR'] = ta.CDLMORNINGDOJISTAR(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'],0)
dfTA['MORNING STAR'] = ta.CDLMORNINGSTAR(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'],0)
dfTA['ON NECK'] = ta.CDLONNECK(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['PIERCING'] = ta.CDLPIERCING(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['RICKSHAW MAN'] = ta.CDLRICKSHAWMAN(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['RISING FALLING 3'] = ta.CDLRISEFALL3METHODS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['SEPARATING LINES'] = ta.CDLSEPARATINGLINES(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['SHOOTING STAR'] = ta.CDLSHOOTINGSTAR(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['SHORT LINE'] = ta.CDLSHORTLINE(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['SPINNING TOP'] = ta.CDLSPINNINGTOP(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['STALLED'] = ta.CDLSTALLEDPATTERN(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['STICK SANDWICH'] = ta.CDLSTICKSANDWICH(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['TAKURI'] = ta.CDLTAKURI(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['TASUKI GAP'] = ta.CDLTASUKIGAP(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['THRUSTING'] = ta.CDLTHRUSTING(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['TRI STAR'] = ta.CDLTRISTAR(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['UNIQUE 3 RIVER'] = ta.CDLUNIQUE3RIVER(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['UPSIDE GAP 2 CROWS'] = ta.CDLUPSIDEGAP2CROWS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])
dfTA['XGAP 3'] = ta.CDLXSIDEGAP3METHODS(dfTA['Open'],dfTA['High'],dfTA['Low'],dfTA['Close'])


# In[5]:


df.replace([np.inf, -np.inf], np.nan)
df=df.fillna(df.mean())
df = df.round(2)


# In[6]:





# In[7]:


dfTA.replace([np.inf, -np.inf], np.nan)
dfTA=dfTA.fillna(dfTA.mean())
dfTA = dfTA.round(2)


# In[14]:


X = df.drop('Open',axis=1)

y = df['Open']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rforest = RandomForestRegressor(1000)
rforest.fit(X_train,y_train)


# In[15]:


predictions = rforest.predict(X_test)
comparison = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})
comparison


# In[16]:


rforest.score(X,y)


# In[17]:


Z = dfTA.drop('Open',axis=1)
q = dfTA['Open']

Z_train, Z_test, q_train, q_test = train_test_split(Z, q, test_size=0.4, random_state=29)

rforestTA = RandomForestRegressor(1000)
rforestTA.fit(Z_train,q_train)


# In[18]:


predictionsTA = rforestTA.predict(Z_test)
comparisonTA = pd.DataFrame({'Actual':q_test,'Predicted':predictionsTA})
comparisonTA


# In[19]:


rforestTA.score(Z,q)


# In[20]:


ogdf = pd.read_csv('/Users/isaacsimons/Downloads/btc_historical.csv')
del ogdf['Price']
del ogdf['Open']
del ogdf['High']
del ogdf['Low']
del ogdf['Vol.']
del ogdf['Change %']
del ogdf['Close']

ogdf1 = df
ogdf1.drop(['Date'],axis=1)
ogdf1 = ogdf1.join(ogdf, how = 'left', lsuffix = '_left', rsuffix = '')
ogdf1 = ogdf1.drop(['Date_left'],axis=1)
ogdf1.index = ogdf1['Date']


# In[21]:


ogdfTA = pd.read_csv('/Users/isaacsimons/Downloads/btc_historical.csv')
del ogdfTA['Price']
del ogdfTA['Open']
del ogdfTA['High']
del ogdfTA['Low']
del ogdfTA['Vol.']
del ogdfTA['Change %']
del ogdfTA['Close']

ogdf1TA = dfTA
ogdf1TA.drop(['Date'],axis=1)
ogdf1TA = ogdf1TA.join(ogdf, how = 'left', lsuffix = '_left', rsuffix = '')
ogdf1TA = ogdf1TA.drop(['Date_left'],axis=1)
ogdf1TA.index = ogdf1['Date']


# In[29]:


app.layout = html.Div([
    html.H1("Bitcoin Price & Analysis"),
    html.P(("ECON 328 Final Project"), 
    style = {'padding' : '20px' , 
                'backgroundColor' : '#3aaab2'}),
    dcc.Checklist(
            id='toggle-rangeslider',
            options=[{'label': 'Include Rangeslider', 
                      'value': 'slider'}],
            value=['slider']
        ),
    dcc.Graph(id="graph"),
    dcc.Dropdown(
        id="indicator1",
        options=[{"label": x, "value": x} 
                 for x in ogdf1.columns[1:]],
        value=ogdf1.columns[1],
        clearable=False,
    ),
    dcc.Dropdown(
        id="indicator2",
        options=[{"label": x, "value": x} 
                 for x in ogdf1.columns[1:]],
        value=ogdf1.columns[1],
        clearable=False,
    ),
    dcc.Dropdown(
        id="indicator3",
        options=[{"label": x, "value": x} 
                 for x in ogdf1.columns[1:]],
        value=ogdf1.columns[1],
        clearable=False,
    ),
    dcc.Graph(id="line")
    ])


@app.callback(
     Output("graph", "figure"),
    [Input("toggle-rangeslider", "value")])

def display_candlestick(value):
    fig = go.Figure(go.Candlestick(
        x=ogdf1['Date'],
        open=ogdf1['Open'],
        high=ogdf1['High'],
        low=ogdf1['Low'],
        close=ogdf1['Close']
    ))
    
    fig.update_layout(
        xaxis_rangeslider_visible='slider' in value
    )
    
    fig['layout']['xaxis']['autorange'] = "reversed"
    
    return fig



@app.callback(
     Output("line","figure"),
    [Input("indicator1", "value"),
     Input("indicator2", 'value'),
     Input("indicator3", 'value')])


def display_time_series(indicator1,indicator2,indicator3):
    fig1 = px.line(ogdf1, x='Date', y=[indicator1,indicator2,indicator3])
    fig1['layout']['xaxis']['autorange'] = "reversed"
    return fig1


if __name__ == '__main__':
    app.run_server()
