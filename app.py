import numpy as np
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


# In[13]:


df = pd.read_csv('/Users/isaacsimons/Downloads/df.csv')
dfTA = pd.read_csv('/Users/isaacsimons/Downloads/dfTA.csv')


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
