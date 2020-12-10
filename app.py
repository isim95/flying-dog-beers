import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from datetime import datetime


df = pd.read_csv('https://raw.githubusercontent.com/isim95/flying-dog-beers/master/df.csv')
dfTA = pd.read_csv('https://raw.githubusercontent.com/isim95/flying-dog-beers/master/dfTA.csv')



df.replace([np.inf, -np.inf], np.nan)
df=df.fillna(df.mean())
df = df.round(2)

dfTA.replace([np.inf, -np.inf], np.nan)
dfTA=dfTA.fillna(dfTA.mean())
dfTA = dfTA.round(2)

ogdf = pd.read_csv('https://raw.githubusercontent.com/isim95/flying-dog-beers/master/df.csv')


ogdf1 = df
ogdf1.drop(['Date'],axis=1)
ogdf1 = ogdf1.join(ogdf, how = 'left', lsuffix = '_left', rsuffix = '')
ogdf1 = ogdf1.drop(['Date_left'],axis=1)
ogdf1.index = ogdf1['Date']


ogdfTA = pd.read_csv('https://raw.githubusercontent.com/isim95/flying-dog-beers/master/dfTA.csv')


ogdf1TA = dfTA
ogdf1TA.drop(['Date'],axis=1)
ogdf1TA = ogdf1TA.join(ogdf, how = 'left', lsuffix = '_left', rsuffix = '')
ogdf1TA = ogdf1TA.drop(['Date_left'],axis=1)
ogdf1TA.index = ogdf1['Date']


app = dash.Dash()
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
