import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle

########### Initiate the app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.title='knn'

########### Set up the layout

app.layout = html.Div(children=[
    html.H1('Classification of Iris Flowers'),
    html.Div([
        html.Div([
            html.Div([
                html.H6('Sepal Length'),
                dcc.Slider(
                    id='slider-1',
                    min=1,
                    max=8,
                    step=0.1,
                    marks={i:str(i) for i in range(1, 9)},
                    value=5
                ),
            html.Br(),
            ], className='four columns'),
            html.Div([
                html.H6('Petal Length'),
                dcc.Slider(
                    id='slider-2',
                    min=1,
                    max=8,
                    step=0.1,
                    marks={i:str(i) for i in range(1, 9)},
                    value=5
                ),
            html.Br(),
            ], className='four columns'),
            html.Div([
                html.H6('# of Neighbors:'),
                dcc.Dropdown(
                    id='k-drop',
                    options=[{'label': i, 'value': i} for i in [5,10,15,20,25]],
                    value=5
                ),
            html.Br(),
            ], className='four columns'),
        ], className='twelve columns'),
        html.Div([
            html.H6(id='message'),
        ], className='twelve columns'),
    html.Br(),
    html.A('Code on Github', href='https://github.com/austinlasseter/knn_iris_plotly'),
    ])
])



############ Execute the app
if __name__ == '__main__':
    app.run_server()
