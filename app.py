import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pandas as pd
import pickle

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data which is included with sklearn
from sklearn.datasets import load_iris
iris = load_iris()

########### Initiate the app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.title='knn'

########### Set up the layout
app.layout = html.Div(children=[
    html.H1('NATS GONNA WIN!!!!'),
    html.Div([
        html.H6('Sepal Length'),
            dcc.Slider(
                id='slider-1',
                min=1,
                max=8,
                step=0.1,
                marks={i:str(i) for i in range(1,9)},
                value=5,
            ),
        html.Br(),
        html.H6('Sepal Width'),
                    dcc.Slider(
                        id='slider-2',
                        min=1,
                        max=8,
                        step=0.1,
                        marks={i:str(i) for i in range(1,9)},
                        value=5,
                    ),
        html.Br(),
        html.H6('# of Neighbors'),
        dcc.Dropdown(
            id = 'k-drop',
            options=[{'label':5, 'value':5},
                    {'label':10, 'value':10},
                    {'label':15, 'value':15},
                    {'label':20, 'value':20},
                    {'label':25, 'value':25}
            ],
            # options = [{'label': i, 'value':i} for i in [5,10,15,20,25]],
            value=5
        )
    ]),
    html.Div(id='output-message', children=''),
    html.Br(),
    html.A('Code on Github', href='https://github.com/brandonmoore38/knn_iris_plotly'),
])

######## Callback go here
@app.callback(Output('output-message', 'children'),
                [Input('k-drop', 'value'),
                 Input('slider-1', 'value'),
                 Input('slider-2', 'value')
                ])
def my_funky_function(k, value0, value1):
    # read in the chosen model
    file = open(f'resources/model_k{k}.pkl', 'rb')
    model = pickle.load(file)
    file.close()
    # define the new observation from the chosen values
    new_obs=[[4.9, 2.7]]
    import numpy as np
    new_obs2=np.array([[4.9,2.7]])
    mymodel.predict(new_obs2)
    mymodel.kneighbors(new_obs)
    my_prediction = model.predict(new_obs)
    return f'you chose {k} and the predicted species number is: {my_prediction}'

# convert these values into a dataframe
df = pd.DataFrame(iris.data, columns=['sl', 'sw', 'pl', 'pw'])
df['species']=iris.target
df.head()

# establish the predictors and the target
X = df[['sl', 'pl']]
y = df['species']
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X , y,
                         test_size=0.3, random_state=52 )
print('length of y-test:', len(y_test))
# instantiate the classifier
mymodel = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
# fit on the training data
mymodel.fit(X_train, y_train)
# predict on the testing data
y_preds = mymodel.predict(X_test)
# evaluate the model performance
print('accuracy score: ', round(metrics.accuracy_score(y_test, y_preds),2))
# examine the confusion matrix
pd.DataFrame(metrics.confusion_matrix(y_test, y_preds))

### Create multiple KNN models and pickle for use in the plotly dash app.
for k in [5, 10, 15, 20, 25]:
    mymodel = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    mymodel.fit(X_train, y_train)
    y_preds = mymodel.predict(X_test)
    file = open(f'resources/model_k{k}.pkl','wb')
    pickle.dump(mymodel, file)
    file.close()

############ Execute the app
if __name__ == '__main__':
    app.run_server(debug=True)
