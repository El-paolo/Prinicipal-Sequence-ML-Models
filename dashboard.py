import pandas as pd
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import State
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load data
data = pd.read_csv("6classcsv.csv")

# Plot initial scatter plot (this can be used later in the app)
initial_fig = px.scatter(
    data[data['Star type'] == 5], 
    x='Temperature (K)', 
    y='Luminosity(L/Lo)', 
    color='Star type',
    title='Clasificación de Estrellas en la Secuencia Principal',
    labels={'Temperature (K)': 'Temperatura (K)', 'Luminosity(L/Lo)': 'Luminosidad (L/Lo)'},
    hover_data=['Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class']
)

# Data settings for training the SVM model
X = data[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']]
y = data['Star type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_train = svm_model.predict(X_train_scaled)
y_pred_test = svm_model.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
    dbc.themes.BOOTSTRAP,
]

# Running the app
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Principal Stars Sequences"

# Layout of the app
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Img(src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f4ab.png", className="header-emoji"),
                html.H1(children="Principal Stars Sequences", className="header-title"),
                html.Div(
                    children=[
                        html.P(
                            children=(
                                "Stellar Analysis Machine Learning Tool"
                                "The Stellar Analysis Machine Learning Tool is designed to analyze and predict stellar properties using key astrophysical parameters. Ideal for astronomers and researchers, this tool provides insights into the characteristics and behavior of stars."
                            ),
                            className="header-description",
                        ),
                        html.P(
                            children=(
                                "Key Parameters:"
                                "    Temperature (K):"
                                "        Surface temperature in Kelvin, crucial for star classification and color."
                                "    Luminosity (L,  Lo):"
                                "        Brightness compared to the Sun, indicating energy output and life stage."
                                "    Radius (R/Ro):"
                                "        Star's radius as a multiple of the Sun's radius, affecting luminosity."
                                "    Absolute Magnitude:"
                                "        Intrinsic brightness at 10 parsecs, essential for true brightness comparison."
                                "    Star Type:"
                                "        Evolutionary stage (e.g., Main Sequence, Giant), indicating lifecycle."
                                "    Star Color:"
                                "        Perceived color correlating with temperature, aiding in spectral classification."
                                "    Spectral Class:"
                                "        Classification based on spectrum, indicating temperature and composition."
                            ),
                            className="header-description",
                        ),
                        html.P(
                            children=(
                                "Features:"
                                "    Predictive Analysis:"
                                "        Machine learning algorithms to predict stellar properties."
                                "    Visualization:"
                                "        Plots and H-R diagrams for visual relationships between properties."
                                "    Classification:"
                                "        Automatic spectral and evolutionary classification."
                                "    Comparative Analysis:"
                                "        Compare with known stars to identify unique features."
                                "    Data Integration:"
                                "        Integrates with astronomical databases for enhanced analysis."
                            ),
                            className="header-description",
                        ),
                        html.P(
                            children=(
                                "The Stellar Analysis Machine Learning Tool is essential for exploring and understanding the diverse properties of stars."
                            ),
                            className="header-description",
                        ),
                    ],
                ),
            ],
            className="header",
        ),
       
       dbc.Container([
    dbc.Tabs([
        dbc.Tab(label='Star Classification', tab_id='tab-1'),
        dbc.Tab(label='Model Information', tab_id='tab-2')
    ], id='tabs', active_tab='tab-1'),
    html.Div(id='content')
])
    ]
)
@app.callback(
    Output('content', 'children'),
    Input('tabs', 'active_tab')
)
def render_content(tab):
    if tab == 'tab-1':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Stars Classification", className='text-center my-4'),
                    dbc.Card(
                        dbc.CardBody([
                            html.Label("Temperature (K)  [1.500 K to 45.000 K]:"),
                            dcc.Input(id='input-temp', type='int', value=5000,min=1500, max=45000, step=100,className='form-control mb-2'),
                            html.Br(),
                            html.Label("Luminosity (L/Lo)  [10^-4 to 10^6]:"),
                            dcc.Input(id='input-lum', type='number', value=1, step=0.000001,max=1000000.0,min=0.0001, className='form-control mb-2'),
                            html.Label("Radius (R/Ro)  [10^-2 to 1000]:"),
                            dcc.Input(id='input-radius', type='number', value=1,min=0.01,max=1000.0,  step=0.001, className='form-control mb-2'),
                            html.Label("Absolute Magnitude (Mv) [-10 to +15]:"),
                            dcc.Input(id='input-mag', type='number', value=5, step=0.1, min=-10.0,max=15.0,className='form-control mb-2'),
                            html.Br(),
                            dbc.Button("Clasiffy", id='classify-button', color='primary', className='mt-2'),
                            html.Hr(),
                            html.Div(id='output-class', className='text-center mt-4')
                        ]),
                    )
                ], width=4),
                dbc.Col([
                    html.H3("Stars Scatter Plot", className='text-center my-4'),
                    dcc.Graph(id='star-scatter', figure={})
                ], width=8)
            ], className='mt-4')
        ])
    elif tab == 'tab-2':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H5("ML Model: Support Vector Machine (SVM)"),
                    html.P("The SVM model with a linear kernel is used for classification."),
                    html.P("The model was chosen for its effectiveness in high-dimensional classification problems."),
                    html.P(f"Training set accuracy: {train_accuracy:.2%}"),
                    html.P(f"Test set accuracy: {test_accuracy:.2%}")
                ], width=12)
            ], className='mt-4'),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='accuracy-graph',
                        figure=go.Figure(
                            data=[go.Bar(x=['Train Accuracy', 'Test Accuracy'], y=[train_accuracy, test_accuracy], text=[f'{train_accuracy:.2%}', f'{test_accuracy:.2%}'], textposition='auto')],
                            layout_title_text='Model accuracy'
                        )
                    )
                ], width=12)
            ], className='mt-4')
        ])

@app.callback(
    Output('output-class', 'children'),
    Output('star-scatter', 'figure'),
    Input('classify-button', 'n_clicks'),
    State('input-temp', 'value'),
    State('input-lum', 'value'),
    State('input-radius', 'value'),
    State('input-mag', 'value')
)
def classify_star(n_clicks, temp, lum, radius, mag):
    if n_clicks is None:
        return "", {}

    # Preprocesar los valores de entrada
    input_data = pd.DataFrame([[temp, lum, radius, mag]], columns=['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)'])
    input_scaled = scaler.transform(input_data)

    # Realizar la predicción
    star_type = svm_model.predict(input_scaled)[0]

    # Transformar el valor de luminosidad a escala logarítmica para la anotación
    log_lum = np.log10(lum)

    # Crear la anotación para la estrella consultada
    annotation = {
        'x': temp,
        'y': log_lum,
        'xref': 'x',
        'yref': 'y',
        'text': f" Star Type of Consulted Star: {star_type}",
        'showarrow': True,
        'arrowhead': 2,
        'ax': 0,
        'ay': -40
    }

    # Actualizar el gráfico de dispersión
    fig = px.scatter(data, x='Temperature (K)', y='Luminosity(L/Lo)', color='Star type',
                     title='Star Classification on The Principal Sequence',
                     labels={'Temperature (K)': 'Temperature (K)', 'Luminosity(L/Lo)': 'Luminosity (L/Lo)'},
                     hover_data=['Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class'])
    fig.update_layout(
        transition_duration=500,
        annotations=[annotation],
        xaxis_autorange='reversed',  # Invertir el eje x
        yaxis_type='log'  # Escala logarítmica para el eje y
    )

    return dbc.Alert(f" Star Type Predicted: {star_type}", color='info'), fig


# Índice de tipos de estrellas
index_content = dbc.Container([
    dbc.Card([
        dbc.CardBody([
            html.H3("Star Type Index", className="card-title mt-4"),
            html.P("Brown Dwarf -> Star Type = 0", className="card-text lead"),
            html.P("Red Dwarf -> Star Type = 1", className="card-text lead"),
            html.P("White Dwarf -> Star Type = 2", className="card-text lead"),
            html.P("Main Sequence -> Star Type = 3", className="card-text lead"),
            html.P("Supergiant -> Star Type = 4", className="card-text lead"),
            html.P("Hypergiant -> Star Type = 5", className="card-text lead"),
        ])
    ], className='mt-4')
])


# Agregar el índice al layout de la aplicación
app.layout.children.append(index_content)

# Ejecutar la aplicación
app.run_server()