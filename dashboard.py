import pandas as pd
from dash import Dash, Input, Output, dcc, html
import dash
from dash import Dash, html, Input, Output, ctx, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
#load of packages


data = (
    pd.read_csv("6classcsv.csv")
)
#plot of stars-default
px.scatter(data[data['Star type'] == 5], x='Temperature (K)', y='Luminosity(L/Lo)', color='Star type',
                     title='Clasificación de Estrellas en la Secuencia Principal',
                     labels={'Temperature (K)': 'Temperatura (K)', 'Luminosity(L/Lo)': 'Luminosidad (L/Lo)'},
                     hover_data=['Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class'])



#data settings for training the svm model
X = data[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']]
y = data['Star type']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluar el modelo
y_pred_train = svm_model.predict(X_train_scaled)
y_pred_test = svm_model.predict(X_test_scaled)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

#array with values of the star type
star_types = data['Star type'].sort_values().unique()



external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "assets/stylesheet",
    },
]

#running the app
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Principal Stars Sequences"

#layout of the app
app.layout = html.Div(
    children=[
        
        html.Div(
            children=[
                html.Img(src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f4ab.png", className="header-emoji"),
                html.H1(
                    children="Principal Stars Sequences", className="header-title"
                ),
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
)
,

            ],
            className="header",
        ),

#Paramters for prediction
        html.Div(
            children=[
               
                html.Div(
                    children=[
                        html.Div(children="Temperature (K)", className="menu-title"),
                        dcc.Input(
                            id="input-temp",
                            type="number",
                            step=1,
                            min = data["Temperature (K)"].min(),
                            max = 40000,
                            value=data["Temperature (K)"].min(),
                            required=True,
                            className="form-control mb-2",
                        ),
                    ]
                ),
                 html.Div(
                    children=[
                        html.Div(children="Luminosity (L/Lo):", className="menu-title"),
                        dcc.Input(
                            id="input-lum",
                            type="number",
                            step=0.1,
                            min = data["Luminosity(L/Lo)"].min(),
                            max = data["Luminosity(L/Lo)"].max(),
                            value=data["Luminosity(L/Lo)"].min(),
                            required=True,
                            className="form-control mb-2",
                        ),
                    ]
                ),
                 html.Div(
                    children=[
                        html.Div(children="Radius (R/Ro)", className="menu-title"),
                        dcc.Input(
                            id="input-radius",
                            type="number",
                            step=0.1,
                            min = data["Radius(R/Ro)"].min(),
                            max = data["Radius(R/Ro)"].max(),
                            value=data["Radius(R/Ro)"].min(),
                            required=True,
                            className="form-control mb-2",
                        ),
                    ]
                ),
                 html.Div(
                    children=[
                        html.Div(children="Absolute magnitude (Mv)", className="menu-title"),
                        dcc.Input(
                            id="input-mag",
                            type="number",
                            step=0.1,
                            max = data["Absolute magnitude(Mv)"].max(),
                            min = data["Absolute magnitude(Mv)"].min(),
                            value=data["Absolute magnitude(Mv)"].min(),
                            required=True,
                            className="form-control mb-2",
                        ),
                    ]
                )
                

                
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.H1( id="spec"
                    
                ),
                html.Div(
                    children = dcc.Graph(
                        id="stars-chart",
                        config={"displayModeBar": False},
                        figure={}
                    ),
                    className="card"
                )
            ],
            className="wrapper"
        ),

    ]
)



#callback for the stars
@app.callback(
    Output("stars-chart", "figure"),
    Output("spec", "children"),
    #Output("")
    Input("input-temp", "value"),
    Input("input-lum", "value"),
    #Input("type-filter", "value"),
    Input("input-radius", "value"),
    Input("input-mag", "value"),

)

def plot_chart(temperature, luminosity, radius, magnitude):

    
     
    
    #query the input data
    search_input_data = pd.DataFrame([[temperature, luminosity, radius, magnitude]], columns=['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)'])
    input_scaled = scaler.transform(search_input_data)

    
    #model classifies the star type
    star_type = svm_model.predict(input_scaled)[0]


        # Crear la anotación para la estrella consultada
    annotation = {
        'x': temperature,
        'y': luminosity,
        'xref': 'x',
        'yref': 'y',
        'text': f"Estrella Consultada<br>Tipo: {star_type}",
        'showarrow': True,
        'arrowhead': 2,
        'ax': 0,
        'ay': -40
    }
    
    
    stars_chart = px.scatter(data[data['Star type'] == star_type], x='Temperature (K)', y='Luminosity(L/Lo)', color='Star type',
                    title='Clasificación de Estrellas en la Secuencia Principal',
                    labels={'Temperature (K)': 'Temperatura (K)', 'Luminosity(L/Lo)': 'Luminosidad (L/Lo)'},
                    hover_data=['Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class'])

    spec = dbc.Alert(f"Tipo de estrella predicho: {star_type}", color='info')

    return stars_chart, spec





if __name__ == "__main__":
    app.run_server(debug=True)
