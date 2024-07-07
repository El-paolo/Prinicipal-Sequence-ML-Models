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
    pd.read_csv("avocado.csv")
    .assign(Date=lambda data: pd.to_datetime(data["Date"], format="%Y-%m-%d"))
    .sort_values(by="Date")
)
regions = data["region"].sort_values().unique()
avocado_types = data["type"].sort_values().unique()




data2 = (
    pd.read_csv("6classcsv.csv")
)
#plot of stars-default
px.scatter(data2[data2['Star type'] == 5], x='Temperature (K)', y='Luminosity(L/Lo)', color='Star type',
                     title='Clasificación de Estrellas en la Secuencia Principal',
                     labels={'Temperature (K)': 'Temperatura (K)', 'Luminosity(L/Lo)': 'Luminosidad (L/Lo)'},
                     hover_data=['Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class'])


# parametros

temperatures = data2["Temperature (K)"].sort_values().unique()




external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "assets/stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Principal Stars Sequences"

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
                "    Luminosity (L, Lo):"
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


        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Region", className="menu-title"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label": region, "value": region}
                                for region in regions
                            ],
                            value="Albany",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Type", className="menu-title"),
                        dcc.Dropdown(
                            id="type-filter",
                            options=[
                                {
                                    "label": avocado_type.title(),
                                    "value": avocado_type,
                                }
                                for avocado_type in avocado_types
                            ],
                            value="organic",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=data["Date"].min().date(),
                            max_date_allowed=data["Date"].max().date(),
                            start_date=data["Date"].min().date(),
                            end_date=data["Date"].max().date(),
                        ),
                    ]
                ),
                
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="price-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="volume-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)

@app.callback(
    Output("price-chart", "figure"),
    Output("volume-chart", "figure"),
    #Output("")
    Input("temperature-filter", "value"),
    Input("region-filter", "value"),g
    Input("type-filter", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_charts(region, avocado_type, start_date, end_date):
    filtered_data = data.query(
        "region == @region and type == @avocado_type"
        " and Date >= @start_date and Date <= @end_date"
    )
    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["AveragePrice"],
                "type": "lines",
                "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Average Price of Avocados",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"tickprefix": "$", "fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    volume_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Total Volume"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Avocados Sold", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return price_chart_figure, volume_chart_figure

if __name__ == "__main__":
    app.run_server(debug=True)
