# Python Libs.:
import os
import pathlib
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import dash
from dash import dcc
from dash import html
from plotly.graph_objects import *
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
import pickle

# Local Libs.:
from plot_data import deforestation_map, data_table


GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 1000 * 60 * 2)


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "HOLD - Title"

server = app.server

app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4(f"HOLD - TITLE", className="app__header__title"),
                        html.P(
                            "Hold - Descripttion",
                            className="app__header__title--grey",
                            style={"color": "white", "font-size": 14}
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("SOURCE CODE", className="link-button", style={"color": "white"}),
                            href="https://github.com/PedroFrias",
                        ),
                        html.A(
                            html.Button("LINKED.IN", className="link-button", style={"color": "white"}),
                            href="https://www.linkedin.com/in/pedro-henrique-abem-athar-frias-a48526119/",
                        ),
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                # live stocks
                html.Div(
                    [
                        html.Div(
                            [
                                html.H6(
                                    "ONDE ESTACIONAR?",
                                    className="graph__title",
                                )
                            ]
                        ),
                        html.Div(
                            children=deforestation_map(),
                            style={
                                'flex-grow': 1,
                                'height': '100%',
                                'width': '100%'
                            }
                        )
                    ],
                    className="two-thirds column live__stocks__container",
                ),

                html.Div(
                    [
                        # correlation
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "DADOS",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Div(
                                    children=data_table(),
                                    style={
                                        'flex-grow': 1,
                                        'height': '100%',
                                        'padding': '10px 10px 25px 25px'
                                    }
                                )
                            ],

                            className="graph__container first",
                            style={
                                'flex-grow': 1,
                                'height': '100%',
                                'width': '100%'
                            },
                        ),

                        # prediction
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


if __name__ == "__main__":
    app.run_server(debug=True)