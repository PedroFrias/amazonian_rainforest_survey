
from plotly.graph_objects import *
import pandas as pd
from dash import dcc
from dash import dash_table
from numpy import arange

ACESS_TOKEN = "pk.eyJ1IjoicGVkcm9mcmlhcyIsImEiOiJjbDR1c3ZjbDcxeHZvM2lxY2w2ZGo4bDl6In0.xQ9ZfVMnucLKh2MInZYwcg"


def deforestation_map():

    dataframe = pd.read_csv("data/dataframe.csv")
    colors_id = ["slateblue", "silver", "lightcoral"]
    colors = [colors_id[int(validation)] for validation in dataframe["Validation"].tolist()]

    return dcc.Graph(
                    id='deforestation_map-graph',
                    figure=Figure(
                        data=[
                            # Data for all rides based on date and time
                            Scattermapbox(
                                lat=dataframe["Lats"],
                                lon=dataframe["Lons"],
                                marker=dict(
                                    color=colors,
                                    size=6
                                ),
                            )
                        ],
                        layout=Layout(
                            paper_bgcolor='#343332',
                            plot_bgcolor='#343332',
                            height=1000,
                            autosize=True,
                            margin=layout.Margin(l=10, r=10, t=25, b=10),
                            showlegend=False,
                            mapbox=dict(
                                accesstoken=ACESS_TOKEN,
                                center=dict(lat=-3.09196, lon=-60.00402),
                                style="dark",
                                zoom=9,
                            )
                        )
                    )
                )


def data_table():
    sites_of_deforestation = pd.read_csv("data/dataframe.csv")
    sites_of_deforestation = sites_of_deforestation[sites_of_deforestation['Validation'] > 0]
    sites_of_deforestation.loc[:, "Validation"] = sites_of_deforestation["Validation"].map('{:.2f}'.format)

    return dash_table.DataTable(
        id="data-table",
        columns=
        [
            dict(
                name=col.upper(),
                id=col
            )
            for col in sites_of_deforestation.columns if col != "Date"
        ],
        data=sites_of_deforestation.to_dict("records"),
        page_current=0,
        page_size=29,

        style_header=dict(
            color="white",
            background="#343332"
        ),

        style_cell=dict(
            textAlign="left",
            width=25,
            font_size=15,
            border="#343332"
        ),

        style_data=dict(
            color="gainsboro",
            background="#343332"
        ),

        style_data_conditional=[
            {
                'if': {'column_id': 'Validation', 'filter_query': '{Validation} < 1'},
                'color': 'cornflowerblue',
                'fontWeight': 'bold'
            }
        ],

        style_as_list_view=True,
    )
