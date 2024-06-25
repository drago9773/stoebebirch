import plotly.express as px

# WA state
fig = px.scatter_mapbox(lat=[47.7511], lon=[-120.7401], zoom=7)

fig.update_layout(
    mapbox_style="carto-positron",  # mapbox style, need to change potentially
    mapbox_zoom=7,
    mapbox_center={"lat": 47, "lon": -120},  # center on WA
    margin={"r":0,"t":0,"l":0,"b":0}, 
    showlegend=False, 
)

fig.update_layout(
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "OpenStreetMap",
            "source": [
                "https://openstreetmap.org",
            ],
        }
      ])

fig.show()
