import plotly.express as px

# Create a plotly figure for Washington State map
fig = px.scatter_mapbox(lat=[47.7511], lon=[-120.7401], zoom=7)

# Update layout for better aesthetics and details
fig.update_layout(
    mapbox_style="carto-positron",  # Choose mapbox style for details like roads and addresses
    mapbox_zoom=7,  # Adjust zoom level
    mapbox_center={"lat": 47.7511, "lon": -120.7401},  # Center the map on Washington State
    margin={"r":0,"t":0,"l":0,"b":0},  # Remove unnecessary margins
    showlegend=False,  # Hide legend
)

# Update mapbox layers for road and address details
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

# Show the plot
fig.show()
