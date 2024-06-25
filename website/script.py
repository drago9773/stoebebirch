import pandas as pd
import plotly.express as px
import json

file_path = 'C:\\Users\\jonat\\Documents\\stoebebirch\\washington_rentals.csv'
df = pd.read_csv(file_path)

fig = px.scatter(df, 
                 x='Longitude', 
                 y='Latitude', 
                 color='Max Rent Price',
                 hover_name='Address',
                 hover_data={
                     'City': True,
                     'State': True,
                     'Max Beds': True,
                     'Max Baths': True,
                     'Max Square Feet': True,
                     'Max Rent Price': ':.2f',
                     'Latitude': False,
                     'Longitude': False,
                     'Property ID': True
                 },
                 title='WA Rent Prices',
                 labels={'Max Rent Price': 'Rent Price ($)', 'Latitude': 'Latitude', 'Longitude': 'Longitude'},
                 color_continuous_scale='Viridis',
                 range_color=[0, 10000])

fig.update_layout(clickmode='event+select')
fig.add_annotation(text="Click on a point to highlight the property", 
                   xref="paper", yref="paper",
                   x=0, y=1, showarrow=False)

plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

htmx_attributes = '''
<script>
document.getElementById('plotly-div').on('plotly_click', function(data){
    var propertyId = data.points[0].customdata[0];
    htmx.trigger("#" + propertyId, 'click');
});
</script>
'''

final_html = plot_html + htmx_attributes

with open('C:\\Users\\jonat\\Documents\\stoebebirch\\website\\views\\plot.html', 'w') as f:
    f.write(final_html)
