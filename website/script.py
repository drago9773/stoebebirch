import pandas as pd
import plotly.express as px
import json

# Read the CSV file
file_path = 'C:\\Users\\jonat\\Documents\\stoebebirch\\washington_rentals.csv'
df = pd.read_csv(file_path)

# Create a scatter plot
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
                     'Property ID': True  # Include Property ID in the hover data
                 },
                 title='WA Rent Prices',
                 labels={'Max Rent Price': 'Rent Price ($)', 'Latitude': 'Latitude', 'Longitude': 'Longitude'},
                 color_continuous_scale='Viridis',
                 range_color=[0, 10000])

# Add JavaScript code to handle point clicks
fig.update_layout(clickmode='event+select')
fig.add_annotation(text="Click on a point to highlight the property", 
                   xref="paper", yref="paper",
                   x=0, y=1, showarrow=False)

# Save the plot as HTML with htmx attributes
plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# Add custom attributes for htmx interaction
htmx_attributes = '''
<script>
document.getElementById('plotly-div').on('plotly_click', function(data){
    var propertyId = data.points[0].customdata[0];
    htmx.trigger("#" + propertyId, 'click');
});
</script>
'''

# Combine plot HTML and htmx attributes
final_html = plot_html + htmx_attributes

# Write the HTML content to the file
with open('C:\\Users\\jonat\\Documents\\stoebebirch\\website\\views\\plot.html', 'w') as f:
    f.write(final_html)
