"""
Lightweight chart loader for pre-rendered weather classification visualizations
"""
import json
import altair as alt
import os

def load_weather_chart(visual_type="confusion_matrix"):
    """
    Load pre-saved weather classification charts without running the model
    
    Parameters:
    visual_type (str): Type of visual to load
                      - "confusion_matrix": Confusion matrix heatmap
                      - "feature_importance": Feature importance bar chart  
                      - "geographic_performance": Map of city performance
    
    Returns:
    altair.Chart: The requested pre-rendered visualization
    """
    # Use absolute path to ensure Quarto can find the files
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chart_file = os.path.join(current_dir, "charts", f"{visual_type}.json")
    
    if os.path.exists(chart_file):
        try:
            with open(chart_file, 'r') as f:
                chart_dict = json.load(f)
            chart = alt.Chart.from_dict(chart_dict)
            # Ensure the chart is properly configured for display
            return chart.resolve_scale(color='independent')
        except Exception as e:
            return create_placeholder_chart(visual_type, f"Error: {str(e)}")
    else:
        return create_placeholder_chart(visual_type, "Chart file not found")

def create_placeholder_chart(visual_type, error_msg="Chart not available"):
    """Create a simple placeholder chart if pre-saved chart is not available"""
    import pandas as pd
    
    placeholder_data = pd.DataFrame({
        'x': [2],
        'y': [2], 
        'message': [f'{visual_type}:\n{error_msg}']
    })
    
    return alt.Chart(placeholder_data).mark_text(
        align='center',
        baseline='middle',
        fontSize=14,
        color='red'
    ).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, 4]), axis=None),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[0, 4]), axis=None),
        text='message:N'
    ).properties(
        width=400,
        height=300,
        title=f'{visual_type.replace("_", " ").title()} - Loading...'
    ).resolve_scale(color='independent')