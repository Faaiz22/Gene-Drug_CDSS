
import plotly.express as px
def docking_scatter(df, x='Docking_Energy', y='Final_Integrated_Score', color='QED'):
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=['SMILES','Pred_Assoc'])
    return fig
