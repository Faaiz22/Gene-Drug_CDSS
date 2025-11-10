
import plotly.express as px
import pandas as pd
def scatter_2d(coords, meta_df, color_by='Association_Label', sample=5000):
    df = pd.DataFrame({'x': coords[:,0], 'y': coords[:,1]})
    df = pd.concat([df, meta_df.reset_index(drop=True)], axis=1)
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
    fig = px.scatter(df, x='x', y='y', color=color_by, hover_data=['Gene_ID','Drug_ID','Pred_Assoc'])
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    return fig
