import pandas as pd
import plotly.graph_objects as go

def generate_plot():
  pg = pd.read_csv('src/SL_vs_RL.csv').iloc[:, :2].set_index('Unnamed: 0').rename_axis('Trajectories').add_suffix('_PG')
  ppo = pd.read_csv('src/SL_vs_PPO.csv').iloc[:, :2].set_index('Unnamed: 0').rename_axis('Trajectories').add_suffix('_PPO')
  cs224r = pg.join(ppo)
  cs224r = cs224r.join(cs224r.rolling(7, 1).mean().add_suffix(' smoothed'))
  fig = go.Figure(layout=go.Layout(width=800, template='plotly_dark'))
  fig = fig.add_trace(go.Scatter(x=cs224r.index, y=cs224r['dIMP_PG'], name='PG', line={'color': clrs[0], 'width': 1.5}, opacity=.7))
  fig = fig.add_trace(go.Scatter(x=cs224r.index, y=cs224r['dIMP_PG smoothed'], name='PG<br>smoothed', line={'color': clrs[0], 'width': 2.5}, opacity=1))
  fig = fig.add_trace(go.Scatter(x=cs224r.index, y=cs224r['dIMP_PPO'], name='PPO + Baseline', line={'color': clrs[1], 'width': 1.5}, opacity=.7))
  fig = fig.add_trace(go.Scatter(x=cs224r.index, y=cs224r['dIMP_PPO smoothed'], name='PPO + Baseline<br>smoothed', line={'color': clrs[1], 'width': 2.5}, opacity=1))
  fig = fig.update_layout(title='IMP Differential of RL Agent vs BC', yaxis={'title': 'IMP Differential'}, xaxis={'title': 'Self-Play Games'})
  return fig

