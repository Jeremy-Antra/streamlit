import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

st.write("""
# ESG For Everyone - Challenge 2
""")

df = pd.read_csv("ESGData13-22.csv")
lst = df['I1: Company Name'].unique()
com = st.selectbox(
    'Select the company you wish to view',
    lst)
vdf = df[df['I1: Company Name'] == com]

st.write("""
## Latest Metrics
""")
esg_last_two = vdf['ESG_Score'].to_list()[-2:]
revenue_last_two = vdf['I3: Revenue'].to_list()[-2:]
stock_last_two = vdf["Stock Price: Average, Min, Max"].to_list()[-2:]
stock_avg_last_two = [float(i) for a in stock_last_two for i in a[1:-1].split(',')]
col1, col2, col3, col4 = st.columns(4)
col1.metric("ESG Score", esg_last_two[1], round(esg_last_two[1] - esg_last_two[0], 2))
col2.metric("Revenue", "$" + str(round(revenue_last_two[1] / 1000000, 2)) + "M",
            str(round((revenue_last_two[1] - revenue_last_two[0]) / revenue_last_two[0], 2)) + "%")
col3.metric("Stock Price YTD", "$" + str(stock_avg_last_two[3]),
            round(stock_avg_last_two[3] - stock_avg_last_two[0], 2))
mean_df = vdf.groupby('I4: Year')['ESG_Score'].mean()
total_mean = mean_df.mean()
min_score, max_score = mean_df.min(), mean_df.max()
means = mean_df.to_list()[-2:]
anti_risk = 100 - (int(abs(total_mean - means[1]) / (max_score - min_score) * 100))
delta_risk = anti_risk - (100 - (int(abs(total_mean - means[0]) / (max_score - min_score) * 100)))
col4.metric("ESG Anti-Risk Level", str(anti_risk) + "%",
            str(delta_risk) + "%")

st.write("""
## Annual Performance vs ESG Score
""")

# Plotting

tab1, tab2 = st.tabs(["Revenue", "Average Stock Price"])
with tab1:
    fig = go.Figure()

    # First trace for Revenue
    fig.add_trace(go.Scatter(x=vdf['I4: Year'], y=vdf['I3: Revenue'], mode='lines+markers',
                             name='Revenue', line=dict(color='blue')))

    # Second trace for ESG Score
    fig.add_trace(go.Scatter(x=vdf['I4: Year'], y=vdf['ESG_Score'], mode='lines+markers', name='ESG Score',
                             yaxis='y2', line=dict(color='orange')))

    # Create a layout with two y-axes
    fig.update_layout(
        title={
            'text': f'{com} Revenue and ESG Scores Over Years',
            'x': 0.3  # Centered title
        },
        xaxis=dict(title='Year'),
        yaxis=dict(title='Revenue', side='left', showgrid=False),
        yaxis2=dict(title='ESG Score', side='right', overlaying='y', showgrid=False),
        legend=dict(x=0, y=1.2),
    )

    # Show the plot
    st.plotly_chart(fig)
with tab2:
    stock_prices = vdf["Stock Price: Average, Min, Max"].to_list()
    stock_averages = []
    for s in stock_prices:
        stock_averages.append([float(x) for x in s[1:-1].split(',')][0])

    fig = go.Figure()

    # First trace for Stock Prices
    fig.add_trace(go.Scatter(x=vdf['I4: Year'], y=stock_averages, mode='lines+markers',
                             name='Stock Price', line=dict(color='green')))

    # Second trace for ESG Score
    fig.add_trace(go.Scatter(x=vdf['I4: Year'], y=vdf['ESG_Score'], mode='lines+markers', name='ESG Score',
                             yaxis='y2', line=dict(color='orange')))

    # Create a layout with two y-axes
    fig.update_layout(
        title={
            'text': f'{com} Stock Prices and ESG Scores Over Years',
            'x': 0.3  # Centered title
        },
        xaxis=dict(title='Year'),
        yaxis=dict(title='Stock Price', side='left', showgrid=False),
        yaxis2=dict(title='ESG Score', side='right', overlaying='y', showgrid=False),
        legend=dict(x=0, y=1.2),
    )

    # Show the plot
    st.plotly_chart(fig)

st.write("""
## Anomaly Detection
""")

esg_cols = vdf.columns.to_list()[4:22]
esg_full = {'E': 'Environmental', 'S': 'Social', 'G': 'Governance'}


def get_model_prediction(xcols, ycols, train_df):
    X = train_df[xcols].to_numpy().reshape(-1, 1)
    Y = train_df[ycols]

    model = MultiOutputRegressor(LinearRegression())

    model.fit(X, Y)

    y_pred = model.predict([[2023]])

    return pd.DataFrame({c: [max(0, min(100, round(d, 2)))] for c, d in zip(ycols, y_pred[0])})


tab1, tab2= st.tabs(["On Features", "On Timely Trends"])
with tab1:
    predict = st.checkbox(label="Show Feature Prediction")
    years = vdf['I4: Year'].unique()
    if predict:
        # Add histogram data
        selected_year = st.select_slider(
            'Predicting based on the previous data',
            [2022, 2023],
            value=2023,
            disabled=True)

        cdf = get_model_prediction('I4: Year', esg_cols, vdf)

    else:
        selected_year = st.select_slider(
            'Select the year you wish to check',
            years, value=years.max())
        # Add histogram data
        cdf = vdf[vdf['I4: Year'] == selected_year]

    cdf = pd.melt(cdf[esg_cols], var_name='ESG Items', value_name='ESG Value')
    cdf['Category'] = cdf['ESG Items'].apply(lambda x: esg_full[x[0]])
    avg = cdf['ESG Value'].mean()
    cdf.loc[cdf['ESG Value'] <= avg / 6, 'Category'] = 'Anomaly'
    adf = cdf[cdf['Category'] == 'Anomaly']
    cdf = cdf.drop(cdf[cdf['Category'] == 'Anomaly'].index)
    cdf = pd.concat([cdf, adf], ignore_index=True)

    fig = px.scatter(
        cdf,
        y="ESG Value",
        color="Category",
        size="ESG Value",
        hover_name="ESG Items",
        size_max=30,
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    predict = st.checkbox(label="Show Trends Prediction")
    level = ['Low', 'Mid', 'Normal', 'High', 'Robust']
    conf = st.select_slider("Choose the confidence bound",
                            options=level, value=level[2])
    conf_lvl = {k:v for k, v in zip(level, [0.97, 0.95, 0.9, 0.8, 0.7])}
    cdf = vdf.rename(columns={'I4: Year': 'Year'})
    stds = round(mean_df.std() * 2, 2)
    maxCol=lambda x: max(abs(x.min()), x.max())
    diff_cols = [l+' Diff' for l in esg_full.keys()]
    for k in diff_cols:
        cdf[k] = cdf[k[0] + ' Average Score'] - cdf['ESG_Score']
    cdf['AbsMax'] = cdf[diff_cols].apply(maxCol, axis=1)
    mami = cdf['AbsMax'].max()
    bound = max(round(mami*conf_lvl[conf], 2), stds)
    if predict:
        pred = get_model_prediction(['Year'], [k+' Average Score' for k in esg_full.keys()]
                                    + ['ESG_Score'], cdf)
        pred['Year'] = 2023
        cdf = pd.concat([cdf, pred], ignore_index=True)
    fig = go.Figure([
        go.Scatter(
            name=esg_full['E'],
            x=cdf['Year'],
            y=cdf['E Average Score'],
            mode='markers',
        ),
        go.Scatter(
            name=esg_full['S'],
            x=cdf['Year'],
            y=cdf['S Average Score'],
            mode='markers',
        ),
        go.Scatter(
            name=esg_full['G'],
            x=cdf['Year'],
            y=cdf['G Average Score'],
            mode='markers',
        ),
        go.Scatter(
            name='ESG Score',
            x=cdf['Year'],
            y=cdf['ESG_Score'],
            mode='lines',
            line=dict(color='green', width=3),
        ),
        go.Scatter(
            name='Upper Bound',
            x=cdf['Year'],
            y=cdf['ESG_Score'] + bound,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=cdf['Year'],
            y=cdf['ESG_Score'] - bound,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    if predict:
        fig.add_shape(type="rect",
                      x0=2022.5, y0=0, x1=2023.5, y1=100,
                      line=dict(color="RoyalBlue"),
                      )
    fig.update_layout(
        title={
            'text': f'{com} Anomaly Trends Detection',
            'x': 0.15  # Centered title
        },
        yaxis_title='ESG Score',
        hovermode="x",
        legend={'traceorder': 'normal'}
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
