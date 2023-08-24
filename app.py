import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np


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
col1.metric("ESG Score", esg_last_two[1], round(esg_last_two[1]-esg_last_two[0], 2))
col2.metric("Revenue", "$" + str(round(revenue_last_two[1]/1000000, 2)) + "M",
            str(round((revenue_last_two[1] - revenue_last_two[0]) / revenue_last_two[0], 2)) + "%")
col3.metric("Stock Price YTD", "$" + str(stock_avg_last_two[3]),
            round(stock_avg_last_two[3]-stock_avg_last_two[0], 2))

# TODO - update after complete all three anomaly analysis
col4.metric("ESG Anti-Risk Level", "86%", "4%")


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
esg_full = {'E':'Environmental', 'S':'Social', 'G':'Governance'}


tab1, tab2, tab3 = st.tabs(["On Features", "On Company Trends", "Across All"])
with tab1:
    years = vdf['I4: Year'].unique()
    selected_year = st.selectbox(
        'Select the year you wish to check',
        years)
    # Add histogram data
    cdf = vdf[vdf['I4: Year'] == selected_year]
    cdf = pd.melt(cdf[esg_cols], var_name='ESG Items', value_name='ESG Value')
    cdf['Category'] = cdf['ESG Items'].apply(lambda x: esg_full[x[0]])
    avg = cdf['ESG Value'].mean()
    cdf.loc[cdf['ESG Value'] <= avg/3, 'Category'] = 'Anomaly'
    adf = cdf[cdf['Category'] == 'Anomaly']
    cdf = cdf.drop(cdf[cdf['Category'] == 'Anomaly'].index)
    cdf = pd.concat([cdf, adf], ignore_index=True)

    print(cdf)
    fig = px.scatter(
        cdf,
        y="ESG Value",
        color="Category",
        size="ESG Value",
        hover_name="ESG Items",
        size_max=30,
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    # years = vdf['I4: Year'].unique()
    # selected_year = st.selectbox(
    #     'Select the year you wish to check',
    #     years)
    # # Add histogram data
    # current_df = vdf[vdf['I4: Year'] == selected_year]
    # x1 = current_df[esgs[0]].values.tolist()[0]
    # x2 = current_df[esgs[1]].values.tolist()[0]
    # x3 = current_df[esgs[2]].values.tolist()[0]
    # # Group data together
    # hist_data = [x1, x2, x3]
    #
    # group_labels = ['E', 'S', 'G']
    #
    # # Create distplot with custom bin_size
    # fig = ff.create_distplot(
    #     hist_data, group_labels, bin_size=[.1, .25, .5])
    #
    # # Plot!
    # st.plotly_chart(fig, use_container_width=True)
with tab2:
    df = px.data.iris()
    fig = px.scatter(
        df,
        x="sepal_width",
        y="sepal_length",
        color="sepal_length",
        color_continuous_scale="reds",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab3:
    df = px.data.gapminder()

    fig = px.scatter(
        df.query("year==2007"),
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=60,
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
