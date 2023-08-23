import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Revenue', color=color)
    ax1.plot(vdf['I4: Year'], vdf['I3: Revenue'], color=color, marker='o', label='Revenue')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('ESG Score', color=color)
    ax2.plot(vdf['I4: Year'], vdf['ESG_Score'], color=color, marker='s', label='ESG Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'{com} Revenue and ESG Scores Over Years')
    plt.grid()
    st.pyplot(plt.gcf())
with tab2:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    stock_prices = vdf["Stock Price: Average, Min, Max"].to_list()
    stock_averages = []
    for s in stock_prices:
        stock_averages.append([float(x) for x in s[1:-1].split(',')][0])

    color = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Stock Price', color=color)
    ax1.plot(vdf['I4: Year'], stock_averages, color=color, marker='o', label='Stock Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('ESG Score', color=color)
    ax2.plot(vdf['I4: Year'], vdf['ESG_Score'], color=color, marker='s', label='ESG Score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'{com} Stock Price and ESG Scores Over Years')
    plt.grid()
    st.pyplot(plt.gcf())

st.write("""
## Anomaly Detection
""")

esg_cols = vdf.columns.to_list()[4:22]
esgs = ['E', 'S', 'G']
esgs = [[x for x in esg_cols if x[0] == c] for c in esgs]


tab1, tab2, tab3 = st.tabs(["On Features", "On Company Trends", "Across All"])
with tab1:
    years = vdf['I4: Year'].unique()
    selected_year = st.selectbox(
        'Select the year you wish to check',
        years)
    # Add histogram data
    current_df = vdf[vdf['I4: Year'] == selected_year]
    x1 = current_df[esgs[0]].values.tolist()[0]
    x2 = current_df[esgs[1]].values.tolist()[0]
    x3 = current_df[esgs[2]].values.tolist()[0]
    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ['E', 'S', 'G']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)
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
