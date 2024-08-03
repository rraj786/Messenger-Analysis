'''
    The following script generates a Streamlit app to summarise all the key
    metrics from conversations.
    Author: Rohit Rajagopal
'''


from datetime import datetime
from metrics import *
import streamlit as st


def generate_report(args, group_name, chat_history):

    # Initialise AnalyseChat class
    metrics = AnalyseChat(chat_history, args.batch_size, args.save_dir)

    # Configure notebook layout
    st.set_page_config(layout = 'wide')
    st.markdown(
        """
        <style>
        .title {
            font-size: 46px;
            text-align: center;
            font-weight: bold;
        }
        .header {
            font-size: 36px;
            text-align: center;
            font-weight: bold;
        }
        .subheader {
            font-size: 32px;
            text-align: center;
            margin-top: 25px;
            margin-bottom: 25px;
            text-decoration: underline;
        }
        .caption {
            font-size: 16px;
            text-align: center;
            margin-top: 25px;
            margin-bottom: 25px;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html = True
    )


    # Set up notebook headers
    st.markdown(f'<div class="title">Messenger Group Chat Report for {group_name}</div>', unsafe_allow_html = True)
    date_today = datetime.now().strftime('%A, %d %b %Y')
    st.markdown(f'<div class="header">Generated on {date_today}</div>', unsafe_allow_html = True)
    start = chat_history['datetime_utc'].min().strftime('%d/%m/%Y %I:%M%p')
    end = chat_history['datetime_utc'].max().strftime('%d/%m/%Y %I:%M%p')
    st.markdown(f'<div class="caption">Chat history between {start} and {end} UTC</div>', unsafe_allow_html = True)

    # Display headline stats as cards across two rows
    st.markdown(f'<div class="subheader">Headline Statistics', unsafe_allow_html = True)
    aggs = metrics.headline_stats()
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        create_card('Number of Participants', '{:,}'.format(aggs[0]))
    with col2:
        create_card('Total Interactions', '{:,}'.format(aggs[1][0]))
    with col3:
        create_card('Total Messages Sent', '{:,}'.format(aggs[1][1]))
    with col4:
        create_card('Mean Messages per Day', '{:,}'.format(aggs[2]))
    with col5:
        create_card('Total Reactions Given', '{:,}'.format(aggs[5]))
    with col6:
        create_card('Total Multimedia Messages Sent', '{:,}'.format(aggs[8]))

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        create_card('Mean Response Time', aggs[3])
    with col8:
        create_card('Most Active Time of Week', aggs[6])
    with col9:
        create_card('Most Active Participant', aggs[4][0])
    with col10:
        create_card('Mean Words per Message', '{:,}'.format(aggs[7]))

    # Display summary plots
    st.markdown(f'<div class="subheader">Summary', unsafe_allow_html = True)

    # Display summary table under expanded tab
    summary, fig = metrics.summary_stats()
    with st.expander('Expand to view full Summary Breakdown'):

        # Highlight totals row
        summary_styled = summary.style.apply(lambda x: ['background-color: yellow' if x.name == 'Chat Aggregate' else '' for _ in x], axis = 1)
        st.write(summary_styled, unsafe_allow_html = True)

    # Display messages over time
    st.markdown(f'<div class="subheader">Messages Sent over Time', unsafe_allow_html = True)
    plots, racecar_output = metrics.cumulative_messages_over_time()

    # Create radio buttons to select time period
    time_period = st.radio('Select a Time Period for Cumulative Plots below', ['Date', 'Week', 'Month', 'Quarter', 'Year'])

    # Select plot to display
    options_map = {'Date': 0, 'Week': 1, 'Month': 2, 'Quarter': 3, 'Year': 4}
    st.pyplot(plots[options_map[time_period]])

    # Display racecar output
    st.plotly_chart(racecar_output)

    # metrics.messages_over_time(args.cumulative_msgs_time_period)
    # metrics.chat_activity(args.chat_activity_time_period)
    # metrics.react_analysis()
    # metrics.word_analysis()

def create_card(title, content):

    with st.container():
        st.write(f'**{title}**')
        st.write(content)
        st.write('---')  
