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
            font-size: 28px;
            text-align: center;
            margin-top: 25px;
            margin-bottom: 5px;
            text-decoration: underline;
        }
        .dfheader {
            font-size: 18px;
            font-weight: bold;
        }
        .caption {
            font-size: 16px;
            text-align: center;
            margin-top: 5px;
            margin-bottom: 30px;
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

    # Section 1
    # Display headline stats as cards across two rows
    st.markdown(f'<div class="subheader">Summary', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">High-level summary of Messenger group chat metrics with plots to highlight key trends.</div>', unsafe_allow_html = True)
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
    @st.cache_data
    def summary_stats():
        
        return metrics.summary_stats()
    
    summary, fig = summary_stats()

    st.plotly_chart(fig)
    
    # Display summary table under expanded tab 
    with st.expander('Expand to view full Summary Breakdown'):

        # Highlight totals row
        summary_styled = summary.style.apply(lambda x: ['background-color: yellow' if x.name == 'Chat Aggregate' else '' for _ in x], axis = 1)
        st.write(summary_styled, unsafe_allow_html = True)

    # Section 2
    # Display cumulative messages over time
    st.markdown(f'<div class="subheader">Messages Sent over Time', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">Message trends by participant and chat overall, with insights into seasonal patterns.</div>', unsafe_allow_html = True)
    @st.cache_data
    def cumulative_messages_over_time():
        
        return metrics.cumulative_messages_over_time()
    
    cum_fig, racecar_fig = cumulative_messages_over_time()

    st.plotly_chart(cum_fig)

    # Display racecar output
    st.plotly_chart(racecar_fig)

    # Display raw messages over time
    @st.cache_data
    def raw_messages_over_time():
        
        return metrics.raw_messages_over_time()
    
    raw_fig = raw_messages_over_time()

    st.plotly_chart(raw_fig)

    # Section 3 
    # Display chat activity over different time periods
    st.markdown(f'<div class="subheader">Chat Activity', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">Analyse chat activity patterns to pinpoint peak engagement times and identify periods of high and low chat activity.</div>', 
                unsafe_allow_html = True)
 
    @st.cache_data
    def chat_activity():

        return metrics.chat_activity()
    
    period_plots, activity_fig, extremes_fig = chat_activity()
    
    # Create dropdown for user to selected desired time period
    mapping = {'Hour of Day': 0, 'Day of Week': 1, 'Week': 2, 'Month': 3, 'Quarter': 4, 'Year': 5}
    period = st.selectbox('Select a time period for Activity Breakdown below:', options = ['Hour of Day', 'Day of Week', 'Week', 'Month', 'Quarter', 'Year'])

    st.plotly_chart(period_plots[mapping[period]])

    # Display heatmap and extreme usage time
    st.plotly_chart(activity_fig)
    st.plotly_chart(extremes_fig)

    # Section 4
    # Display top used and received reacts by participant
    st.markdown(f'<div class="subheader">Reactions Analysis', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">Identify most used reacts, group interactions, and top messages by number of reacts received.</div>', 
                unsafe_allow_html = True)
    given, received, react_fig, top_msgs, top_msgs_participant = metrics.react_analysis()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="dfheader">Top 10 Reacts Given</div>', unsafe_allow_html = True)
        st.write(given)

    with col2:
        st.markdown(f'<div class="dfheader">Top 10 Reacts Received</div>', unsafe_allow_html = True)
        st.write(received)
    
    # Display heatmap of react group interactions
    st.plotly_chart(react_fig)

    # Display top messages in chat
    st.write(top_msgs)
    st.write(top_msgs_participant)


    # metrics.word_analysis()

def create_card(title, content):

    # Create container to house metric
    with st.container():
        st.write(f'**{title}**')
        st.write(content)
        st.write('---')  
