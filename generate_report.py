'''
    The following script generates a Streamlit app to summarise all the key
    metrics from conversations.
    Author: Rohit Rajagopal
'''


from datetime import datetime
from metrics import *
import streamlit as st
import os


def generate_report(args, group_name, chat_history):

    # Initialise AnalyseChat class
    metrics = AnalyseChat(chat_history, args.batch_size, args.save_dir)

    # Set up report headers
    st.title('Messenger Group Chat Report for ' + group_name)
    st.header('Generated on: ' + datetime.now().strftime('%A, %d %b %Y'))
    st.caption('Chat history: ' + chat_history['datetime_utc'].min().strftime('%d/%m/%Y %I:%M%p') + ' to ' +
                chat_history['datetime_utc'].max().strftime('%d/%m/%Y %I:%M%p') + ' UTC')
    
    # Display headline stats as cards
    aggs = metrics.headline_stats()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        create_card('Number of Participants', '{:,}'.format(aggs[0]))
    with col2:
        create_card('Total Interactions', '{:,}'.format(aggs[1][0]))
    with col3:
        create_card('Interactions per Day', '{:,}'.format(aggs[2]))
    with col4:
        create_card('Mean Response Time (mins)', '{:,}'.format(aggs[3]))
    with col5:
        create_card('Most Active Participant', aggs[4][0])

    # Second row with 4 cards
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        create_card('Total Reactions Given', '{:,}'.format(aggs[5]))
    with col7:
        create_card('Most Active Time', '{:,}'.format(aggs[6][0]))
    with col8:
        create_card('Mean Words per Message', '{:,}'.format(aggs[7]))
    with col9:
        create_card('Total Multimedia Sent', '{:,}'.format(aggs[8]))


    os.system('streamlit run generate_report.py')




        # metrics.summary_stats()
    # metrics.messages_over_time(args.cumulative_msgs_time_period)
    # metrics.chat_activity(args.chat_activity_time_period)
    # metrics.react_analysis()
    # metrics.word_analysis()

def create_card(title, content):
    # Using st.empty() to create a card-like container with Streamlit components
    with st.container():
        st.write(f'**{title}**')
        st.write(content)
        st.write('---')  # A horizontal line to separate cards