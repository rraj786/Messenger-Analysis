'''
    The following script generates a Streamlit app to summarise all the key
    metrics from conversations.

    Author: Rohit Rajagopal
'''


from metrics import *
import streamlit as st


def generate_report(chat_history, group_name, batch_size):
       
    """
        Generates an interactive Streamlit report for Messenger group chat history.

        Args:
            - chat_history (pd.DataFrame): DataFrame of chat history.
            - group_name (str): Name of the Messenger group.
            - batch_size (int): The batch size for processing the text data.

        Returns:
            None
        
        Notes:
            - Application opens in browser as http://localhost:8501/.
    """

    # Initialise AnalyseChat class
    metrics = AnalyseChat(chat_history, batch_size)

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
            font-size: 16px;
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
        .reportview-container .dataframe-container {
            width: 100% !important;
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
    st.markdown(f'<div class="caption">An overview of Messenger group chat metrics, featuring plots to highlight significant trends and key insights.</div>', unsafe_allow_html = True)
   
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
    summary, fig = metrics.summary_stats()

    st.plotly_chart(fig)
    
    # Display summary table under expanded tab 
    with st.expander('Expand to view full Summary Breakdown'):

        # Highlight totals row
        summary_styled = summary.style.apply(lambda x: ['background-color: yellow' if x.name == 'Chat Aggregate' else '' for _ in x], axis = 1)
        summary_styled = summary_styled.format({
                            'Messages Sent': "{:,.0f}",
                            'Reactions Received': "{:,.0f}",
                            'Messages that Received Reactions': "{:,.0f}",
                            'Reactions Given': "{:,.0f}",
                            'Reactions Received per Message': "{:.4f}",
                            'Messages Sent per Reaction': "{:.4f}",
                            'Emojis Sent': "{:,.0f}",
                            'Audio': "{:,.0f}",
                            'Deleted Message': "{:,.0f}",
                            'File Attachment': "{:,.0f}",
                            'Gif': "{:,.0f}",
                            'Photo/Video': "{:,.0f}",
                            'Shared Link': "{:,.0f}",
                            'Sticker': "{:,.0f}",
                            'Added Member to Group': "{:,.0f}",
                            'Member Left Group': "{:,.0f}",
                            'Shared Location': "{:,.0f}",
                            'Started Call': "{:,.0f}"})
        st.write(summary_styled)

    # Section 2
    # Display cumulative messages over time
    st.markdown(f'<div class="subheader">Messages Sent over Time', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">Message trends by participant and chat overall, with insights into seasonal patterns.</div>', unsafe_allow_html = True)
    
    cum_fig, racecar_fig = metrics.cumulative_messages_over_time()

    st.plotly_chart(cum_fig)

    # Display racecar output
    st.plotly_chart(racecar_fig)

    # Display raw messages over time    
    raw_fig = metrics.raw_messages_over_time()

    st.plotly_chart(raw_fig)

    # Section 3 
    # Display chat activity over different time periods
    st.markdown(f'<div class="subheader">Chat Activity', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">Examination of chat activity patterns to identify peak engagement times and determine the most and least active days historically.</div>', 
                unsafe_allow_html = True)
    
    periods_fig, activity_fig, extremes_fig = metrics.chat_activity()
    
    st.plotly_chart(periods_fig)

    # Display heatmap and extreme usage time
    st.plotly_chart(activity_fig)
    st.plotly_chart(extremes_fig)

    # Section 4
    # Display top used and received reacts by participant
    st.markdown(f'<div class="subheader">Reactions Analysis', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">Identify most used reacts,group and individual interactions, and top messages by number of reacts received.</div>', 
                unsafe_allow_html = True)
    given, received, react_fig1, react_fig2, top_msgs, top_msgs_participant = metrics.react_analysis()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="dfheader">Top 10 Reacts Given</div>', unsafe_allow_html = True)
        st.dataframe(given)
    with col2:
        st.markdown(f'<div class="dfheader">Top 10 Reacts Received</div>', unsafe_allow_html = True)
        st.dataframe(received)
    
    # Display number of reacts given and received by participant
    st.plotly_chart(react_fig1)

    # Display heatmap of react group interactions
    st.plotly_chart(react_fig2)

    # Display top messages in chat
    st.markdown(f'<div class="dfheader">Top 25 Messages Sent based on Number of Reacts Received</div>', unsafe_allow_html = True)
    st.dataframe(top_msgs, use_container_width = True)
    st.markdown(f'<div class="dfheader">Top Message Sent by each Participant based on Number of Reacts Received</div>', unsafe_allow_html = True)
    st.dataframe(top_msgs_participant, use_container_width = True)

    # Section 5
    # Display word length aggregates 
    st.markdown(f'<div class="subheader">Word Analysis', unsafe_allow_html = True)
    st.markdown(f'<div class="caption">Analyse messages lengths by participant and general tone through sentiment and emotion analysis.</div>', 
                unsafe_allow_html = True)
    word_length_fig, wordclouds, sentiment_fig, emotion_fig = metrics.word_analysis()

    st.plotly_chart(word_length_fig)

    # Display chat and participant word clouds
    st.plotly_chart(wordclouds)

    # Display sentiment and emotion analysis chart
    st.plotly_chart(sentiment_fig)
    st.plotly_chart(emotion_fig)

    st.markdown(f'<div class="caption">End of report.</div>', unsafe_allow_html = True)

def create_card(title, content):

    """
        Displays a styled metric card.

        Args:
            - title (str): Card title.
            - content (str): Metric value

        Returns:
            None
    """

    # Create container to house metric
    with st.container():
        st.write(f'**{title}**')
        st.write(content)
        st.write('---')  
