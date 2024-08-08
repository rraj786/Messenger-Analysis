'''
    The following script contains various class methods to analyse Messenger
    chats and save outputs to the designated directory.
    Author: Rohit Rajagopal
'''


from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from raceplotly.plots import barplot
import re
import seaborn as sns
from torch import cuda, device
from tqdm import tqdm
from transformers import pipeline
from wordcloud import WordCloud
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset


# Initialise stopwords for text processing
stopwords = set(stopwords.words('english'))

# Check if GPU exists for running transformers models
selected_device = device('cuda' if cuda.is_available() else 'cpu')


class AnalyseChat:

    def __init__(self, chat_history, batch_size, save_dir):

        # Remove any react notifications as these are only relevant to the participant that downloaded the messages
        self.chat_history = chat_history[chat_history['content_type'] != 'React Notification']
        self.batch_size = batch_size
        self.save_dir = save_dir

        # A message is deemed to be any form of content that can receive reacts
        msgs_only_filter = ['Message', 'Shared Location']
        self.msgs_only = self.chat_history[self.chat_history['content_type'].isin(msgs_only_filter)]

        # Extract text messages for word analysis
        self.texts_only = self.msgs_only[(self.msgs_only['content_type'] == 'Message') & (self.msgs_only['media_type'] == 'Text')]

        # Get all unique participants in chat
        self.participants = self.chat_history['sender_name'].unique()

    def headline_stats(self):

        multimedia_filters = ['Photo/Video', 'File Attachment', 'Shared Link', 'Audio', 'Gif', 'Sticker']

        # Find number of participants
        no_participants = len(self.participants)

        # Find total chat interactions and messages sent
        totals = (len(self.chat_history), len(self.msgs_only))

        # Find average messages sent per day
        avg_contributions = round(self.msgs_only.groupby('date').size().mean(), 2)

        # Find average delay in minutes between contributions
        avg_delay = str(round((self.chat_history['datetime_local'].diff().dt.total_seconds() / 60).mean(), 2)) + ' minutes'

        # Find most active member in chat
        participant_contributions = self.chat_history.groupby('sender_name').size()
        most_active_participant = (participant_contributions.idxmax(), participant_contributions.max())

        # Find number of reactions given
        reactions = self.chat_history['reacts_count'].sum()

        # Find most active period in a week
        hour_day_contributions = self.chat_history.groupby(['hour_of_day', 'day_of_week']).size()
        time_id = hour_day_contributions.idxmax()
        end_time = datetime.strptime(time_id[0], '%I:%M%p') + timedelta(hours = 1)
        end_time_str = end_time.strftime('%I:%M%p').lstrip('0')
        most_active_time = time_id[1] + ' ' + time_id[0] + ' to ' + end_time_str

        # Find average text message length in words
        avg_words = round(self.texts_only['word_count'].mean(), 2)

        # Find total multimedia messages sent
        multimedia = len(self.chat_history[self.chat_history['media_type'].isin(multimedia_filters)])

        # Combine outputs
        aggs = [no_participants, totals, avg_contributions, avg_delay, most_active_participant, reactions, most_active_time, avg_words, multimedia]
        
        return aggs
    
    def summary_stats(self):

        # Find messages sent and reacts received aggregations
        summary = self.msgs_only.groupby('sender_name').agg(messages_sent = ('timestamp_ms', 'count'), 
                                                            reacts_received = ('reacts_count', 'sum'),
                                                            messages_with_reacts = ('reacts_count', lambda x: (x > 0).sum()))
        
        # Find number of emojis sent
        summary['Emojis Sent'] = self.texts_only.groupby('sender_name').agg(emojis_sent = ('emojis_count', 'sum'))

        # Find number of reacts given
        reacts_exploded = self.msgs_only.explode('reactions').dropna(subset = ['reactions'])
        summary['Reacts Given'] = reacts_exploded['reactions'].apply(lambda x: x['actor']).value_counts()

        # Find ratios for reacts per message and number of messages before receiving a react
        summary['Reacts Received per Message'] = summary['reacts_received'] / summary['messages_sent']
        summary['Messages Sent per React'] = summary['messages_sent'] / summary['messages_with_reacts']

        # Find media type aggregations
        media_filters = ['Other', 'Text']
        media_type_data = self.chat_history[~self.chat_history['media_type'].isin(media_filters)]
        media_pivot = media_type_data.pivot_table(index = 'sender_name', columns = 'media_type', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Find content type aggregations
        content_filters = ['Started Call', 'Member Left Group', 'Added Member to Group', 'Shared Location']
        content_type_data = self.chat_history[self.chat_history['content_type'].isin(content_filters)]
        content_pivot = content_type_data.pivot_table(index = 'sender_name', columns = 'content_type', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Consolidate final dataset by appending pivot tables above
        summary = pd.concat([summary, media_pivot, content_pivot], axis = 1, sort = True)

        # Add chat aggregates as last row
        summary.loc['Chat Aggregate'] = summary.sum()
        summary.at['Chat Aggregate', 'Reacts Received per Message'] = summary.at['Chat Aggregate', 'reacts_received'] / summary.at['Chat Aggregate', 'messages_sent']
        summary.at['Chat Aggregate', 'Messages Sent per React'] = summary.at['Chat Aggregate', 'messages_sent'] / summary.at['Chat Aggregate', 'messages_with_reacts']

        # Rename columns for consistency
        summary = summary.rename_axis('Participant')
        summary = summary.rename(columns = {'messages_sent': 'Messages Sent', 'reacts_received': 'Reacts Received', 'messages_with_reacts': 'Messages that Received Reacts'})

        # Add colour palette to distinguish between each participant
        summary['palette'] = px.colors.qualitative.Plotly[:len(summary.index)]

        # Create subplots to display summarised stats from chat
        fig = make_subplots(rows = 2, cols = 2, specs = [[{'type': 'domain'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
                            subplot_titles = ('Breakdown of Messages Sent', 'Reacts Received per Message', 'Total Emojis Sent', 'Calls over Time (by Month)'))
        
        # Plot 1 (pie chart)
        fig.add_trace(go.Pie(labels = summary.index[:-1], values = summary['Messages Sent'], marker = dict(colors = summary['palette'][:-1].tolist()), textinfo = 'label+percent',
                             textposition = 'outside', hole = 0.2), row = 1, col = 1)

        # Plot 2 (bar plot)
        # Extract chat aggregate value to plot
        chat_aggregate_value = summary.loc['Chat Aggregate', 'Reacts Received per Message']

        # Filter out chat aggregate and sort table
        reacts_data = summary.drop('Chat Aggregate').sort_values(by = 'Reacts Received per Message', ascending = False)

        # Plot chart and horizontal line for chat aggregate
        fig.add_trace(go.Bar(x = reacts_data.index,  y = reacts_data['Reacts Received per Message'], marker_color = reacts_data['palette'], text = reacts_data['Reacts Received per Message'], 
                      textposition = 'outside'), row = 1, col = 2)
        fig.add_shape(go.layout.Shape(type = 'line', x0 = -0.5, x1 = len(reacts_data.index), y0 = chat_aggregate_value, y1 = chat_aggregate_value, line = dict(color = 'Black', width = 2, 
                      dash = 'dash')), row = 1, col = 2)

        # Plot 3 (horizontal bar plot)
        # Filter out chat aggregate and sort table
        emojis_data = summary.drop('Chat Aggregate').sort_values(by = 'Emojis Sent', ascending = True)

        # Plot chart
        fig.add_trace(go.Bar(x = emojis_data['Emojis Sent'], y = emojis_data.index, orientation = 'h', marker_color = emojis_data['palette'], text = emojis_data['Emojis Sent'], 
                      textposition = 'outside'), row = 2, col = 1)

        # Plot 4 (line chart)
        # Filter for calls only
        calls_data = self.chat_history[self.chat_history['content_type'] == 'Started Call'].groupby('month_start').size()
    
        # Plot chart
        fig.add_trace(go.Scatter(x = calls_data.index, y = calls_data.values, mode = 'lines+markers'), row = 2, col = 2)        
        
        # Update layout
        fig.update_layout(showlegend = False, margin = dict(t = 100, l = 50, b = 50, r = 50), font = dict(size = 12), height = 800)
        fig.update_xaxes(tickangle = 0, tickfont = dict(size = 12))
        fig.update_yaxes(tickfont = dict(size = 12))
        for annotation in fig['layout']['annotations']:
            annotation['yanchor'] = 'bottom'
            annotation['y'] = annotation['y'] + 0.05  
            annotation['font'] = dict(size = 16)

        # Drop colour_palette column as it is no longer needed
        summary = summary.drop('palette', axis = 1)

        return summary, fig

    def cumulative_messages_over_time(self):
        
        # Find cumulative count of messages sent by date
        self.msgs_only['cumulative_count_msgs'] = self.msgs_only.groupby('sender_name').cumcount() + 1
        pivot_msg_counts = self.msgs_only.pivot_table(index = 'date', columns = 'sender_name', values = 'cumulative_count_msgs', aggfunc = 'last')
        pivot_msg_counts.fillna(method = 'ffill', inplace = True)

        # Find total messages sent over time for the user-defined time period
        total_msgs_over_time = pivot_msg_counts.sum(axis = 1).reset_index()
        total_msgs_over_time.columns = ['Date', 'Cumulative Count']

        # Unstack pivot table 
        cum_msg_counts = pivot_msg_counts.stack().reset_index()
        cum_msg_counts.columns = ['Date', 'Participant', 'Cumulative Count']

        # Plot total cumulative message count
        fig = px.line(total_msgs_over_time, x = 'Date', y = 'Cumulative Count', title = 'Cumulative Message Count')

        # Create racecar plot to dynamically visualise changes in cumulative count of messages
        racecar = barplot(cum_msg_counts, item_column = 'Participant', value_column = 'Cumulative Count', time_column = 'Date', top_entries = 10)
        racecar_output = racecar.plot(title = 'Cumulative Message Count by Participant', frame_duration = 75)

        return fig, racecar_output

    def raw_messages_over_time(self):

        # Find count of messages sent by each participant per week only (captures low-level changes well without being too granular)]
        pivot_msg_counts = self.msgs_only.pivot_table(index = 'week_start', columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count')
        pivot_msg_counts = pivot_msg_counts.fillna(0)
        
        # Find total messages sent per period
        pivot_msg_counts['Chat Aggregate'] = pivot_msg_counts.sum(axis = 1)

        # Unstack pivot table 
        raw_msg_counts = pivot_msg_counts.stack().reset_index()
        raw_msg_counts.columns = ['Week', 'Participant', 'Message Count']

        # Create an interactive plot to select what lines to display
        fig = px.line(raw_msg_counts, x = 'Week', y = 'Message Count', color = 'Participant', title = 'Raw Message Counts by Week', 
                        color_discrete_sequence = px.colors.qualitative.Plotly)

        # Add button to show all lines
        fig.update_layout(updatemenus = [dict(type = 'buttons', x = 1.1, y = 1.2, buttons = [dict(label = 'Show All', method = 'update',
                            args=[{'visible': [True] * len(raw_msg_counts['Participant'].unique())}])])])
        
        return fig
    
    def chat_activity(self):

        # Create list of time periods to consider
        plot_period = ['hour_of_day', 'day_of_week', 'week', 'month', 'quarter', 'year'] 

        # Find all chat interactions for each participant based on each time period
        # This looks at all content, not only messages 
        area_plots = []
        for period in plot_period:
            clean_period = ' '.join([word.capitalize() if word != 'of' else word for word in period.split('_')])

            # Group interactions by period and participant
            content_counts = self.chat_history.pivot_table(index = period, columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

            # Unstack pivot table
            plot_data = content_counts.stack().reset_index()
            plot_data.columns = [clean_period, 'Participant', 'Interactions']

            # Plot user interactions
            fig = px.area(plot_data, x = clean_period, y = 'Interactions', color = 'Participant', title = 'Number of Interactions by ' + clean_period, 
                          color_discrete_sequence = px.colors.qualitative.Plotly)

            area_plots.append(fig)     
        
        # Group number of chat interactions by hour of day and day of week
        activity_data = self.chat_history.pivot_table(index = 'hour_of_day', columns = 'day_of_week', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Plot heatmap of total chat activity
        heatmap = go.Figure(data = go.Heatmap(z = activity_data.values, x = activity_data.columns, y = activity_data.index, text = activity_data.values,
                        texttemplate = '%{text:.0f}', colorscale = 'Viridis'))
        heatmap.update_layout(title = 'Heatmap of Chat Activity by Hour and Day of Week', xaxis_title = 'Day of Week', yaxis_title = 'Hour of Day', height = 800)

        # Find the top 10 most and least active days in the chat
        activity_by_date = self.chat_history.pivot_table(index = 'date', columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)
        activity_by_date['total'] = activity_by_date.sum(axis = 1)
        top_10_most_active = activity_by_date.nlargest(10, 'total').drop(columns = ['total'])
        top_10_least_active = activity_by_date.nsmallest(10, 'total').drop(columns = ['total'])

        # Unstack pivot tables
        top_10_most_active_plot = top_10_most_active.stack().reset_index()
        top_10_most_active_plot.columns = ['Date', 'Participant', 'Interactions']
        top_10_least_active_plot = top_10_least_active.stack().reset_index()
        top_10_least_active_plot.columns = ['Date', 'Participant', 'Interactions']

        # Plot figure with subplots for most and least active dates
        extremes = make_subplots(rows = 1, cols = 2, subplot_titles = ('Top 10 Most Active Days', 'Top 10 Least Active Days'))
        most = px.bar(top_10_most_active_plot, x = 'Date', y = 'Interactions', color = 'Participant', color_discrete_sequence = px.colors.qualitative.Plotly)    
        for trace in most.data:
            trace.showlegend = False
            extremes.add_trace(trace, row = 1, col = 1)

        least = px.bar(top_10_least_active_plot, x = 'Date', y = 'Interactions', color = 'Participant', color_discrete_sequence = px.colors.qualitative.Plotly)        
        for trace in least.data:
            extremes.add_trace(trace, row = 1, col = 2)

        extremes.update_layout(barmode = 'stack')
        extremes.update_xaxes(type = 'category', categoryorder = 'array', categoryarray = top_10_most_active_plot['Date'], row = 1, col = 1)
        extremes.update_xaxes(type = 'category', categoryorder = 'array', categoryarray = top_10_least_active_plot['Date'], row = 1, col = 2)

        return area_plots, heatmap, extremes
    
    def react_analysis(self):

        # Extract actor and react emoji into new columns
        reacts_exploded = self.msgs_only.explode('reactions').dropna(subset = ['reactions'])
        reacts_exploded['actor'] = reacts_exploded['reactions'].apply(lambda x: x['actor'])
        reacts_exploded['reaction'] = reacts_exploded['reactions'].apply(lambda x: x['reaction'])       

        # Find count of each distinct react given/received (top 10)
        total_count_reactions = reacts_exploded['reaction'].value_counts().nlargest(10)

        # Find count of each distinct react given by each participant (top 10)
        reactions_given = reacts_exploded.groupby(['actor', 'reaction']).size().reset_index(name = 'count')
        reactions_given_sorted = reactions_given.sort_values(['actor', 'count'], ascending = [True, False])
        top_reactions_given = reactions_given_sorted.groupby('actor').head(10)
        top_reactions_given = (top_reactions_given.reset_index(drop = True)
                           .set_index(top_reactions_given.groupby('actor').cumcount() + 1))
        reactions_given_participant = top_reactions_given.pivot(columns = 'actor', values = 'reaction').fillna('')
        reactions_given_participant['Group Aggregate'] = total_count_reactions.index

        # Find count of each distinct react received by each participant (top 10)
        reactions_received = reacts_exploded.groupby(['sender_name', 'reaction']).size().reset_index(name = 'count')
        reactions_received_sorted = reactions_received.sort_values(['sender_name', 'count'], ascending = [True, False])
        top_reactions_received = reactions_received_sorted.groupby('sender_name').head(10)
        top_reactions_received = (top_reactions_received.reset_index(drop = True)
                           .set_index(top_reactions_received.groupby('sender_name').cumcount() + 1))
        reactions_received_participant = top_reactions_received.pivot(columns = 'sender_name', values = 'reaction').fillna('')
        reactions_received_participant['Group Aggregate'] = total_count_reactions.index

        # Calculate ratio of messages reacted to by each participant for other participants in the chat
        reacts_grid = reacts_exploded.pivot_table(index = 'actor', columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)
        reacts_grid_ratio = 100 * reacts_grid.div(reacts_grid.sum(axis = 0), axis = 'columns')

        # Plot heatmap of inter-group reaction trends
        heatmap = go.Figure(data = go.Heatmap(z = reacts_grid_ratio.values, x = reacts_grid_ratio.columns, y = reacts_grid_ratio.index, text = reacts_grid_ratio.values,
                texttemplate = '%{text:.2f}%', colorscale = 'Viridis'))
        heatmap.update_layout(title = 'Heatmap of Reactions Sent and Received by Participants', xaxis_title = 'Received by', yaxis_title = 'Given by', height = 800)

        # Find the 25 most reacted to messages all-time (in case of tie-breakers, consider the most recent message)
        reacted_msgs_sorted = self.msgs_only.sort_values(by = ['reacts_count', 'date', 'sender_name'], ascending = [False, False, True])
        reacted_msgs_sorted['coalesced_content'] = reacted_msgs_sorted['content'].fillna(reacted_msgs_sorted['media_type'])
        top_reacted_msgs = reacted_msgs_sorted[['date', 'sender_name', 'coalesced_content', 'reacts_count']].head(25)
        top_reacted_msgs.columns = ['Date', 'Participant', 'Content', 'Count of Reacts']
        top_reacted_msgs.set_index('Date')

        # Find the top reacted message for each participant all-time (in case of tie-breakers, consider the most recent message)
        top_reacted_msgs_participant = reacted_msgs_sorted.groupby('sender_name').first().reset_index()
        top_reacted_msgs_participant = top_reacted_msgs_participant[['date', 'sender_name', 'coalesced_content', 'reacts_count']]
        top_reacted_msgs_participant.columns = ['Date', 'Participant', 'Content', 'Count of Reacts']
        top_reacted_msgs_participant.set_index('Date')

        return reactions_given_participant, reactions_received_participant, heatmap, top_reacted_msgs, top_reacted_msgs_participant

    def word_analysis(self):

        # Add to notes in docstring, no processing was done for sentiment/emotion analysis such as removing
        # stopwords as raw text offers better insights on what language people actually use (are they more self-centered
        # or do they speak about everyone inclusively)
        # Also, an example for sentiment/emotion analysis - 'I am not having a great day'
        # If we were to remove NLTK stopwords from this, it would be 'great day' which is evidently very different
        # to the original text and has a completely different sentiment to what was originally intended 
        # Left emojis in as well as they affect the output of sentiment/emotion analysis
        # Wordclouds removed stopwords, lowered text and non-alphanumeric
        # Topic modelling removed stopwords, lowered text, non-alphanumeric and lemmatisation

        # Get word length aggregates for text messages sent in chat
        word_summary_chat = self.texts_only.agg(average_words = ('word_count', 'mean'),
                                                max_words = ('word_count', 'max'))
        
        # Get word length aggregates for text messages sent by each participant
        word_summary_participant = self.texts_only.groupby('sender_name').agg(average_words = ('word_count', 'mean'),
                                                                              max_words = ('word_count', 'max'))
        
        # Process text to normalise text, and remove stop words, emojis, and punctuation for word cloud plots
        self.texts_only['processed_text'] = self.texts_only['content'].apply(self.process_text)

        # Build word cloud for chat overall (top 75 words)
        most_common_chat = Counter(' '.join(self.texts_only['processed_text']).split()).most_common(75)
        wordcloud_chat = WordCloud(width = 800, height = 600, background_color = 'white').generate_from_frequencies(dict(most_common_chat))
        plt.figure(figsize = (10, 6))
        plt.title('75 Most Common Words used in Chat')
        plt.imshow(wordcloud_chat, interpolation = 'bilinear')
        plt.axis('off')

        # Save plot (create new directory to save output if not available already)
        word_analysis_dir = os.path.join(self.save_dir, 'word_analysis')
        if not os.path.exists(word_analysis_dir):
            os.makedirs(word_analysis_dir)

        plt.savefig(os.path.join(word_analysis_dir, 'wordcloud_chat.jpg'), bbox_inches = 'tight')
        plt.close()

        # Build word clouds for each participant (top 75 words)
        for person in self.participants:
            words_participant = self.texts_only[self.texts_only['sender_name'] == person]['processed_text']
            most_common_participant = Counter(' '.join(words_participant).split()).most_common(75)
            wordcloud_partcipant = WordCloud(width = 800, height = 600, background_color = 'white').generate_from_frequencies(dict(most_common_participant))
            plt.figure(figsize = (10, 6))
            plt.title('75 Most Common Words used by ' + person)
            plt.imshow(wordcloud_partcipant, interpolation = 'bilinear')
            plt.axis('off')

            # Save plot
            file_name = 'wordcloud_' + person + '.jpg'
            plt.savefig(os.path.join(word_analysis_dir, file_name), bbox_inches = 'tight')
            plt.close()

        # Convert column of messages to Dataset object for efficient model processing
        data_dict = {'text': self.texts_only['content'].tolist()}
        text_dataset = Dataset.from_dict(data_dict)

        # Perform sentiment analysis on each text message using pre-trained transformers model
        # Retain the label with the highest score
        self.texts_only['sentiment'] = self.sentiment_analysis(text_dataset, self.batch_size)

        # Perform emotion analysis on each text message using pre-trained transformers model
        # Retain the label with the highest score
        self.texts_only['emotion'] = self.emotion_analysis(text_dataset, self.batch_size)

        proportions = self.texts_only.groupby(['sender_name', 'sentiment']).size().unstack(fill_value=0).apply(lambda x: x / x.sum(), axis=1)
        proportions = proportions.reset_index().melt(id_vars='sender_name', var_name='sentiment', value_name='proportion')

        # Plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(x='sender_name', y='proportion', hue='sentiment', data=proportions, palette='Set3')
        plt.xlabel('Sender Name')
        plt.ylabel('Proportion of Messages')
        plt.title('Proportion of Sentiments by Sender')
        plt.tight_layout()
        plt.show()
        
        return
    
    @staticmethod
    def process_text(text):

        # Convert text to lower case
        text = text.lower()

        # Remove non-alphanumeric characters (punctuation, emojis etc.)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove stop words 
        text = ' '.join([word for word in text.split() if word not in stopwords])

        return text
    
    @staticmethod
    def sentiment_analysis(dataset, batch_size):

        # Intialise transformers model for sentiment analysis
        sentiment_model = pipeline('text-classification', model = 'cardiffnlp/twitter-roberta-base-sentiment-latest', 
                                   device = selected_device, batch_size = batch_size, truncation = True, max_length = 512)

        # Run model
        results = sentiment_model(KeyDataset(dataset, 'text'))

        # Collect results and track progress
        labels = []
        for result in tqdm(results, total = len(dataset), position = 0, leave = True, desc = 'Sentiment Analysis'):
            labels.append(result['label'])

        return labels 
    
    @staticmethod
    def emotion_analysis(dataset, batch_size):

        # Intialise transformers model for emotion analysis
        emotion_model = pipeline('text-classification', model = 'michellejieli/emotion_text_classifier', 
                                 device = selected_device, batch_size = batch_size, truncation = True, max_length = 512)

        # Run model
        results = emotion_model(KeyDataset(dataset, 'text'))

        # Collect results and track progress
        labels = []
        for result in tqdm(results, total = len(dataset), position = 0, leave = True, desc = 'Emotion Analysis'):
            labels.append(result['label'])

        return labels 
    