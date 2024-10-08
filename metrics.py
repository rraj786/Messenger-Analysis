'''
    The following script contains various class methods to analyse Messenger
    chats and save outputs to the designated directory.

    Author: Rohit Rajagopal
'''


from collections import Counter
from datetime import datetime, timedelta
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from raceplotly.plots import barplot
import re
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
print('\nUsing ' + str(selected_device))


class AnalyseChat:

    """
        A class for analysing Messenger chat history across various metrics.

        Attributes:
            - chat_history (pd.DataFrame): The full chat history including various content types.
            - batch_size (int): The size of batches used for processing in sentiment and emotion analysis.
            - msgs_only (pd.DataFrame): Filtered chat history containing only messages and shared locations.
            - texts_only (pd.DataFrame): Filtered messages that are text-only (excluding media).
            - participants (list): List of unique participants in the chat.

        Notes:
            - React notifications are excluded from analysis as they are only relevant to individual participants.
            - Sentiment and emotion analyses are performed using pre-trained models and require processing in batches.
            - The models are expected to take a while to run, so please enable CUDA/local GPUs.
    """

    def __init__(self, chat_history, batch_size):

        # Remove any react notifications as these are only relevant to the participant that downloaded the messages
        self.chat_history = chat_history[chat_history['content_type'] != 'React Notification']
        self.batch_size = batch_size

        # A message is deemed to be any form of content that can receive reacts
        msgs_only_filter = ['Message', 'Shared Location']
        self.msgs_only = self.chat_history[self.chat_history['content_type'].isin(msgs_only_filter)]

        # Extract text messages for word analysis
        self.texts_only = self.msgs_only[(self.msgs_only['content_type'] == 'Message') & (self.msgs_only['media_type'] == 'Text')]

        # Get all unique participants in chat
        self.participants = sorted(self.chat_history['sender_name'].unique())

    def headline_stats(self):

        """
            Compute headline statistics from chat history, such as total interactions, most active member, and 
            average response time.

            Args:
                None

            Returns:
                - aggs (list): List containing various aggregated group chat statistics.
        """

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
        reactions = self.msgs_only['reacts_count'].sum()

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

        """
            Computes and visualises summary statistics from chat history such as message breakdown and reactions 
            received.

            Args:
                None

            Returns:
                - summary (pd.DataFrame): DataFrame containing summary statistics for each participant.
                - fig (Figure): Plotly figure containing various subplots visualising the summary statistics.
        """

        # Find messages sent and reactions received aggregations
        summary = self.msgs_only.groupby('sender_name').agg(messages_sent = ('timestamp_ms', 'count'), 
                                                            reacts_received = ('reacts_count', 'sum'),
                                                            messages_with_reacts = ('reacts_count', lambda x: (x > 0).sum()))

        # Find number of reactions given
        reacts_exploded = self.msgs_only.explode('reactions').dropna(subset = ['reactions'])
        summary['Reactions Given'] = reacts_exploded['reactions'].apply(lambda x: x['actor']).value_counts()

        # Find ratios for reactions per message and number of messages before receiving a reaction
        summary['Reactions Received per Message'] = summary['reacts_received'] / summary['messages_sent']
        summary['Messages Sent per Reaction'] = summary['messages_sent'] / summary['messages_with_reacts']

        # Find number of emojis sent
        summary['Emojis Sent'] = self.texts_only.groupby('sender_name').agg(emojis_sent = ('emojis_count', 'sum'))

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
        summary.at['Chat Aggregate', 'Reactions Received per Message'] = summary.at['Chat Aggregate', 'reacts_received'] / summary.at['Chat Aggregate', 'messages_sent']
        summary.at['Chat Aggregate', 'Messages Sent per Reaction'] = summary.at['Chat Aggregate', 'messages_sent'] / summary.at['Chat Aggregate', 'messages_with_reacts']

        # Rename columns for consistency
        summary = summary.rename_axis('Participant')
        summary = summary.rename(columns = {'messages_sent': 'Messages Sent', 'reacts_received': 'Reactions Received', 'messages_with_reacts': 'Messages that Received Reactions'})

        # Add colour palette to distinguish between each participant
        summary['palette'] = px.colors.qualitative.Plotly[:len(summary.index)]

        # Create subplots to display summarised stats from chat
        fig = make_subplots(rows = 2, cols = 2, specs = [[{'type': 'domain'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
                            subplot_titles = ('Breakdown of Messages Sent', 'Reactions Received per Message', 'Total Emojis Sent', 'Calls over Time (by Month)'))
        
        # Plot 1 (pie chart)
        fig.add_trace(go.Pie(labels = summary.index[:-1], values = summary['Messages Sent'][:-1], marker = dict(colors = summary['palette'][:-1].tolist()), textinfo = 'label+percent',
                             textposition = 'outside', hole = 0.2), row = 1, col = 1)

        # Plot 2 (bar plot)
        # Sort table
        reacts_data = summary.sort_values(by = 'Reactions Received per Message', ascending = False)

        # Plot chart
        fig.add_trace(go.Bar(x = reacts_data.index,  y = reacts_data['Reactions Received per Message'], marker_color = reacts_data['palette'], text = reacts_data['Reactions Received per Message'], 
                      textposition = 'outside', texttemplate = '%{text:.2f}'), row = 1, col = 2)

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
        fig.update_layout(showlegend = False, margin = dict(t = 100, l = 50, b = 50, r = 50), font = dict(size = 12), height = 900)
        fig.update_xaxes(tickangle = 0, tickfont = dict(size = 10))
        fig.update_yaxes(tickfont = dict(size = 12))
        for annotation in fig['layout']['annotations']:
            annotation['yanchor'] = 'bottom'
            annotation['y'] = annotation['y'] + 0.03  
            annotation['font'] = dict(size = 16)

        # Store summary for use in react analysis
        self.summary = summary

        # Drop colour_palette column as it is no longer needed
        summary = summary.drop('palette', axis = 1)

        return summary, fig

    def cumulative_messages_over_time(self):

        """
            Computes and visualises cumulative message counts over time.

            Args:
                None

            Returns:
                - fig (Figure): Plotly figure showing overall cumulative message count over time.
                - racecar_output (Figure): Interactive race plot showing cumulative message count by participant over time.

            Notes:
                - Modified raw source code (__get_colors() method in plots.py) in raceplotly package as I wasn't able to manually 
                  select the colours in the bar plot.
                - Have raised an issue under https://github.com/lucharo/raceplotly/issues/22.
        """
        
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

        # Create racecar plot to dynamically visualise changes in cumulative count of messages (address colour scheme)
        rgb_colours = self.convert_hex_to_rgb(px.colors.qualitative.Plotly[:len(self.participants)])
        plot_colours = dict(zip(self.participants, rgb_colours))
        racecar = barplot(cum_msg_counts, item_column = 'Participant', value_column = 'Cumulative Count', time_column = 'Date', item_color = plot_colours)
        racecar_output = racecar.plot(title = 'Cumulative Message Count by Participant', item_label = 'Participant', value_label = 'Cumulative Count', 
                                      frame_duration = 75)

        return fig, racecar_output

    def raw_messages_over_time(self):

        """
            Computes and visualises raw message counts by week.

            Args:
                None

            Returns:
                - fig (Figure): Interactive Plotly line chart showing raw message counts by participant and week.
        """

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

        """
            Analyses and visualises chat activity across different time periods.

            Args:
                None

            Returns:
                - fig (Figure): Plotly figure with subplots showing interactions over various time periods.
                - heatmap (Figure): Plotly heatmap showing chat activity by hour of day and day of week.
                - extremes (Figure): Plotly figure with subplots showing the top 10 most and least active days.
        """

        # Create list of time periods to consider
        plot_period = ['hour_of_day', 'day_of_week', 'week_number', 'month', 'quarter_number', 'year'] 

        # Find all chat interactions for each participant based on each time period
        # This looks at all content, not only messages 
        # Create subplots to display summarised stats from chat
        rows = 3
        cols = 2
        titles = []
        for period in plot_period:
            titles.append(' '.join([word.capitalize() if word != 'of' else word for word in period.split('_')]))
        
        fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles)
        for idx, period in enumerate(plot_period):
            clean_period = ' '.join([word.capitalize() if word != 'of' else word for word in period.split('_')])

            # Group interactions by period and participant
            content_counts = self.chat_history.pivot_table(index = period, columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

            # Unstack pivot table
            plot_data = content_counts.stack().reset_index()
            plot_data.columns = [clean_period, 'Participant', 'Interactions']

            # Create subplot and add to figure
            area_fig = px.area(plot_data, x = clean_period, y = 'Interactions', color = 'Participant', color_discrete_sequence = px.colors.qualitative.Plotly)
            for trace in area_fig.data:

                # Only show legend for last plot 
                if idx == len(plot_period) - 1:
                    trace.showlegend = True
                else:
                    trace.showlegend = False
                    
                fig.add_trace(trace, row = (idx // cols) + 1, col = (idx % cols) + 1)

        fig.update_layout(margin = dict(t = 100, l = 50, b = 50, r = 50), font = dict(size = 12), height = 1200, title_text = 'Interactions Over Different Time Periods',
                          title_font = dict(size = 16))
        
        # Group number of chat interactions by hour of day and day of week
        activity_data = self.chat_history.pivot_table(index = 'hour_of_day', columns = 'day_of_week', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Plot heatmap of total chat activity
        heatmap = go.Figure(data = go.Heatmap(z = activity_data.values, x = activity_data.columns, y = activity_data.index, text = activity_data.values,
                        texttemplate = '%{text:.0f}', colorscale = 'Viridis'))
        heatmap.update_layout(title = 'Heatmap of Chat Activity by Hour and Day of Week', xaxis_title = 'Day of Week', yaxis_title = 'Hour of Day', height = 800)

        # Find the top 10 most and least active days in the chat
        activity_by_date = self.chat_history.pivot_table(index = 'date', columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)
        activity_by_date['total'] = activity_by_date.sum(axis = 1)
        top_10_most_active = activity_by_date.nlargest(10, 'total')
        top_10_most_active_total = top_10_most_active['total']
        top_10_most_active = top_10_most_active.drop(columns = ['total'])
        top_10_least_active = activity_by_date.nsmallest(10, 'total')
        top_10_least_active_total = top_10_least_active['total']
        top_10_least_active = top_10_least_active.drop(columns = ['total'])

        # Unstack pivot tables
        top_10_most_active_plot = top_10_most_active.stack().reset_index()
        top_10_most_active_plot.columns = ['Date', 'Participant', 'Interactions']
        top_10_least_active_plot = top_10_least_active.stack().reset_index()
        top_10_least_active_plot.columns = ['Date', 'Participant', 'Interactions']

        # Plot figure with subplots for most and least active dates
        extremes = make_subplots(rows = 1, cols = 2, subplot_titles = ('Top 10 Most Active Days', 'Top 10 Least Active Days'))
        
        # Plot 1 (stacked bar chart)
        most = px.bar(top_10_most_active_plot, x = 'Date', y = 'Interactions', color = 'Participant', color_discrete_sequence = px.colors.qualitative.Plotly)    
        for trace in most.data:
            trace.showlegend = False
            extremes.add_trace(trace, row = 1, col = 1)

        # Add total interactions as text for each date
        for _, date in enumerate(top_10_most_active_total.index):
            extremes.add_annotation(text = f'{top_10_most_active_total[date]}', x = date, y = top_10_most_active_total[date], showarrow = False, yanchor = 'bottom',
                                    row = 1, col = 1)

        # Plot 2 (stacked bar chart)
        least = px.bar(top_10_least_active_plot, x = 'Date', y = 'Interactions', color = 'Participant', color_discrete_sequence = px.colors.qualitative.Plotly)
        for trace in least.data:
            extremes.add_trace(trace, row = 1, col = 2)

        # Add total interactions as text for each date
        for _, date in enumerate(top_10_least_active_total.index):
            extremes.add_annotation(text = f'{top_10_least_active_total[date]}', x = date, y = top_10_least_active_total[date], showarrow = False, yanchor='bottom',
                                    row = 1, col = 2)

        extremes.update_layout(barmode = 'stack')
        extremes.update_xaxes(type = 'category', categoryorder = 'array', categoryarray = top_10_most_active_plot['Date'], row = 1, col = 1)
        extremes.update_xaxes(type = 'category', categoryorder = 'array', categoryarray = top_10_least_active_plot['Date'], row = 1, col = 2)

        return fig, heatmap, extremes
    
    def react_analysis(self):

        """
            Analyses and visualises reaction data in the chat by participant and over time.

            Args:
                None

            Returns:
                - reactions_given_participant (pd.DataFrame): DataFrame showing the top 10 reactions given by each participant.
                - reactions_received_participant (pd.DataFrame): DataFrame showing the top 10 reactions received by each participant.
                - fig (Figure): Plotly figure with subplots displaying various high-level views of reactions.
                - heatmap (Figure): Plotly heatmap showing reactions sent and received by participants.
                - top_reacted_msgs (pd.DataFrame): DataFrame containing the 25 most reacted-to messages.
                - top_reacted_msgs_participant (pd.DataFrame): DataFrame containing the top reacted message for each participant.
        """

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

        # Plot figure with subplots containing number of reacts given and received
        fig = make_subplots(rows = 2, cols = 2, specs = [[{'type': 'domain'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
                            subplot_titles = ('Breakdown of Reactions Given', 'Proportion of Messages that Received Reactions vs. Reactions Given', 'Messages Sent per Reaction', 
                                              'Reactions Sent over Time (by Month)'))
        
        # Plot 1 (pie chart)
        fig.add_trace(go.Pie(labels = self.summary.index[:-1], values = self.summary['Reactions Given'][:-1], marker = dict(colors = self.summary['palette'][:-1].tolist()), textinfo = 'label+percent',
                             textposition = 'outside', hole = 0.2, showlegend = False), row = 1, col = 1)

        # Plot 2 (grouped bar plot)
        # Calculate proportion of reactions to messages given and received
        self.summary['Reactions Received'] = 100 * self.summary['Messages that Received Reactions'] / self.summary['Messages Sent']
        self.summary['Reactions Given'] = 100 * self.summary['Reactions Given'] / self.summary.loc['Chat Aggregate', 'Messages Sent']

        # Plot chart
        fig.add_trace(go.Bar(x = self.summary.index[:-1],  y = self.summary['Reactions Received'][:-1], text = self.summary['Reactions Received'][:-1], textposition = 'outside',
                             texttemplate = '%{text:.2f}%', marker_color = 'rgba(255, 100, 102, 0.8)', name = 'Reactions Received %', showlegend = True), row = 1, col = 2)
        fig.add_trace(go.Bar(x = self.summary.index[:-1],  y = self.summary['Reactions Given'][:-1], text = self.summary['Reactions Given'][:-1], textposition = 'outside', 
                             texttemplate = '%{text:.2f}%', marker_color = 'rgba(100, 200, 102, 0.8)', name = 'Reactions Given %', showlegend = True), row = 1, col = 2)
        fig.update_layout(barmode = 'group')

        # Plot 3 (bar plot)
        # Sort table
        reacts_data1 = self.summary.sort_values(by = 'Messages Sent per Reaction', ascending = True)
        
        # Plot chart and horizontal line for chat aggregate
        fig.add_trace(go.Bar(x = reacts_data1.index,  y = reacts_data1['Messages Sent per Reaction'], marker_color = reacts_data1['palette'], text = reacts_data1['Messages Sent per Reaction'], 
                             textposition = 'outside', texttemplate = '%{text:.2f}', showlegend = False), row = 2, col = 1)

        # Plot 4 (line chart)
        # Find number of reactions given by month
        reacts_data2 = reacts_exploded.groupby('month_start').size()
                            
        # Plot chart
        fig.add_trace(go.Scatter(x = reacts_data2.index, y = reacts_data2.values, mode = 'lines+markers', showlegend = False), row = 2, col = 2)        
        
        # Update layout
        fig.update_layout(margin = dict(t = 100, l = 50, b = 50, r = 50), font = dict(size = 12), height = 900)
        fig.update_xaxes(tickangle = 0, tickfont = dict(size = 10))
        fig.update_yaxes(tickfont = dict(size = 12))
        for annotation in fig['layout']['annotations']:
            annotation['yanchor'] = 'bottom'
            annotation['y'] = annotation['y'] + 0.03  
            annotation['font'] = dict(size = 16)

        # Calculate ratio of messages reacted to by each participant for other participants in the chat
        reacts_grid = reacts_exploded.pivot_table(index = 'actor', columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)
        reacts_grid_ratio = 100 * reacts_grid.div(reacts_grid.sum(axis = 0), axis = 'columns')

        # Plot heatmap of inter-group reaction trends
        heatmap = go.Figure(data = go.Heatmap(z = reacts_grid_ratio.values, x = reacts_grid_ratio.columns, y = reacts_grid_ratio.index, text = reacts_grid_ratio.values,
                texttemplate = '%{text:.2f}%', colorscale = 'Viridis'))
        heatmap.update_layout(title = 'Heatmap of Reactions Sent and Received by Participants', xaxis_title = 'Received by', yaxis_title = 'Given by', height = 800)
        
        # Find the 25 most reacted to messages all-time (in case of tie-breakers, consider the most recent message)
        reacted_msgs_sorted = self.msgs_only.sort_values(by = ['reacts_count', 'date', 'sender_name'], ascending = [False, False, True])
        top_reacted_msgs = reacted_msgs_sorted[['date', 'sender_name', 'content', 'reacts_count', 'reactions', 'emojis_count', 'media_type']].head(25)
        top_reacted_msgs['reactions'] = top_reacted_msgs['reactions'].astype('str')
        top_reacted_msgs.columns = ['Date', 'Participant', 'Content', 'Count of Reactions', 'Reactions', 'Count of Emojis', 'Media Type']
        top_reacted_msgs.reset_index().drop(columns = ['index']).set_index('Date')

        # Find the top reacted message for each participant all-time (in case of tie-breakers, consider the most recent message)
        top_reacted_msgs_participant = reacted_msgs_sorted.groupby('sender_name').first().reset_index()
        top_reacted_msgs_participant = top_reacted_msgs_participant[['date', 'sender_name', 'content', 'reacts_count', 'reactions', 'emojis_count', 'media_type']]
        top_reacted_msgs_participant['reactions'] = top_reacted_msgs_participant['reactions'].astype('str')
        top_reacted_msgs_participant.columns = ['Date', 'Participant', 'Content', 'Count of Reactions', 'Reactions', 'Count of Emojis', 'Media Type']
        top_reacted_msgs_participant.reset_index().drop(columns = ['index']).set_index('Date')

        return reactions_given_participant, reactions_received_participant, fig, heatmap, top_reacted_msgs, top_reacted_msgs_participant

    def word_analysis(self):

        """
            Analyses and visualises the words and sentimental/emotional tone used in text messages.

            Args:
                None

            Returns:
                - word_fig (Figure): A bar plot of average message lengths by participant.
                - cloud_fig (Figure): A figure with word clouds for overall chat and individual participants.
                - sentiment_fig (Figure): A grouped bar chart of sentiment proportions by participant.
                - emotion_fig (Figure): A grouped bar chart of emotion proportions by participant.

            Notes:
                - No word processing is done for sentiment/emotion analysis (e.g. removing stopwords), as raw text
                  offers more insights.
                - For example, if NLTK stopwords were removed, the phrase "I am not having a great day" would become "great day" 
                  which evidently evokes a completely different sentiment/emotion to the original text.
                - Emojis also affect the output of sentiment/emotion analysis.
        """

        # Get word length aggregates for text messages sent in chat
        word_summary_chat = self.texts_only.agg(average_words = ('word_count', 'mean'))
        
        # Get word length aggregates for text messages sent by each participant
        word_summary_participant = self.texts_only.groupby('sender_name').agg(average_words = ('word_count', 'mean'))
        
        # Combine data and create bar plot of message lengths
        word_summary_participant.loc['Chat Aggregate'] = word_summary_chat.values[0]
        word_summary_participant['palette'] = self.summary['palette']
        word_summary_participant = word_summary_participant.sort_values(by = ['average_words'], ascending = False)
        word_fig = go.Figure(data = [go.Bar(x = word_summary_participant.index, y = word_summary_participant['average_words'], marker_color = word_summary_participant['palette'], 
                     text = word_summary_participant['average_words'], textposition = 'outside', texttemplate = '%{text:.2f}', showlegend = False)])
        word_fig.update_layout(title = 'Average Message Length by Words Used')

        # Process text to normalise text, and remove stop words, emojis, and punctuation for word cloud plots
        self.texts_only['processed_text'] = self.texts_only['content'].apply(self.process_text)
             
        # Create figure with subplots for wordclouds across all participants and group overall
        cloud_fig = go.Figure()
        rows = (len(self.participants) // 2) + 1
        cols = 2
        titles = ['Top 75 Most Words Used by Chat Overall']
        [titles.append('Top 75 Most Words Used by ' + name) for name in self.participants]
        cloud_fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles)

        # Build wordcloud for group overall (top 75 words)
        most_common_chat = Counter(' '.join(self.texts_only['processed_text']).split()).most_common(75)
        wordcloud_chat = WordCloud(width = 1600, height = 1000, background_color = 'white').generate_from_frequencies(dict(most_common_chat))
        
        # Convert word cloud to image, resize image with bilinear interpolation, and add to figure
        wordcloud_image = wordcloud_chat.to_image()
        wordcloud_image = wordcloud_image.resize((1600, 1000), Image.BILINEAR)
        wordcloud_array = np.array(wordcloud_image)
        cloud_fig.add_trace(go.Image(z = wordcloud_array), row = 1, col = 1)

        # Build word clouds for each participant (top 75 words)
        for idx, person in enumerate(self.participants):
            words_participant = self.texts_only[self.texts_only['sender_name'] == person]['processed_text']
            most_common_participant = Counter(' '.join(words_participant).split()).most_common(75)
            wordcloud_participant = WordCloud(width = 1600, height = 1000, background_color = 'white').generate_from_frequencies(dict(most_common_participant))
            
            # Convert word cloud to image, resize image with bilinear interpolation, and add to figure
            wordcloud_image_participant = wordcloud_participant.to_image()
            wordcloud_image_participant = wordcloud_image_participant.resize((1600, 1000), Image.BILINEAR)
            wordcloud_array_participant = np.array(wordcloud_image_participant)
            cloud_fig.add_trace(go.Image(z = wordcloud_array_participant), row = ((idx + 1) // cols) + 1, col = ((idx + 1) % cols) + 1)
       
        cloud_fig.update_xaxes(showgrid = False, zeroline = False, showticklabels = False, visible = False)
        cloud_fig.update_yaxes(showgrid = False, zeroline = False, showticklabels = False, visible = False)
        cloud_fig.update_layout(margin = dict(t = 40, l = 30, b = 40, r = 30), font = dict(size = 12), height = 2100)
        for annotation in cloud_fig['layout']['annotations']:
            annotation['yanchor'] = 'bottom'
            annotation['y'] = annotation['y'] + 0.01  
            annotation['font'] = dict(size = 16)

        # Convert column of messages to Dataset object for efficient model processing
        data_dict = {'text': self.texts_only['content'].tolist()}
        text_dataset = Dataset.from_dict(data_dict)

        # Perform sentiment analysis on each text message using pre-trained transformers model
        # Retain the label with the highest score
        self.texts_only['sentiment'] = self.sentiment_analysis(text_dataset, self.batch_size)

        # Perform emotion analysis on each text message using pre-trained transformers model
        # Retain the label with the highest score
        self.texts_only['emotion'] = self.emotion_analysis(text_dataset, self.batch_size)

        # Generate sentiment and emotion datasets for plotting
        sentiment_proportions = self.texts_only.groupby(['sender_name', 'sentiment']).size().unstack(fill_value = 0).apply(lambda x: 100 * x / x.sum(), axis = 1)
        sentiment_proportions = sentiment_proportions.reset_index().melt(id_vars = 'sender_name', var_name = 'sentiment', value_name = 'proportion')
        sentiment_proportions.columns = ['Participant', 'Sentiment', 'Proportion (%)']

        emotion_proportions = self.texts_only.groupby(['sender_name', 'emotion']).size().unstack(fill_value = 0).apply(lambda x: 100 * x / x.sum(), axis = 1)
        emotion_proportions = emotion_proportions.reset_index().melt(id_vars = 'sender_name', var_name = 'emotion', value_name = 'proportion')
        emotion_proportions.columns = ['Participant', 'Emotion', 'Proportion (%)']

        # Plot grouped bar chart for sentiment analysis
        sentiment_fig = px.bar(sentiment_proportions, x = 'Participant', y = 'Proportion (%)', color = 'Sentiment', color_discrete_sequence = px.colors.qualitative.Plotly,
                               text = 'Proportion (%)')    
        sentiment_fig.update_traces(texttemplate = '%{text:.2f}%', textposition = 'outside', textfont = dict(size = 12))
        sentiment_fig.update_xaxes(tickangle = 0, tickfont = dict(size = 10))
        sentiment_fig.update_layout(title = 'Sentiment Analysis by Participant', barmode = 'group')

        # Plot grouped bar chart for emotion analysis
        emotion_fig = px.bar(emotion_proportions, x = 'Participant', y = 'Proportion (%)', color = 'Emotion', color_discrete_sequence = px.colors.qualitative.Plotly,
                             text = 'Proportion (%)')        
        emotion_fig.update_traces(texttemplate = '%{text:.2f}%', textposition = 'outside', textfont = dict(size = 10))
        emotion_fig.update_xaxes(tickangle = 0, tickfont = dict(size = 10))
        emotion_fig.update_layout(title = 'Emotion Analysis by Participant', barmode = 'group')
        
        return word_fig, cloud_fig, sentiment_fig, emotion_fig

    @staticmethod
    def convert_hex_to_rgb(hex_list):

        """
            Converts a list of hexadecimal color codes to RGB format.

            Args:
                hex_list (list): A list of color codes in hexadecimal format (e.g., '#FF5733').

            Returns:
                rgb_list (list): A list of color codes in RGB format (e.g., 'rgb(255, 87, 51)').
        """

        # Iterate through list and convert hex string to rgb format
        rgb_list = []
        for hex_colour in hex_list:
            hex_str = hex_colour.lstrip('#')
            rgb_list.append('rgb' + str(tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))))

        return rgb_list

    @staticmethod
    def process_text(text):

        """
            Processes and normalises text for further analysis.

            Args:
                text (str): The text to be processed.

            Returns:
                text (str): The processed text with stop words and non-alphanumeric characters removed.
        """

        # Convert text to lower case
        text = text.lower()

        # Remove non-alphanumeric characters (punctuation, emojis etc.)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove stop words 
        text = ' '.join([word for word in text.split() if word not in stopwords])

        return text
    
    @staticmethod
    def sentiment_analysis(dataset, batch_size):

        """
            Analyses sentiment of the provided text data using a pre-trained model.

            Args:
                dataset (Dataset): A Dataset object containing text data.
                batch_size (int): The batch size for processing the text data.

            Returns:
                labels (list): A list of sentiment labels for each text entry in the dataset.

            Notes:
                - Uses the cardiffnlp/twitter-roberta-base-sentiment-latest model 
                  (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
                - Each text is classified into sentiment categories (e.g., positive, negative, neutral).
        """

        # Intialise transformers model for sentiment analysis
        sentiment_model = pipeline('text-classification', model = 'cardiffnlp/twitter-roberta-base-sentiment-latest', 
                                   device = selected_device, batch_size = batch_size, truncation = True, max_length = 512)

        # Run model
        results = sentiment_model(KeyDataset(dataset, 'text'))

        # Collect results and track progress
        labels = []
        for result in tqdm(results, total = len(dataset), position = 0, leave = True, desc = 'Sentiment Analysis'):
            labels.append(result['label'].title())

        return labels 
    
    @staticmethod
    def emotion_analysis(dataset, batch_size):

        """
            Analyses emotion of the provided text data using a pre-trained model.

            Args:
                dataset (Dataset): A Dataset object containing text data.
                batch_size (int): The batch size for processing the text data.

            Returns:
                labels (list): A list of emotion labels for each text entry in the dataset.

            Notes:
                - Uses the michellejieli/emotion_text_classifier model
                  (https://huggingface.co/michellejieli/emotion_text_classifier).
                - Each text is classified into emotion categories (e.g., joy, sadness, anger).
        """

        # Intialise transformers model for emotion analysis
        emotion_model = pipeline('text-classification', model = 'michellejieli/emotion_text_classifier', 
                                 device = selected_device, batch_size = batch_size, truncation = True, max_length = 512)

        # Run model
        results = emotion_model(KeyDataset(dataset, 'text'))

        # Collect results and track progress
        labels = []
        for result in tqdm(results, total = len(dataset), position = 0, leave = True, desc = 'Emotion Analysis'):
            labels.append(result['label'].title())

        return labels 
    