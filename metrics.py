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
        start_time = datetime.strptime(f"{time_id[0]:02d}:00", "%H:%M")
        end_time = start_time + timedelta(hours = 1)
        start_time_str = start_time.strftime('%I:%M%p').lstrip('0')
        end_time_str = end_time.strftime('%I:%M%p').lstrip('0')
        most_active_time = time_id[1] + ' ' + start_time_str + ' to ' + end_time_str

        # Find average text message length in words
        avg_words = round(self.texts_only['word_count'].mean(), 2)

        # Find total multimedia messages sent
        multimedia = len(self.chat_history[self.chat_history['media_type'].isin(multimedia_filters)])

        # Combine outputs
        aggs = [no_participants, totals, avg_contributions, avg_delay, most_active_participant, reactions, most_active_time,
                avg_words, multimedia]
        
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
        summary['palette'] = sns.color_palette('pastel', len(summary.index))

        # Create subplots to display summarised stats from chat
        # Plot 1 (pie chart)
        fig, axs = plt.subplots(2, 2, figsize = (10, 10))
        axs[0, 0].pie(summary['Messages Sent'][:-1], labels = summary.index[:-1], autopct='%1.0f%%', colors = summary['palette'][:-1])
        axs[0, 0].set_title('Breakdown of Messages Sent')

        # Plot 2 (bar plot)
        # Extract chat aggregate value to plot
        chat_aggregate_value = summary.loc['Chat Aggregate', 'Reacts Received per Message']

        # Filter out chat aggregate and sort table
        reacts_data = summary.drop('Chat Aggregate').sort_values(by = 'Reacts Received per Message', ascending = False)

        # Plot chart and horizontal line for chat aggregate
        sns.barplot(x = reacts_data.index, y = reacts_data['Reacts Received per Message'], palette = reacts_data['palette'].tolist(), ax = axs[0, 1])
        axs[0, 1].axhline(y = chat_aggregate_value, color = 'grey', linestyle = '--', label = 'Chat Aggregate')
        axs[0, 1].set_title('Reacts Received per Message')
        axs[0, 1].set_xlabel('Participant')
        axs[0, 1].set_ylabel('Reacts Received per Message')
        for label in axs[0, 1].containers:
            axs[0, 1].bar_label(label, padding = 3)

        axs[0, 1].legend()

        # Plot 3 (horizontal bar plot)
        # Filter out chat aggregate and sort table
        emojis_data = summary.drop('Chat Aggregate').sort_values(by = 'Emojis Sent', ascending = False)

        # Plot chart
        sns.barplot(x = emojis_data['Emojis Sent'], y = emojis_data.index, palette = emojis_data['palette'].tolist(), ax = axs[1, 0])
        axs[1, 0].set_title('Emojis Sent')
        axs[1, 0].set_xlabel('Emojis Sent')
        axs[1, 0].set_ylabel('Participants')
        for label in axs[1, 0].containers:
            axs[1, 0].bar_label(label, padding = 3)

        # Plot 4 (line chart)
        # Filter for calls only
        calls_data = self.chat_history[self.chat_history['content_type'] == 'Started Call'].groupby('month_start').size()
    
        # Plot chart
        calls_data.plot(kind = 'line', marker = 'x', ax = axs[1, 1])
        axs[1, 1].set_title('Calls over Time (by Month)')
        axs[1, 1].set_xlabel('Month')
        axs[1, 1].set_ylabel('Number of Calls')

        plt.tight_layout()

        # Drop colour_palette column as it is no longer needed
        summary = summary.drop('palette', axis = 1)

        return summary, fig

    def cumulative_messages_over_time(self):
        
        # Find cumulative count of messages sent by each participant by each date
        self.msgs_only['cumulative_count_msgs'] = self.msgs_only.groupby('sender_name').cumcount() + 1
        pivot_msg_counts = self.msgs_only.pivot_table(index = 'date', columns = 'sender_name', values = 'cumulative_count_msgs', aggfunc = 'last')
        pivot_msg_counts.fillna(method = 'ffill', inplace = True)

        # Find total messages sent over time for the user-defined time period
        total_msgs_over_time = pivot_msg_counts.sum(axis = 1)

        # Unstack pivot table 
        cum_msg_counts = pivot_msg_counts.stack().reset_index()
        cum_msg_counts.columns = ['date', 'sender_name', 'cumulative_count_msgs']

        # Plot figure with subplots for total cumulative message count and pivot table data for each participant
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 8), gridspec_kw = {'height_ratios': [1, 2]})

        # Plot total message count
        total_msgs_over_time.plot(kind = 'line', linewidth = 1.5, ax = ax1)
        ax1.set_title('Total Message Count')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Count')
        ax1.grid(True)

        # Plot participant message count
        pivot_msg_counts.plot(kind = 'line', linewidth = 1.5, ax = ax2)
        ax2.set_title('Message Count by Participant')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Count')
        ax2.legend(title = 'Participant', loc = 'best', fontsize = 8).get_title().set_fontsize('8')
        ax2.grid(True)

        fig.suptitle('Cumulative Count of Messages Sent', fontsize = 16)
        plt.subplots_adjust(top = 0.5)
        plt.tight_layout()

        # Create racecar plot to dynamically visualise changes in cumulative count of messages
        racecar = barplot(cum_msg_counts, item_column = 'sender_name', value_column = 'cumulative_count_msgs', time_column = 'date', top_entries = 10)
        racecar_output = racecar.plot(title = 'Cumulative Count of Messages Sent', item_label = 'Participant', value_label = 'Cumulative Count', 
                                      frame_duration = 75)

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
        fig.update_layout(
            updatemenus = [
                dict(
                    type = "buttons",
                    x = 1.1,
                    y = 1.25,
                    buttons = [
                        dict(label = "Show All",
                            method = "update",
                            args=[{"visible": [True] * len(raw_msg_counts['Participant'].unique())}])
                    ]
                )
            ]
        )
        
        return fig
    
    def chat_activity(self, time_period):

        # Create dictionary to map time period to user-defined input
        plot_period = {'hour':'hour_of_day', 'day':'day_of_week', 'week':'week', 'month':'month', 'quarter':'quarter', 'year':'year'} 

        # Find all chat interactions for each participant based on user-defined time period
        # This looks at all content, not only messages 
        frequency = plot_period[time_period]
        content_counts = self.chat_history.pivot_table(index = frequency, columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Plot user interactions
        interactions = content_counts.plot(kind = 'bar', stacked = True, figsize = (12, 6))
        plt.xlabel(time_period.title())
        plt.ylabel('Count')
        plt.title('Total Chat Interactions (by ' + time_period.title() + ')')
        plt.legend(title = 'Participant', loc = 'best', fontsize = 8).get_title().set_fontsize('8')
        plt.grid(True)
        plt.tight_layout()

        # Save plot (create new directory to save output if not available already)
        chat_activity_dir = os.path.join(self.save_dir, 'chat_activity')
        if not os.path.exists(chat_activity_dir):
            os.makedirs(chat_activity_dir)

        interactions.figure.savefig(os.path.join(chat_activity_dir, 'chat_interactions.jpg'))

        # Group number of chat interactions by hour of day and day of week
        activity_data = self.chat_history.pivot_table(index = 'hour_of_day', columns = 'day_of_week', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Plot heatmap of total chat activity
        plt.figure(figsize = (12, 10))
        activity_heatmap = sns.heatmap(activity_data, cmap = 'YlGnBu', annot = True, fmt = 'd', cbar = True)
        plt.title('Total Chat Activity Heatmap')
        plt.xlabel('Day of Week')
        plt.ylabel('Hour of Day')

        # Save plot
        activity_heatmap.figure.savefig(os.path.join(chat_activity_dir, 'chat_activity_heatmap.jpg'))

        # Find the top 10 most and least active days in the chat
        activity_by_date = self.chat_history.pivot_table(index = 'date', columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)
        activity_by_date['total'] = activity_by_date.sum(axis = 1)
        top_10_most_active = activity_by_date.sort_values(by = 'total', ascending = False).head(10).drop(columns = ['total'])
        top_10_most_active.index = top_10_most_active.index.astype('category')
        top_10_least_active = activity_by_date.sort_values(by = 'total', ascending = True).head(10).drop(columns = ['total'])
        top_10_least_active.index = top_10_least_active.index.astype('category')

        # Plot figure with subplots for most and least active dates
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 6))

        # Plot total message count
        top_10_most_active.plot(kind = 'bar', stacked = True, ax = ax1)
        ax1.set_title('Top 10 Most Active Dates')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Count of Interactions')
        plt.xticks(rotation = 0)
        ax1.grid(True)

        # Plot participant message count
        top_10_least_active.plot(kind = 'bar', stacked = True, ax = ax2)
        ax2.set_title('Top 10 Most Active Dates')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Count of Interactions')
        plt.xticks(rotation = 0)
        ax2.grid(True)

        fig.suptitle('Top 10 Most and Least Active Dates by Total Chat Interactions', fontsize = 16)
        # fig.legend(title = 'Participant', fontsize = 8).get_title().set_fontsize('8')
        plt.subplots_adjust(top = 0.5)
        plt.tight_layout()    

        # Save plot
        fig.savefig(os.path.join(chat_activity_dir, 'most_least_activity.jpg'))

        return
    
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
        reactions_given_participant[''] = pd.NA
        reactions_given_participant['Group Aggregate'] = total_count_reactions.index

        # Find count of each distinct react received by each participant (top 10)
        reactions_received = reacts_exploded.groupby(['sender_name', 'reaction']).size().reset_index(name = 'count')
        reactions_received_sorted = reactions_received.sort_values(['sender_name', 'count'], ascending = [True, False])
        top_reactions_received = reactions_received_sorted.groupby('sender_name').head(10)
        top_reactions_received = (top_reactions_received.reset_index(drop = True)
                           .set_index(top_reactions_received.groupby('sender_name').cumcount() + 1))
        reactions_received_participant = top_reactions_received.pivot(columns = 'sender_name', values = 'reaction').fillna('')
        reactions_received_participant[''] = pd.NA
        reactions_received_participant['Group Aggregate'] = total_count_reactions.index

        # Save outputs as separate sheets in Excel file (create new directory to save output if not available already)
        reactions_dir = os.path.join(self.save_dir, 'reactions')
        if not os.path.exists(reactions_dir):
            os.makedirs(reactions_dir)
        
        with pd.ExcelWriter(os.path.join(reactions_dir, 'react_analysis.xlsx')) as writer:
            reactions_given_participant.to_excel(writer, sheet_name = 'Top 10 Reacts Given', index = True)
            reactions_received_participant.to_excel(writer, sheet_name = 'Top 10 Reacts Received', index = True)

        # Calculate ratio of messages reacted to by each participant for other participants in the chat
        reacts_grid = reacts_exploded.pivot_table(index = 'actor', columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)
        reacts_grid_ratio = 100 * reacts_grid.div(reacts_grid.sum(axis = 0), axis = 'columns')

        # Create heatmap
        react_heatmap = sns.heatmap(reacts_grid_ratio, cmap = 'YlGnBu', annot = True, fmt = '.4f', cbar = True)
        plt.title('Reactions Heatmap (% Contribution by each Participant)', fontsize = 18)
        plt.xlabel('Received by', fontsize = 16)
        plt.ylabel('Given by', fontsize = 16)
        plt.yticks(fontsize = 12, rotation = 0)
        plt.xticks(fontsize = 12, rotation = 0)  

        # Save plot
        react_heatmap.figure.savefig(os.path.join(reactions_dir, 'reactions_heatmap.jpg'))

        # Find the 10 most reacted to messages all-time
        reacted_msgs_sorted = self.msgs_only.sort_values(by = ['reacts_count', 'date', 'sender_name'], ascending = [False, False, True])
        top_reacted_msgs = reacted_msgs_sorted[['sender_name', 'content', 'reactions', 'date']].head(10)

        # Find the top reacted message for each participant all-time
        top_reacted_msgs_participant = reacted_msgs_sorted.groupby('sender_name').first().reset_index()[['sender_name', 'content', 'reactions', 'date']]
        
        # Save outputs as separate sheets in Excel file
        with pd.ExcelWriter(os.path.join(reactions_dir, 'top_reacted_messages.xlsx')) as writer:
            top_reacted_msgs.to_excel(writer, sheet_name = 'Top 10 Reacted Messages', index = True)
            top_reacted_msgs_participant.to_excel(writer, sheet_name = 'Most Reacted Messages by Participant', index = True)

        return

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
        word_summary_chat = self.texts_only.agg(median_words = ('word_count', 'median'),
                                               average_words = ('word_count', 'mean'),
                                               max_words = ('word_count', 'max'))
        
        # Get word length aggregates for text messages sent by each participant
        word_summary_participant = self.texts_only.groupby('sender_name').agg(median_words = ('word_count', 'median'),
                                                                             average_words = ('word_count', 'mean'),
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
    