'''
    The following script contains various class methods to analyse Messenger
    chats and save outputs to the designated directory.
    Author: Rohit Rajagopal
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from raceplotly.plots import barplot
import seaborn as sns
from wordcloud import WordCloud


class AnalyseChat:

    def __init__(self, chat_history, save_dir):
        self.chat_history = chat_history
        self.save_dir = save_dir
        msgs_only_filter = ['Message', 'Shared Location']
        self.msgs_only = self.chat_history[self.chat_history['content_type'].isin(msgs_only_filter)]

    def summary_stats(self):

        # Find messages sent and reacts received aggregations
        summary = self.msgs_only.groupby('sender_name').agg(messages_sent = ('timestamp_ms', 'count'), 
                                                            reacts_received = ('reacts_count', 'sum'),
                                                            messages_with_reacts = ('reacts_count', lambda x: (x > 0).sum()),
                                                            emojis_sent = ('emojis_count', 'sum'))
        
        # Find number of reacts given
        reacts_exploded = self.msgs_only.explode('reactions').dropna(subset = ['reactions'])
        summary['reacts_given'] = reacts_exploded['reactions'].apply(lambda x: x['actor']).value_counts().values

        # Find ratios for reacts per message and number of messages before receiving a react
        summary['reacts_received_per_message'] = summary['reacts_received'] / summary['messages_sent']
        summary['messages_sent_per_react'] = summary['messages_sent'] / summary['messages_with_reacts']

        # Find media type aggregations
        media_filters = ['Other', 'Text']
        media_type_data = self.chat_history[~self.chat_history['media_type'].isin(media_filters)]
        media_pivot = media_type_data.pivot_table(index = 'sender_name', columns = 'media_type', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Find content type aggregations
        content_filters = ['Started Call', 'Left Chat', 'Added Member to Chat', 'Shared Location']
        content_type_data = self.chat_history[self.chat_history['content_type'].isin(content_filters)]
        content_pivot = content_type_data.pivot_table(index = 'sender_name', columns = 'content_type', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Consolidate final dataset by appending pivot tables above
        summary = pd.concat([summary, media_pivot, content_pivot], axis = 1, sort = True)

        # Add chat aggregates as last row
        summary.loc[''] = pd.Series([np.nan] * len(summary.columns), index = summary.columns, name = '')
        summary.loc['Chat Aggregate'] = summary.sum()
        summary.at['Chat Aggregate', 'reacts_received_per_message'] = summary.at['Chat Aggregate', 'reacts_received'] / summary.at['Chat Aggregate', 'messages_sent']
        summary.at['Chat Aggregate', 'messages_sent_per_react'] = summary.at['Chat Aggregate', 'messages_sent'] / summary.at['Chat Aggregate', 'messages_with_reacts']

        # Save output
        summary.to_csv(os.path.join(self.save_dir, 'summary.csv'), index = True)

        return summary
    
    def messages_over_time(self, time_period):
        
        # Create dictionary to map time period to user-defined input
        plot_period = {'date':'date', 'week':'week_start', 'month':'month_start', 'quarter':'quarter_start', 'year':'year'} 

        # Find cumulative count of messages sent by each participant for the user-defined time period
        frequency = plot_period[time_period]
        self.msgs_only['cumulative_count_msgs'] = self.msgs_only.groupby('sender_name').cumcount() + 1
        pivot_msg_counts = self.msgs_only.pivot_table(index = frequency, columns = 'sender_name', values = 'cumulative_count_msgs', aggfunc = 'last')
        pivot_msg_counts.fillna(method = 'ffill', inplace = True)

        # Find total messages sent over time for the user-defined time period
        total_msgs_over_time = pivot_msg_counts.sum(axis = 1)

        # Unstack pivot table 
        cum_msg_counts = pivot_msg_counts.stack().reset_index()
        cum_msg_counts.columns = [frequency, 'sender_name', 'cumulative_count_msgs']

        # Plot figure with subplots for total cumulative message count and pivot table data for each participant
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 6), gridspec_kw = {'height_ratios': [1, 2]})

        # Plot total message count
        total_msgs_over_time.plot(kind = 'line', linewidth = 1.5, ax = ax1)
        ax1.set_title('Total Message Count')
        ax1.set_xlabel(time_period.title())
        ax1.set_ylabel('Cumulative Count')
        ax1.grid(True)

        # Plot participant message count
        pivot_msg_counts.plot(kind = 'line', linewidth = 1.5, ax = ax2)
        ax2.set_title('Message Count by Participant')
        ax2.set_xlabel(time_period.title())
        ax2.set_ylabel('Cumulative Count')
        ax2.legend(title = 'Participant', loc = 'best', fontsize = 8).get_title().set_fontsize('8')
        ax2.grid(True)

        fig.suptitle('Cumulative Count of Messages Sent Over Time (by ' + time_period.title() + ')', fontsize = 16)
        plt.subplots_adjust(top = 0.5)
        plt.tight_layout()

        # Save plot (create new directory to save output if not available already)
        msg_count_dir = os.path.join(self.save_dir, 'message_counts')
        if not os.path.exists(msg_count_dir):
            os.makedirs(msg_count_dir)

        fig.savefig(os.path.join(msg_count_dir, 'cumulative_message_count.jpg'))

        # Create racecar plot to dynamically visualise changes in cumulative count of messages
        racecar = barplot(cum_msg_counts, item_column = 'sender_name', value_column = 'cumulative_count_msgs', time_column = frequency, top_entries = 10)
        output = racecar.plot(title = 'Cumulative Count of Messages Sent Over Time (by ' + time_period.title() + ')', 
                     item_label = 'Participant', value_label = 'Cumulative Count', frame_duration = 250)

        # Save plot 
        output.write_html(os.path.join(msg_count_dir, 'cumulative_racecar_count.html'))

        return

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
        reactions_given_participant[''] = ''
        reactions_given_participant['Group Aggregate'] = total_count_reactions.index

        # Find count of each distinct react received by each participant (top 10)
        reactions_received = reacts_exploded.groupby(['sender_name', 'reaction']).size().reset_index(name = 'count')
        reactions_received_sorted = reactions_received.sort_values(['sender_name', 'count'], ascending = [True, False])
        top_reactions_received = reactions_received_sorted.groupby('sender_name').head(10)
        top_reactions_received = (top_reactions_received.reset_index(drop = True)
                           .set_index(top_reactions_received.groupby('sender_name').cumcount() + 1))
        reactions_received_participant = top_reactions_received.pivot(columns = 'sender_name', values = 'reaction').fillna('')
        reactions_received_participant[''] = ''
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

        # Get word length aggregates for messages sent by each participant
        word_summary_participant = self.msgs_only.groupby('sender_name').agg(median_words = ('word_count', 'median'),
                                                                             average_words = ('word_count', 'mean'),
                                                                             max_words = ('word_count', 'max'))
        
        # Get word length aggregates for messages sent in chat
        word_summary_chat = self.msgs_only.agg(median_words = ('word_count', 'median'),
                                               average_words = ('word_count', 'mean'),
                                               max_words = ('word_count', 'max'))
        
        # Conduct sentiment and emotion analysis for each particpant

        # Conduct sentiment and emotion analysis for chat overall
        
        # Consolidate word analysis summary by appending above tables

        # Build word clouds for each participant (top 100 words)
        participants = self.msgs_only['sender_name'].unique()
        for person in participants:
            words = self.msgs_only[self.msgs_only['sender_name'] == person]['processed_text'].tolist()
            words_text = ' '.join(words)

            # Save plot

        # Build word cloud for chat overall (top 100 words)

        # Save plot

        # do topic modelling here as well

        pass
        
    @staticmethod
    def sentiment_emotion_analysis(text):
        #

        pass

    @staticmethod
    def topic_modelling(text):
        pass