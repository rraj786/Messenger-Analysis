'''
    The following script contains various class methods to analyse Messenger
    chats and save outputs to the designated directory.
    Author: Rohit Rajagopal
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


class AnalyseChat:

    def __init__(self, chat_history):
        self.chat_history = chat_history
        msgs_only_filter = ['Message', 'Shared Location']
        self.msgs_only = self.chat_history[self.chat_history['content_type'].isin(msgs_only_filter)]
        self.participants = self.msgs_only['sender_name'].unique()

    def summary_stats(self):

        # Find messages sent and reacts received aggregations
        summary = self.msgs_only.groupby('sender_name').agg(messages_sent = ('content', 'count'), 
                                                            reacts_received = ('reacts_count', 'sum'),
                                                            messages_with_reacts = ('reacts_count', lambda x: (x > 0).sum()))
        
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
        content_filters = ['Started Call', 'Left GC', 'Added Member to GC', 'Shared Location']
        content_type_data = self.chat_history[self.chat_history['content_type'].isin(content_filters)]
        content_pivot = content_type_data.pivot_table(index = 'sender_name', columns = 'content_type', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Consolidate final dataset by appending pivot tables above
        summary = pd.concat([summary, media_pivot, content_pivot], axis = 1, sort = True)

        return summary
    
    def messages_over_time(self, time_period):
        
        # Create dictionary to map on to correct period based on user-defined time period
        plot_period = {'day':'date', 'week':'week_start', 'month':'month_start', 'quarter':'quarter_start', 'year':'year'} 

        # Group messages by each participant over time for the user-defined time period
        frequency = plot_period[time_period]
        msgs_over_time = self.msgs_only.pivot_table(index = frequency, columns = 'sender_name', values = 'timestamp_ms', aggfunc = 'count', fill_value = 0)

        # Find total messages sent over time for the user-defined time period
        total_msgs_over_time = msgs_over_time.sum(axis = 1)

        # Plot pivot table data with a line for each participant
        fig, ax1 = plt.subplots(figsize = (12, 6))
        msgs_over_time.plot(kind = 'line', ax = ax1)
        ax1.set_xlabel('time_period.title()')
        ax1.set_ylabel('Count by Participant')
        ax1.set_title('Count of Messages Sent over Time (by ' + time_period.title() + ')')
        max_ax1 = msgs_over_time.max().max()
        ax1.set_ylim(0, max_ax1 * 1.3)
        ax2 = ax1.twinx()
        total_msgs_over_time.plot(ax = ax2, color = 'black', linestyle = '-')
        ax2.set_ylabel('Total Count for Group')
        ax1.legend(title = 'Participant', loc = 'best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Create racecar plot to dynamically visualise changes in cumulative count of messages






        pass

    def user_activity(self):
        activity_data = self.chat_history.groupby(['sender_name', 'day_of_week']).size().reset_index(name='activity_count')

        # Pivot table to reshape the data for heatmap
        heatmap_data = activity_data.pivot('sender_name', 'day_of_week', 'activity_count')

        # Create a heatmap using Seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', cbar=True)
        plt.title('User Activity Heatmap')
        plt.xlabel('Day of Week')
        plt.show()

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Convert day_of_week to categorical with custom order
        self.chat_history['day_of_week'] = pd.Categorical(self.chat_history['day_of_week'], categories=day_order, ordered=True)
        activity_data = self.chat_history.groupby(['hour_of_day', 'day_of_week']).size().reset_index(name='activity_count')

        # Pivot table to reshape the data for heatmap
        heatmap_data = activity_data.pivot('hour_of_day', 'day_of_week', 'activity_count')

        # Create a heatmap using Seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', cbar=True)
        plt.title('User Activity Heatmap')
        plt.xlabel('Day of Week')
        plt.ylabel('Hour of Day')
        plt.show()

        pass

