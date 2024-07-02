'''
    The following script contains various class methods to analyse Messenger
    chats and save outputs to the designated directory.
    Author: Rohit Rajagopal
'''


import numpy as np
import os
import pandas as pd


class AnalyseChat:

    def __init__(self, chat_history):
        self.chat_history = chat_history
        self.msgs_only = self.chat_history[self.chat_history['content_type'] == "Message"]
        self.participants = self.msgs_only['sender_name'].unique()

    def summary_stats(self):

        # Find message and react received aggregations
        summary = self.msgs_only.groupby('sender_name').agg(messages_sent = ('content', 'count'), 
                                                            reacts_received = ('reacts_count', 'sum'),
                                                            messages_with_reacts = ('reacts_count', lambda x: (x > 0).sum()))
        
        # Find number of reacts given
        reacts_exploded = self.msgs_only.explode('reactions').dropna(subset = ['reactions'])
        summary['reacts_given'] = reacts_exploded['reactions'].apply(lambda x: x['actor']).value_counts().values

        # Find ratios for reacts per message and messages that got a react
        summary['reacts_per_message'] = summary['reacts_received'] / summary['messages_sent']
        summary['messages_per_react'] = summary['messages_with_reacts'] / summary['messages_sent']

        # Find content type aggregations
        content_filters = ['NA', 'Reactions Notification', 'Message']
        content_type_data = self.chat_history[~self.chat_history['content_type'].isin(content_filters)]
        content_pivot = content_type_data.pivot_table(index = 'sender_name', columns = 'content_type', values = 'content', aggfunc = 'count', fill_value = 0)

        # Find media type aggregations
        media_type_data = self.chat_history[self.chat_history['media_type'] != "Other"]
        media_pivot = media_type_data.pivot_table(index = 'sender_name', columns = 'media_type', values = 'content', aggfunc = 'count', fill_value = 0)

        # Consolidate final dataset by appending pivot tables above
        summary = pd.concat([summary, content_pivot, media_pivot], axis = 1, sort = True)

        return summary
