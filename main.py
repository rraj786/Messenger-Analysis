'''
    The following script reads in Messenger chats, analyses messages to generate 
    statistical summaries, and conducts sentiment analysis to assess the emotional 
    tone of conversations.
    Author: Rohit Rajagopal
'''


import argparse
from metrics import *
import os
from process_json import *


# Set up user inputs
parser = argparse.ArgumentParser()
curr_dir = os.getcwd()
parser.add_argument('--dir', type = str, default = os.path.join(curr_dir, 'data'), help = 'Path to directory containing raw Messenger Data')
parser.add_argument('--save_dir', type = str, default = os.path.join(curr_dir, 'outputs'), help = 'Path to save Messenger chat metrics')
parser.add_argument('--cumulative_msgs_time_period', type = str, default = 'week', 
    help = 'Time period to aggregate cumulative messages over time (must be date, week, month, quarter, or year)')
parser.add_argument('--chat_activity_time_period', type = str, default = 'hour', 
    help = 'Time period to aggregate chat activity (must be hour, day, week, month, quarter, or year)')
args = parser.parse_args()

# Parse raw data into dataframe
chat_history = process_json(args.dir)

# Create new directory to save outputs if not available already
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Generate metrics and save to relevant directory
metrics = AnalyseChat(chat_history, args.save_dir)
metrics.summary_stats()
metrics.messages_over_time(args.cumulative_msgs_time_period)
metrics.chat_activity(args.chat_activity_time_period)
metrics.react_analysis()
metrics.word_analysis()
