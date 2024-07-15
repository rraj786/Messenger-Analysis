'''
    The following script reads in Messenger chats, analyses messages to generate 
    statistical summaries, and conducts sentiment analysis to assess the emotional 
    tone of conversations.
    Author: Rohit Rajagopal
'''


import argparse
from metrics import *
import os
from process_html import *
from process_json import *


# Set up user inputs
parser = argparse.ArgumentParser()
curr_dir = os.getcwd()
parser.add_argument('--dir', type = str, default = os.path.join(curr_dir, 'data'), help = 'Path to directory containing raw Messenger Data')
parser.add_argument('--file_type', type = str, default = 'json', help = 'File type of Messenger files (must be json or html)')
parser.add_argument('--save_dir', type = str, default = os.path.join(curr_dir, 'outputs'), help = 'Path to save Messenger chat metrics')
args = parser.parse_args()

# Parse raw data into dataframe
if args.file_type == 'json':
    chat_history = process_json(args.dir)
elif args.file_type == 'html':
    pass
    chat_history = process_html(args.dir)

# Create new directory to save outputs if not available already
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Generate metrics and save to relevant directory
metrics = AnalyseChat(chat_history, args.save_dir)
metrics.summary_stats()
metrics.messages_over_time('month')
metrics.chat_activity('hour')
metrics.react_analysis()
metrics.emoji_analysis()
metrics.conversations()
metrics.word_analysis()
metrics.sentiment_emotion_analysis()
