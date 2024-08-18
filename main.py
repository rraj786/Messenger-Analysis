'''
    The following script reads in a Messenger chat, analyses its content to generate 
    statistical summaries, and conducts sentiment and emotion analysis to assess the
    tone of conversations.
    Author: Rohit Rajagopal
'''


import argparse
from generate_report import *
from process_json import *


# Set up user inputs
parser = argparse.ArgumentParser()
curr_dir = os.getcwd()
parser.add_argument('--dir', type = str, default = os.path.join(curr_dir, 'data'), help = 'Path to directory containing raw Messenger Data')
parser.add_argument('--batch_size', type = str, default = 48, help = 'Batch size to run Sentiment and Emotion Analysis models')
parser.add_argument('--save_dir', type = str, default = os.path.join(curr_dir, 'outputs'), help = 'Path to save Messenger chat metrics')
args = parser.parse_args()

# Parse raw data into dataframe
group_name, chat_history = process_json(args.dir)

# Create new directory to save outputs if not available already
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Generate Streamlit HTML report
generate_report(chat_history, group_name, args)
