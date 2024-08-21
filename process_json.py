'''
    The following script reads in Messenger chats provided in JSON format
    and parses them into a Pandas dataframe for further analysis.
    
    Author: Rohit Rajagopal
'''


import json
import os
from transform_data import *


def process_json(directory):

    """
        Reads JSON files from a directory and combines them into a single Pandas DataFrame.

        Args:
            - directory (str): Path to the directory containing JSON files.

        Returns:
            - group_name (str): Name of the chat group.
            - chat_history (pd.DataFrame): DataFrame with parsed chat data.
    """

    messages = []
    files = os.listdir(directory)
    for file in files:
        if '.json' in file:
            with open(directory + '/' + file, 'r') as block:

                # Convert JSON to dictionary
                data = json.load(block)
                
                # Append messages on to each other
                messages += data['messages']

    # Extract group name
    group_name = data['title']

    # Normalise messages into dataframe
    normalised_messages = pd.json_normalize(messages)
    
    # Transform dataframe and apply feature engineering to generate useful columns
    chat_history = transform_data(normalised_messages)

    return group_name, chat_history
