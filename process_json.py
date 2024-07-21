'''
    The following script reads in Messenger chats provided in JSON format
    and parses them into a Pandas dataframe for further analysis.
    Author: Rohit Rajagopal
'''


import json
import os
import pandas as pd
from transform_data import *


def process_json(directory):

    """
        Read in all JSON files from specified directory and construct a 
        Pandas dataframe containing all content.
        
        Inputs:
            - directory (str)

        Returns:
            - chat_history (df)
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
                
    # Normalise messages into dataframe
    normalised_messages = pd.json_normalize(messages)
    
    # Transform dataframe and apply feature engineering to generate useful columns
    chat_history = transform_data(normalised_messages)

    return chat_history
