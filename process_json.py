'''
    The following script reads in Messenger chats provided in JSON format
    and parses them into a Pandas dataframe for further analysis.
    Author: Rohit Rajagopal
'''


import emoji
import json
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import pytz
import re
from tzlocal import get_localzone


# Initialise stopwords for text processing
stopwords = stopwords.words("english")


def process_json(directory):

    """
        Read in all JSON files from specified directory and generate a 
        complete dataset contaning all chat content and new rows using 
        feature engineering.
        
        Inputs:
            - directory (str)

        Returns:
            - chat_history (df)
    """

    messages = []
    files = os.listdir(directory)
    for file in files:
        if ".json" in file:
            with open(directory + '/' + file, 'r') as block:

                # Convert JSON to dictionary
                data = json.load(block)
                
                # Append messages on to each other
                messages += data['messages']
                
    # Normalise messages into dataframe
    chat_history = pd.json_normalize(messages)
    
    # Create column for datetime in local time zone
    chat_history['timestamp_sec'] = chat_history['timestamp_ms'] / 1000.0
    chat_history['datetime_utc'] = pd.to_datetime(chat_history['timestamp_sec'], unit = 's')
    local_tz = str(get_localzone())
    target = pytz.timezone(local_tz)
    chat_history['datetime_local'] = chat_history['datetime_utc'].dt.tz_localize(pytz.utc).dt.tz_convert(target)
    
    # Create columns for relative times
    chat_history['hour_of_day'] = chat_history['datetime_local'].dt.hour
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    chat_history['day_of_week'] = pd.Categorical(chat_history['datetime_local'].dt.day_name(), categories = day_order, ordered = True)
    chat_history['date'] = chat_history['datetime_local'].dt.date
    chat_history['week'] = chat_history['datetime_local'].dt.isocalendar().week
    chat_history['week_start'] = chat_history['datetime_local'].dt.to_period('W').apply(lambda x: x.start_time).dt.date
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    chat_history['month'] = pd.Categorical(chat_history['datetime_local'].dt.month_name(), categories = month_order, ordered = True)
    chat_history['month_start'] = chat_history['datetime_local'].dt.to_period('M').apply(lambda x: x.start_time).dt.date
    chat_history['quarter'] = chat_history['datetime_local'].dt.quarter
    chat_history['quarter_start'] = chat_history['datetime_local'].dt.to_period('Q').apply(lambda x: x.start_time).dt.date
    chat_history['year'] = chat_history['datetime_local'].dt.year
    
    # Convert raw UTF-8 format of messages to readable format
    chat_history['content'] = chat_history['content'].apply(lambda x: x.encode('latin1').decode('utf8') if isinstance(x, str) else x)

    # Create column to find number of words used in each message
    chat_history['word_count'] = chat_history['content'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
                                                                
    # Create column to indicate how many reacts each message got
    chat_history['reacts_count'] = chat_history['reactions'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Convert raw UTF-8 format of reactions to associated emojis
    chat_history['reactions'] = chat_history['reactions'].apply(lambda x: extract_reactions(x) if isinstance(x, list) else x)

    # Create column to indicate media type
    chat_history['media_type'] = chat_history.apply(categorise_media_type, axis = 1)

    # Create column to indicate content type
    chat_history['content_type'] = chat_history.apply(categorise_content_type, axis = 1)

    # Create column containing processed text to analyse (for simplicity consider for text messages only)
    chat_history['processed_text'] = chat_history.apply(lambda x: process_text(x['content']) if x['media_type'] == 'Message' else np.nan)

    # Create column to indicate number of emojis sent for messages only
    chat_history['emojis_count'] = chat_history['content'].apply(lambda x: emoji.emoji_count(x) if isinstance(x, str) else 0)
    
    # Arrange dataframe in ascending order of datetime
    chat_history = chat_history.sort_values(by = 'datetime_local')
    
    return chat_history

def process_text(text):

    # Convert text to lower case
    text = text.lower()

    # Remove non-alphanumeric characters (punctuation, emojis etc.)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove stop words 
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text
    
def extract_reactions(reactions):

    # Extract reactions from each message
    for react in reactions:
        emoji_char = react['reaction'].encode('latin1').decode('utf8')

        # Update the dictionary in place with the emoji
        react['reaction'] = emoji_char

    return reactions

def categorise_media_type(row):
       
    # Assign each instance to a media type (if there are multiple media types, then the hierarchy below selects the
    # "primary" type)
    if isinstance(row['is_unsent'], bool):
        return "Deleted Message"
    elif isinstance(row['photos'], list) or isinstance(row['videos'], list):
        return "Photo/Video"
    elif isinstance(row['files'], list):
        return "File Attachment"
    elif isinstance(row['share.link'], str) or isinstance(row['share.share_text'], str):
        return "Shared Link"
    elif isinstance(row['audio_files'], list):
        return "Audio"
    elif isinstance(row['gifs'], list):
        return "Gif"
    elif isinstance(row['sticker.ai_stickers'], list) or isinstance(row['sticker.uri'], str):
        return "Sticker"
    elif isinstance(row['content'], str):
        return "Text"
    else:
        return "Other"
    
def categorise_content_type(row):
    
    # Set up phrases to search for
    in_call = [" joined the call.", " joined the video call.", " started sharing video."]
    start_call = [" started a call.", " started a video call."]
    poll = [" created a poll: ", "\" in the poll.", "\" to the poll.", "This poll is no longer available."]
    left_group = [" left the group."]
    add_group = [" to the group."]
    group_chat = [" set your nickname to ", " cleared the nickname for ",
                           " cleared your nickname ", " set the nickname for ", " set her own nickname to ",
                           " set his own nickname to ", " set the quick reaction to ", " changed the theme to ",
                           " as the word effect for ", " pinned a message.", " named the group ", " changed the group photo.",
                           " turned off member approval. Anyone with the link can join the group."]
    location = [" sent a live location."]
    reactions_misc = [" to your message "]
    
    # Assign messages to content types
    if not isinstance(row['content'], str):
        if row['media_type'] == "Other":
            return "NA"
        elif row['media_type'] == "Deleted Message":
            return row['media_type']
        else:
            return "Message"
    elif any(phrase in row['content'] for phrase in in_call):
        return "In Call Settings"
    elif any(phrase in row['content'] for phrase in start_call):
        return "Started Call"
    elif any(phrase in row['content'] for phrase in poll):
        return "Poll Settings"
    elif any(phrase in row['content'] for phrase in left_group):
        return "Left Chat"
    elif any(phrase in row['content'] for phrase in add_group):
        return "Added Member to Chat"
    elif any(phrase in row['content'] for phrase in group_chat):
        return "Chat Settings"
    elif any(phrase in row['content'] for phrase in location):
        return "Shared Location"
    elif any(phrase in row['content'] for phrase in reactions_misc):
        return "Reactions Notification"
    else:
        return "Message"
