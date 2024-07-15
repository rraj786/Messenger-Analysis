'''
    The following script reads in Messenger chats provided in JSON format
    and parses them into a Pandas dataframe for further analysis.
    Author: Rohit Rajagopal
'''


import json
import os
import pandas as pd
import pytz
from tzlocal import get_localzone

def process_json(directory):

    """
        Read in all JSON files from specified directory and generate a 
        unified dataset.
        
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

    # Create column to indicate how many reacts each message got
    chat_history['reacts_count'] = chat_history['reactions'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Convert raw UTF-8 format of reactions to associated emojis
    chat_history['reactions'] = chat_history['reactions'].apply(extract_reactions)

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
    
    # Create column to indicate media type
    chat_history['media_type'] = chat_history.apply(categorise_media_type, axis = 1)

    # Create column to indicate content type
    chat_history['content_type'] = chat_history.apply(categorise_content_type, axis = 1)
    
    # Arrange dataframe in ascending order of datetime
    chat_history = chat_history.sort_values(by = 'datetime_local')
    
    return chat_history

def extract_reactions(reactions):

    # Check if reaction key exists for each message
    if not isinstance(reactions, float):
        for react in reactions:
            emoji_char = react['reaction'].encode('latin1').decode('utf8')
            # Update the dictionary in place with the emoji
            react['reaction'] = emoji_char

    return reactions

def categorise_media_type(row):
       
    # Assign each instance to a media type (Gifs, Sticker, Text, Image, Video, Link, Audio)
    if not isinstance(row['audio_files'], float):
        return "Audio"
    elif not isinstance(row['is_unsent'], float):
        return "Deleted Message"
    elif not isinstance(row['files'], float):
        return "File Attachment"
    elif not isinstance(row['gifs'], float):
        return "Gif"
    elif not isinstance(row['photos'], float):
        return "Image"
    elif not isinstance(row['share.link'], float) or not isinstance(row['share.share_text'], float):
        return "Shared Link"
    elif not isinstance(row['sticker.ai_stickers'], float) or not isinstance(row['sticker.uri'], float):
        return "Sticker"
    elif not isinstance(row['videos'], float):
        return "Video"
    elif not isinstance(row['content'], str):
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
        return "GC Settings"
    elif any(phrase in row['content'] for phrase in location):
        return "Shared Location"
    elif any(phrase in row['content'] for phrase in reactions_misc):
        return "Reactions Notification"
    else:
        return "Message"
