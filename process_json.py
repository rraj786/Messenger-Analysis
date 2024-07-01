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

def parse_json(directory):

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

    # Create column for datetime in local time zone
    chat_history['timestamp_sec'] = chat_history['timestamp_ms'] / 1000.0
    chat_history['datetime_utc'] = pd.to_datetime(chat_history['timestamp_sec'], unit = 's')
    local_tz = str(get_localzone())
    target = pytz.timezone(local_tz)
    chat_history['datetime_local'] = chat_history['datetime_utc'].dt.tz_localize(pytz.utc).dt.tz_convert(target)
    
    # Create columns for relative times
    chat_history['hour_of_day'] = chat_history['datetime_local'].dt.hour
    chat_history['day_of_week'] = chat_history['datetime_local'].dt.day_name()
    chat_history['date'] = chat_history['datetime_local'].dt.date
    chat_history['week_start'] = chat_history['datetime_local'].dt.to_period('W').apply(lambda x: x.start_time).dt.date
    chat_history['month'] = chat_history['datetime_local'].dt.month_name()
    chat_history['year'] = chat_history['datetime_local'].dt.year
    
    # Create column to indicate content type (message, image, audio file, call, poll etc.)
    chat_history['content_type'] = chat_history['content'].apply(categorise_content_type)
    
    # Create new column to indicate media type
    chat_history['media_type'] = chat_history.apply(categorise_media_type, axis = 1)
    
    # Arrange dataframe in ascending order of datetime
    chat_history = chat_history.sort_values(by = 'datetime_local')
    
    return chat_history
    
def categorise_content_type(content):
    
    # Set up phrases to search for
    in_call = [" joined the call.", " joined the video call.", " started sharing video."]
    start_call = [" started a call.", " started a video call."]
    in_poll = ["\" in the poll.", "\" to the poll.", "This poll is no longer available."]
    start_poll = [" created a poll: "]
    members = [" left the group.", " to the group.", " turned off member approval. Anyone with the link can join the group."]
    group_chat = [" set your nickname to ", " cleared the nickname for ",
                           " cleared your nickname ", " set the nickname for ", " set her own nickname to ",
                           " set his own nickname to ", " set the quick reaction to ", " changed the theme to ",
                           " as the word effect for ", " pinned a message.", " named the group ", " changed the group photo."]
    location = [" sent a live location."]
    reactions_misc = [" reacted \\u"]
    
    # Assign messages to content types
    if not isinstance(content, str):
        return "NA"
    elif any(phrase in content for phrase in in_call):
        return "In Call"
    elif any(phrase in content for phrase in start_call):
        return "Start Call"
    elif any(phrase in content for phrase in in_poll):
        return "In Poll"
    elif any(phrase in content for phrase in start_poll):
        return "Start Poll"
    elif any(phrase in content for phrase in members):
        return "Members"
    elif any(phrase in content for phrase in group_chat):
        return "GC Settings"
    elif any(phrase in content for phrase in location):
        return "Shared Location"
    elif any(phrase in content for phrase in reactions_misc):
        return "Reactions Notification"
    else:
        return "Message"

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
    else:
        return "Other"
    