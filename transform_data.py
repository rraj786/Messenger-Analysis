'''
    The following script reads in Messenger chats provided in JSON format
    and parses them into a Pandas dataframe for further analysis.
    Author: Rohit Rajagopal
'''


import emoji
import pandas as pd
import pytz
from tzlocal import get_localzone


def transform_data(df):

    # Create column for datetime in local time zone
    df['timestamp_sec'] = df['timestamp_ms'] / 1000.0
    df['datetime_utc'] = pd.to_datetime(df['timestamp_sec'], unit = 's')
    local_tz = str(get_localzone())
    target = pytz.timezone(local_tz)
    df['datetime_local'] = df['datetime_utc'].dt.tz_localize(pytz.utc).dt.tz_convert(target)
    
    # Create columns for relative time periods
    df['hour_of_day'] = df['datetime_local'].dt.hour
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['datetime_local'].dt.day_name(), categories = day_order, ordered = True)
    df['date'] = df['datetime_local'].dt.date
    df['week'] = df['datetime_local'].dt.isocalendar().week
    df['week_start'] = df['datetime_local'].dt.to_period('W').apply(lambda x: x.start_time).dt.date
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df['month'] = pd.Categorical(df['datetime_local'].dt.month_name(), categories = month_order, ordered = True)
    df['month_start'] = df['datetime_local'].dt.to_period('M').apply(lambda x: x.start_time).dt.date
    df['quarter'] = df['datetime_local'].dt.quarter
    df['quarter_start'] = df['datetime_local'].dt.to_period('Q').apply(lambda x: x.start_time).dt.date
    df['year'] = df['datetime_local'].dt.year
    
    # Convert raw UTF-8 format of content string to readable format
    df['content'] = df['content'].apply(lambda x: x.encode('latin1').decode('utf8') if isinstance(x, str) else x)

    # Create column to find number of words used in each record
    df['word_count'] = df['content'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
                                                                
    # Create column to count number of reacts received for each record
    df['reacts_count'] = df['reactions'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Convert raw UTF-8 format of reactions to extract associated emojis
    df['reactions'] = df['reactions'].apply(lambda x: extract_reactions(x) if isinstance(x, list) else x)

    # Create column to indicate media type of each record
    df['media_type'] = df.apply(categorise_media_type, axis = 1)

    # Create column to indicate content type of each record
    df['content_type'] = df.apply(categorise_content_type, axis = 1)

    # Create column to indicate number of emojis used in each record
    df['emojis_count'] = df['content'].apply(lambda x: emoji.emoji_count(x) if isinstance(x, str) else 0)
    
    # Arrange dataframe in ascending order of datetime
    df = df.sort_values(by = 'datetime_local')
    
    return df
  
def extract_reactions(reactions):

    # Extract reactions from each message
    for react in reactions:
        emoji_char = react['reaction'].encode('latin1').decode('utf8')

        # Update the dictionary in place with the emoji
        react['reaction'] = emoji_char

    return reactions

def categorise_media_type(row):
       
    # Assign each instance to a media type (if there are multiple media types, then the hierarchy below selects the
    # 'primary' type)
    if isinstance(row['is_unsent'], bool):
        return 'Deleted Message'
    elif isinstance(row['photos'], list) or isinstance(row['videos'], list):
        return 'Photo/Video'
    elif isinstance(row['files'], list):
        return 'File Attachment'
    elif isinstance(row['share.link'], str) or isinstance(row['share.share_text'], str):
        return 'Shared Link'
    elif isinstance(row['audio_files'], list):
        return 'Audio'
    elif isinstance(row['gifs'], list):
        return 'Gif'
    elif isinstance(row['sticker.ai_stickers'], list) or isinstance(row['sticker.uri'], str):
        return 'Sticker'
    elif isinstance(row['content'], str):
        return 'Text'
    else:
        return 'Other'
    
def categorise_content_type(row):
    
    # Set up phrases to search for
    in_call = [' joined the call.', ' joined the video call.', ' started sharing video.']
    start_call = [' started a call.', ' started a video call.']
    poll = [' created a poll: ', '\" in the poll.', '\" to the poll.', 'This poll is no longer available.']
    left_group = [' left the group.']
    add_group = [' to the group.']
    chat = [' set your nickname to ', ' cleared the nickname for ',
                           ' cleared your nickname ', ' set the nickname for ', ' set her own nickname to ',
                           ' set his own nickname to ', ' set the quick reaction to ', ' changed the theme to ',
                           ' as the word effect for ', ' pinned a message.', ' named the group ', ' changed the group photo.',
                           ' turned off member approval. Anyone with the link can join the group.']
    location = [' sent a live location.']
    reactions_misc = [' to your message ']
    
    # Assign messages to content types
    if not isinstance(row['content'], str):
        if row['media_type'] == 'Other':
            return 'NA'
        elif row['media_type'] == 'Deleted Message':
            return row['media_type']
        else:
            return 'Message'
    elif any(phrase in row['content'] for phrase in in_call):
        return 'In Call Settings'
    elif any(phrase in row['content'] for phrase in start_call):
        return 'Started Call'
    elif any(phrase in row['content'] for phrase in poll):
        return 'Poll Settings'
    elif any(phrase in row['content'] for phrase in left_group):
        return 'Member Left Group'
    elif any(phrase in row['content'] for phrase in add_group):
        return 'Added Member to Group'
    elif any(phrase in row['content'] for phrase in chat):
        return 'Chat Settings'
    elif any(phrase in row['content'] for phrase in location):
        return 'Shared Location'
    elif any(phrase in row['content'] for phrase in reactions_misc):
        return 'React Notification'
    else:
        return 'Message'
