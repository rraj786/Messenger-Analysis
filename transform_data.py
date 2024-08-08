'''
    The following script reads in Messenger chats provided in JSON format
    and parses them into a Pandas dataframe for further analysis.
    Author: Rohit Rajagopal
'''


import emoji
import pandas as pd
import pytz
import regex as re
from tzlocal import get_localzone


def transform_data(df):

    # Create column for datetime in local time zone
    df['timestamp_sec'] = df['timestamp_ms'] / 1000.0
    df['datetime_utc'] = pd.to_datetime(df['timestamp_sec'], unit = 's')
    local_tz = str(get_localzone())
    target = pytz.timezone(local_tz)
    df['datetime_local'] = df['datetime_utc'].dt.tz_localize(pytz.utc).dt.tz_convert(target)
    
    # Create columns for relative time periods
    hour_order = ['12:00AM', '01:00AM', '02:00AM', '03:00AM', '04:00AM', '05:00AM', '06:00AM', '07:00AM', '08:00AM', '09:00AM', '10:00AM', 
                  '11:00AM', '12:00PM', '01:00PM', '02:00PM', '03:00PM', '04:00PM', '05:00PM', '06:00PM', '07:00PM', '08:00PM', '09:00PM', 
                  '10:00PM', '11:00PM']
    df['hour_of_day'] = pd.Categorical(df['datetime_local'].dt.strftime('%I:00%p'), categories = hour_order, ordered = True)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['datetime_local'].dt.day_name(), categories = day_order, ordered = True)
    df['date'] = df['datetime_local'].dt.date
    df['week'] = df['datetime_local'].dt.isocalendar().week
    df['week_start'] = df['datetime_local'].dt.to_period('W').dt.start_time
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df['month'] = pd.Categorical(df['datetime_local'].dt.month_name(), categories = month_order, ordered = True)
    df['month_start'] = df['datetime_local'].dt.to_period('M').dt.start_time
    df['quarter'] = df['datetime_local'].dt.quarter
    df['quarter_start'] = df['datetime_local'].dt.to_period('Q').dt.start_time
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
    
    # Set up phrases to search for and compile regex patterns
    patterns = {
        'In Call Settings': re.compile(r' joined the call\.| joined the video call\.| started sharing video\.'),
        'Started Call': re.compile(r' started a call\.| started a video call\.'),
        'Poll Settings': re.compile(r' created a poll: | voted for [\w\W]*? in the poll\.| added [\w\W]*? to the poll\.|This poll is no longer available\.'),
        'Member Left Group': re.compile(r' left the group\.'),
        'Added Member to Group': re.compile(r' added [\w\W]*? to the group\.'),
        'Chat Settings': re.compile(r' set your nickname to | cleared the nickname for | cleared your nickname | set the nickname for | set her own nickname to | set his own nickname to | set the quick reaction to | changed the theme to | as the word effect for | pinned a message\.| named the group | changed the group photo\.| turned off member approval\. Anyone with the link can join the group\.'),
        'Shared Location': re.compile(r' sent a live location\.'),
        'React Notification': re.compile(r'reacted\s[\p{So}]\s*to your message', re.UNICODE)
    }

    # Check media type first
    if not isinstance(row['content'], str):
        if row['media_type'] == 'Other':
            return 'NA'
        elif row['media_type'] == 'Deleted Message':
            return 'Deleted Message'
        else:
            return 'Message'
    
    # Check for other patterns
    for content_type, pattern in patterns.items():
        if pattern.search(row['content']):
            return content_type

    # Default case
    return 'Message'
