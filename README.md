# Messenger Group Chat Analysis

This repository allows users to analyse Messenger group chat data in JSON format. It extracts high-level metrics by participant, performs time-series and word analysis, and presents the results through a user-friendly Streamlit application.

## Installation

This script requires Python 3 and the following dependencies:

- argparse (parse user inputs)
- collections (count most frequent entries)
- dataset (convert list to dataset object)
- datetime (parse datetime information)
- emoji (analyze emojis and reactions)
- json (parse JSON data to dictionary)
- nltk (NLP tool)
- numpy (manipulate arrays and apply mathematical operations)
- os (create new file directories and save outputs)
- pandas (store and manipulate information in dataframes)
- Pillow (image processing tool, the fork of PIL)
- plotly (plotting results)
- pytz (determine time in local time zone)
- raceplotly (create racecar plot)
- re (word processing)
- streamlit (build interactive user-friendly reports)
- torch (enable model processing using GPU)
- tqdm (monitor prediction progress)
- transformers (access pre-trained NLP models)
- tzlocal (get local time zone information)
- wordcloud (generate word clouds)

```bash
pip install argparse
pip install collections
pip install dataset
pip install datetime
pip install emoji 
pip install json 
pip install nltk 
pip install numpy
pip install os 
pip install pandas 
pip install Pillow 
pip install plotly 
pip install pytz
pip install raceplotly
pip install re
pip install streamlit 
pip install torch
pip install tqdm
pip install transformers
pip install tzlocal
pip install wordcloud 
```

Setting Up PyTorch with CUDA - to run Transformers models locally with GPU support:

1. Check GPU Compatibility
    - Identify your GPU model and compute capability from the [NVIDIA CUDA GPUs page](https://developer.nvidia.com/cuda-gpus).

2. Install CUDA Toolkit
    - Download and install the appropriate version from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

3. Install PyTorch
    - Use the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to get the installation command. 

4. Check if CUDA is available:
    ```bash
    import torch
    print(torch.cuda.is_available())  # Should return True
    ```

Before running the code, please ensure that the __get_colors() method in your local version of plots.py (part of the raceplotly package) has been modified as shown below:
```bash
 def __get_colors(self):
        if (self.item_color == None):
            colors = {item: 'rgb({}, {}, {})'.format(*sample(range(256), 3)) for item in self.df[self.item_column].unique()}
            self.df['color'] = self.df[self.item_column].map(colors)
        elif (self.item_color != None and len(self.item_color) != self.df[self.item_column].nunique()):
            for item in self.df[self.item_column].unique():
                if item not in self.item_color.keys():
                    self.item_color[item] = 'rgb({}, {}, {})'.format(*sample(range(256), 3))
            self.df['color'] = self.df[self.item_column].map(self.item_color)
        else:
            self.df['color'] = self.df[self.item_column].map(self.item_color)
```

I've raised an issue under [this link](https://github.com/lucharo/raceplotly/issues/22) to update the code at source - am yet to hear back at the time of writing.
    
## Usage

To view the project, follow these steps:
1. Clone the repository or download a zip file containing the scripts.
2. Ensure you have installed the required dependencies.
3. Run the following line within your IDE of choice or in a command-line interface like Anaconda Powershell Prompt:
```bash
streamlit run main.py
```
4. If you would like to change the path to your Messenger data or model batch size, adjust the arguments as shown below:
```bash
streamlit run main.py --dir .\data --batch_size 48
```

## Extracting Messenger Data from Facebook

To download your Messenger group chat data from Facebook in JSON format, follow these steps:

1. Request Your Facebook Data
    - Log in to Facebook:
    - Open your browser and go to Facebook. Log in with your credentials.

2. Access Your Facebook Settings:
    - Click on the downward-facing arrow in the top-right corner of the Facebook page.
    - Select "Settings & privacy" and then "Settings."

3. Go to Your Facebook Information:
    - In the left sidebar, click on "Your Facebook Information."

4. Download Your Information:
    - Click on "Download Your Information."
    - You'll be directed to a page where you can choose the data you want to download.

5. Select Data to Include:
    - By default, all data categories are selected. To focus on Messenger data, deselect everything except "Messages."
    - You can choose the date range and media quality, but please ensure that you have selected JSON format.

6. Request a Download:
    - Click on "Request a Download."
    - Facebook will prepare your data. You’ll receive a notification when it’s ready.

7. Download Your Data:
    - Go back to "Download Your Information" and click on "Available Copies."
    - Download the file once it’s ready. The file will typically be in a ZIP format.

8. Extract and Locate Messenger Data
    - Unzip the downloaded file on your computer.

9. Find the Messenger Data:
    - Open the extracted folder and locate the "messages" folder. This folder contains your Messenger chat data in JSON format.

Please note that the code in this repository can only handle one group chat at a time.

## Metrics

The Streamlit report contains 5 sections to capture various trends

1. Summary: An overview of Messenger group chat metrics, featuring plots to highlight significant trends and key insights.

2. Messages Sent over Time: Message trends by participant and chat overall, with insights into seasonal patterns.

3. Chat Activity: Examination of chat activity patterns to identify peak engagement times and determine the most and least active days historically.

4. Reactions Analysis: Identify most used reacts,group and individual interactions, and top messages by number of reacts received.

5. Word Analysis: Analyse messages lengths by participant and general tone through sentiment and emotion analysis.

## References

- Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
Ashritha R Murthy and K M Anil Kumar 2021 IOP Conf. Ser.: Mater. Sci. Eng. 1110 012009

- Cardiff NLP, "CardiffNLP Twitter RoBERTa Base Sentiment". https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/, 2024.
