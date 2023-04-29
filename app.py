from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import TextTilingTokenizer 
from semantic_search import SemanticSearch 
import pandas as pd
import gradio as gr
import numpy as np
import requests
import tiktoken
import openai
import json
import re

tt = TextTilingTokenizer()
searcher = SemanticSearch()

# Initialize a counter for duplicate titles
title_counter = {}

# One to one mapping from titles to urls
titles_to_urls = {}

def get_youtube_data(url):

    video_id = url.split("=")[1]

    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)
    except:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            raw = transcript.translate('en').fetch()
            break

    response = requests.get(f"https://noembed.com/embed?dataType=json&url={url}")
    data = json.loads(response.content)

    title, author = data["title"], data["author_name"]

    df = pd.DataFrame(raw)

    df['end'] = df['start'] + df['duration']
    df['total_words'] = df['text'].apply(lambda x: len(x.split())).cumsum()
    df["text"] = df["text"] + "\n\n"

    return df, title, author

def to_timestamp(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds_remaining = seconds % 60
    
    if seconds >= 3600:
        return f"{hours:02d}:{minutes:02d}:{seconds_remaining:02d}"
    else:
        return f"{minutes:02d}:{seconds_remaining:02d}"

def to_seconds(timestamp):
    time_list = timestamp.split(':')
    total_seconds = 0
    if len(time_list) == 2:  # Minutes:Seconds format
        total_seconds = int(time_list[0]) * 60 + int(time_list[1])
    elif len(time_list) == 3:  # Hours:Minutes:Seconds format
        total_seconds = int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2])
    else:
        raise ValueError("Invalid timestamp format")
    return total_seconds

def get_segments(df, title, author, split_by_topic, segment_length = 200):

    transcript = df['text'].str.cat(sep=' ')

    if not split_by_topic:
        words = transcript.split()
        segments = [' '.join(words[i:i+segment_length]) for i in range(0, len(words), segment_length)]
    else:
        segments = tt.tokenize(transcript)

    segments = [segment.replace('\n','').strip() for segment in segments]

    segments_wc = [len(segment.split()) for segment in segments]
    segments_wc = np.cumsum(segments_wc)

    idx = [np.argmin(np.abs(df['total_words'] - total_words)) for total_words in segments_wc]

    segments_end_times = df['end'].iloc[idx].values
    segments_end_times = np.insert(segments_end_times, 0, 0.0)

    segments_times = [f"({to_timestamp(segments_end_times[i-1])}, {to_timestamp(segments_end_times[i])})" for i in range(1,len(segments_end_times))]

    segments_text = [f"Segment from '{title}' by {author}\nTimestamp: {segment_time}\n\n{segment}\n" for segment, segment_time in zip(segments, segments_times)]

    return segments_text

def fit_searcher(segments, n_neighbours):
    global searcher
    searcher.fit(segments, n_neighbors=n_neighbours)
    return True

def num_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def refencify(text):
    title_pattern = r"Segment from '(.+)'"
    timestamp_pattern = r"Timestamp: \((.+)\)"

    title = re.search(title_pattern, text).group(1)
    start_timestamp = re.search(timestamp_pattern, text).group(1).split(",")[0]

    url = titles_to_urls[title]
    start_seconds = to_seconds(start_timestamp)

    return f"Segment URL: {url}&t={start_seconds}\n" + text

def form_query(question, model, token_budget):

    results = searcher(question)

    introduction = 'Use the below segments from multiple youtube videos to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer." Cite each references using the [title, author, timestamp] notation. Every sentence should have a citation at the end.'

    message = introduction

    question = f"\n\nQuestion: {question}"

    references = ""

    for i, result in enumerate(results):
        result = result + "\n\n"
        if (
            num_tokens(message + result + question, model=model)
            > token_budget
        ):
            break
        else:
            message += result
            references += f"### Segment {i+1}:\n" + refencify(result)

    # Remove the last extra two newlines
    message = message[:-2]

    references = "Segments that might have been used to answer your question: (If you specified more segments than shown here, consider increasing your token budget)\n\n" + references

    return message + question, references

def generate_answer(question, model, token_budget, temperature):
    
    message, references = form_query(question, model, token_budget)

    messages = [
        {"role": "system", "content": "You answer questions about YouTube videos."},
        {"role": "user", "content": message},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    response_message = response["choices"][0]["message"]["content"]
    return response_message, references

def add_to_dict(title, url):
    global title_counter

    if title not in titles_to_urls:
        # This is the first occurrence of this title
        titles_to_urls[title] = url
        return title
    else:
        # This title has already been seen, so we need to add a number suffix to it
        # First, check if we've already seen this title before
        if title in title_counter:
            # If we have, increment the counter
            title_counter[title] += 1
        else:
            # If we haven't, start the counter at 1
            title_counter[title] = 1
        
        # Add the suffix to the title
        new_title = f"{title} ({title_counter[title]})"
        
        # Add the new title to the dictionary
        titles_to_urls[new_title] = url
        return new_title

def main(urls_text, question, split_by_topic, segment_length, n_neighbours, model, token_budget, temperature):

    global title_counter
    title_counter = {}

    urls = list(set(urls_text.split("\n")))
    segments = []

    for url in urls:
        df, title, author = get_youtube_data(url)
        
        title = add_to_dict(title, url)

        video_segments = get_segments(df, title, author, split_by_topic, segment_length)

        segments.extend(video_segments)

    print("Segments generated successfully!")

    if fit_searcher(segments, n_neighbours):
        print("Searcher fit successfully!")
        answer, references = generate_answer(question, model, token_budget, temperature)

    return answer, references

title = "Ask YouTube GPT 📺"

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(f'Ask YouTube GPT allows you to ask questions about a set of Youtube Videos using Universal Sentence Encoder and Open AI. The returned response cites the video title, author and timestamp in square brackets where the information is located, adding credibility to the responses and helping you locate incorrect information. If you need one, get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a>.</p>')

    with gr.Row():
        
        with gr.Group():
            
            openAI_key=gr.Textbox(label='Enter your OpenAI API key here')

            # Allow the user to input multiple links, adding a textbox for each
            urls_text = gr.Textbox(lines=5, label="Enter the links to the YouTube videos you want to search (one per line):", placeholder="https://www.youtube.com/watch?v=4xWJf8cERoM\nhttps://www.youtube.com/watch?v=vx-Si9gbijA")

            question = gr.Textbox(label='Enter your question here')

            with gr.Accordion("Advanced Settings", open=False):
                split_by_topic = gr.Checkbox(label="Split segments by topic", value=True, info="Whether the video transcripts are to be segmented by topic or by word count. Splitting by topic may result in a more coherent response, but results in a slower response time, especially for lengthy videos.")
                segment_length = gr.Slider(label="Segment word count", minimum=50, maximum=500, step=50, value=200, visible=False)

                def fn(split_by_topic):
                    return gr.Slider.update(visible=not split_by_topic)

                # If the user wants to split by topic, allow them to set the maximum segment length. (Make segment_length visible)
                split_by_topic.change(fn, split_by_topic, segment_length)
                
                n_neighbours = gr.Slider(label="Number of segments to retrieve", minimum=1, maximum=20, step=1, value=5, info="The number of segments to retrieve from each video and feed to the GPT model for answering.")
                model = gr.Dropdown(label="Model", value="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"])
                token_budget = gr.Slider(label="Prompt token budget", minimum=100, maximum=4000, step=100, value=1000, info="The maximum number of tokens the prompt can take.")
                temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=0, info="The GPT model's temperature. Recommended to use a low temperature to decrease the likelihood of hallucinations.")

            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            
            with gr.Tabs():
                with gr.TabItem("Answer"):
                    answer = gr.Markdown()
                with gr.TabItem("References"):
                    references = gr.Markdown()

        btn.click(main, inputs=[urls_text, question, split_by_topic, segment_length, n_neighbours, model, token_budget, temperature], outputs=[answer, references])    
            
#openai.api_key = os.getenv('Your_Key_Here') 
demo.launch()