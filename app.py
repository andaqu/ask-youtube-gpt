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

tt = TextTilingTokenizer()
searcher = SemanticSearch()

def get_youtube_data(url):

    video_id = url.split("=")[1]

    raw = YouTubeTranscriptApi.get_transcript(video_id)

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

def get_segments(df, title, author, split_by_topic, segment_length = 200):

    transcript = df['text'].str.cat(sep=' ')

    if not split_by_topic:
        words = transcript.split()
        segments = [' '.join(words[i:i+segment_length]) for i in range(0, len(words), segment_length)]
    else:
        segments = tt.tokenize(transcript)

    segments = [segment.replace('\n\n','').strip() for segment in segments]

    segments_wc = [len(segment.split()) for segment in segments]
    segments_wc = np.cumsum(segments_wc)

    idx = [np.argmin(np.abs(df['total_words'] - total_words)) for total_words in segments_wc]

    segments_end_times = df['end'].iloc[idx].values
    segments_end_times = np.insert(segments_end_times, 0, 0.0)

    segments_times = [(to_timestamp(segments_end_times[i-1]), to_timestamp(segments_end_times[i])) for i in range(1,len(segments_end_times))]

    segments_text = [f"Segment from '{title}' by {author}\nSegment timestamp: {segment_time}\n\n{segment}" for segment, segment_time in zip(segments, segments_times)]

    return segments_text

def fit_searcher(segments, n_neighbors):
    global searcher
    searcher.fit(segments, n_neighbors)
    return True

def num_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def form_query(question, model, token_budget):

    results = searcher(question)

    introduction = 'Use the below segments from multiple youtube videos to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer." Cite each reference using the [title, author, timestamp] notation. Every sentence should have a citation at the end.'

    message = introduction

    question = f"\n\nQuestion: {question}"

    reference = []

    for result in results:
        result = "\n\n" + result
        if (
            num_tokens(message + result + question, model=model)
            > token_budget
        ):
            break
        else:
            reference.append(result)
            message += result

    return message + question, reference

def generate_answer(question, model, token_budget):
    
    message, reference = form_query(question, model, token_budget)

    messages = [
        {"role": "system", "content": "You answer questions about legal contracts."},
        {"role": "user", "content": message},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    
    response_message = response["choices"][0]["message"]["content"]
    return response_message, reference


if False:
    data = {}

    question = "Why do some men have trouble with feminism?"
    n_neighbors = 5

    urls = ["https://www.youtube.com/watch?v=4xWJf8cERoM", "https://www.youtube.com/watch?v=vx-Si9gbijA"]
    segments = []

    for url in urls:
        df, title, author = get_youtube_data(url)

        video_segments = get_segments(df, title, author, split_by_topic = True)

        segments.extend(video_segments)

    print("Segments generated successfully!")

    if fit_searcher(segments, n_neighbors):
        print("Searcher fit successfully!")
        answer, reference = generate_answer(question, model = "gpt-3.5-turbo", token_budget = 1000)
        print(answer)
        print(reference)

title = "Ask Youtube GPT"

description = """  """

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(f'Ask YouTube GPT allows you to ask questions about a set of Youtube Videos using Universal Sentence Encoder and Open AI. The returned response cites the video title, author and timestamp in square brackets where the information is located, adding credibility to the responses and helping you to locate incorrect information. If you need one, get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a>.</p>')

    with gr.Row():
        
        with gr.Group():
            
            openAI_key=gr.Textbox(label='Enter your OpenAI API key here')

            # Allow the user to input multiple links, adding a textbox for each
            links = gr.Textbox(lines=5, label="Enter the links to the YouTube videos you want to search (one per line):", placeholder="https://www.youtube.com/watch?v=4xWJf8cERoM\nhttps://www.youtube.com/watch?v=vx-Si9gbijA")

            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        # btn.click(question_answer, inputs=[url, file, question,openAI_key], outputs=[answer])    
            
#openai.api_key = os.getenv('Your_Key_Here') 
demo.launch()