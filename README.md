---
title: Ask Youtube Gpt
emoji: 📺
colorFrom: white
colorTo: red
sdk: gradio
sdk_version: 3.28.0
app_file: app.py
pinned: false
---

# Ask Youtube GPT

Ask YouTube GPT allows you to ask questions about a set of YouTube Videos using Topic Segmentation, Universal Sentence Encoding, and Open AI.

## Ideas / Future Improvements

- [x] Omit the need for a set of videos, and instead use a search query to find videos on the fly.
- [ ] Add "Suggest a question" feature given the videos (maybe through clustering?)
- [ ] Add explainable segment retrieval (i.e. why did that specific segment get chosen to answer the question?)
- [ ] Add OpenAI embeddings
- [ ] Improve the retrieval process allowing for the retrieval to return at least one segment from each video (This would in turn enable comparision of videos, i.e. which video explains topic X in more depth?)