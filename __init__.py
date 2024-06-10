from .nodes.chat_tts import ChatTTSNode,multiPersonPodcast


NODE_CLASS_MAPPINGS = {
    "ChatTTS_": ChatTTSNode,
    "MultiPersonPodcast":multiPersonPodcast
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTS_": "ChatTTS",
    "MultiPersonPodcast":"Multi Person Podcast"
}