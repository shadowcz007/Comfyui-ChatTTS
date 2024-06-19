from .nodes.chat_tts import ChatTTSNode,multiPersonPodcast,CreateSpeakers


NODE_CLASS_MAPPINGS = {
    "ChatTTS_": ChatTTSNode,
    "CreateSpeakers":CreateSpeakers,
    "MultiPersonPodcast":multiPersonPodcast
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTS_": "ChatTTS",
    "MultiPersonPodcast":"Multi Person Podcast",
    "CreateSpeakers":"Create Speakers",
}