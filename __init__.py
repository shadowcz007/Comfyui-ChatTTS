from .nodes.chat_tts import ChatTTSNode,multiPersonPodcast,CreateSpeakers,SaveSpeaker,LoadSpeaker,MergeSpeaker,RenameSpeaker


NODE_CLASS_MAPPINGS = {
    "ChatTTS_": ChatTTSNode,
    "CreateSpeakers":CreateSpeakers,
    "MultiPersonPodcast":multiPersonPodcast,
    "SaveSpeaker":SaveSpeaker,
    "LoadSpeaker":LoadSpeaker,
    "MergeSpeaker":MergeSpeaker,
    "RenameSpeaker":RenameSpeaker
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTS_": "ChatTTS",
    "MultiPersonPodcast":"Multi Person Podcast",
    "CreateSpeakers":"Create Speakers",
    "SaveSpeaker":"Save Speaker",
    "LoadSpeaker":"Load Speaker",
    "MergeSpeaker":"Merge Speaker",
    "RenameSpeaker":"Rename Speaker"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"
