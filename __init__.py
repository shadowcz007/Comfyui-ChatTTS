from .nodes.chat_tts import ChatTTSNode,multiPersonPodcast
from .nodes.audio_play import AudioPlayNode_TEST

NODE_CLASS_MAPPINGS = {
    "ChatTTS_": ChatTTSNode,
    "MultiPersonPodcast":multiPersonPodcast,
    "AudioPlayNode_TEST":AudioPlayNode_TEST
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTS_": "ChatTTS",
    "MultiPersonPodcast":"Multi Person Podcast"
}

# web ui的节点功能
WEB_DIRECTORY = "./web0007"