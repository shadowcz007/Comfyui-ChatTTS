from .nodes.chat_tts import ChatTTSNode,multiPersonPodcast,CreateSpeakers,SaveSpeaker,LoadSpeaker,MergeSpeaker,RenameSpeaker,OpenVoiceClone,LoadWhisperModel,WhisperTranscribe,OpenVoiceCloneBySpeaker


NODE_CLASS_MAPPINGS = {
    "ChatTTS_": ChatTTSNode,
    "CreateSpeakers":CreateSpeakers,
    "MultiPersonPodcast":multiPersonPodcast,
    "OpenVoiceClone":OpenVoiceClone,
    "OpenVoiceCloneBySpeaker":OpenVoiceCloneBySpeaker,
    "SaveSpeaker":SaveSpeaker,
    "LoadSpeaker":LoadSpeaker,
    "MergeSpeaker":MergeSpeaker,
    "RenameSpeaker":RenameSpeaker,
    "LoadWhisperModel":LoadWhisperModel,
    "WhisperTranscribe":WhisperTranscribe
}

# dict = { "key":value }

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatTTS_": "ChatTTS",
    "MultiPersonPodcast":"Multi Person Podcast",
    "CreateSpeakers":"Create Speakers",
    "OpenVoiceClone":"OpenVoice Clone",
    "OpenVoiceCloneBySpeaker":"OpenVoice Clone By Speaker",
    "SaveSpeaker":"Save Speaker",
    "LoadSpeaker":"Load Speaker",
    "MergeSpeaker":"Merge Speaker",
    "RenameSpeaker":"Rename Speaker",
    "LoadWhisperModel":"Load Whisper Model",
    "WhisperTranscribe":"Whisper Transcribe"
}

# web ui的节点功能
WEB_DIRECTORY = "./web"
