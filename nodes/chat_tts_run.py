import ChatTTS

import torchaudio,torch

import folder_paths

import os

def run(audio_file,text):
    # 需要运行chat tts 的代码
    
    output_dir = folder_paths.get_output_directory()
    
    (
        full_output_folder,
        filename,
        counter,
        subfolder,
         _,
    ) = folder_paths.get_save_image_path('mixlab_chat_tts', output_dir)

    audio_path=os.path.join(full_output_folder, audio_file)

    # from IPython.display import Audio
    print(audio_path)
    chat = ChatTTS.Chat()
    chat.load_models(compile=False) # 设置为True以获得更快速度

    texts = [text,]

    wavs = chat.infer(texts, use_decoder=True)

    torchaudio.save(audio_path, torch.from_numpy(wavs[0]), 24000)

    return audio_path