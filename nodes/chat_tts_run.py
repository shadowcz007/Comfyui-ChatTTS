import ChatTTS

import torchaudio,torch

import folder_paths

import os


import folder_paths

# 修改模型的本地缓存地址
# os.environ['HF_HOME'] = os.path.join(folder_paths.models_dir,'chat_tts')

model_local_path=os.path.join(folder_paths.models_dir,'chat_tts')


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

    # 添加文件名后缀
    audio_file = f"{audio_file}_{counter:05}.wav"
    
    audio_path=os.path.join(full_output_folder, audio_file)

    # from IPython.display import Audio
    print(audio_path)
    chat = ChatTTS.Chat()
    chat.load_models(local_path=model_local_path,compile=False) # 设置为True以获得更快速度

    texts = [text,]

    wavs = chat.infer(texts, use_decoder=True)

    torchaudio.save(audio_path, torch.from_numpy(wavs[0]), 24000)

    return audio_path