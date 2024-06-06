import ChatTTS

import torchaudio,torch

import folder_paths

import os

# 修改模型的本地缓存地址
# os.environ['HF_HOME'] = os.path.join(folder_paths.models_dir,'chat_tts')

model_local_path=os.path.join(folder_paths.models_dir,'chat_tts')

# 写一个python文件，用来 判断文件夹内命名为 所有chat_tts开头的文件数量（chat_tts_00001），并输出新的编号
def get_new_counter(full_output_folder, filename_prefix):
    # 获取目录中的所有文件
    files = os.listdir(full_output_folder)
    
    # 过滤出以 filename_prefix 开头并且后续部分为数字的文件
    filtered_files = []
    for f in files:
        if f.startswith(filename_prefix):
            # 去掉文件名中的前缀和后缀，只保留中间的数字部分
            base_name = f[len(filename_prefix)+1:]
            number_part = base_name.split('.')[0]  # 假设文件名中只有一个点，即扩展名
            if number_part.isdigit():
                filtered_files.append(int(number_part))

    if not filtered_files:
        return 1

    # 获取最大的编号
    max_number = max(filtered_files)
    
    # 新的编号
    return max_number + 1


def run(audio_file,text):
    # 需要运行chat tts 的代码
    
    output_dir = folder_paths.get_output_directory()
    
    counter=get_new_counter(output_dir,audio_file)
    # print('#audio_path',folder_paths, )
    # 添加文件名后缀
    audio_file = f"{audio_file}_{counter:05}.wav"
    
    audio_path=os.path.join(output_dir, audio_file)

    # from IPython.display import Audio
    print('#audio_path',audio_path)
    chat = ChatTTS.Chat()
    chat.load_models(local_path=model_local_path,compile=False) # 设置为True以获得更快速度

    texts = [text,]

    wavs = chat.infer(texts, use_decoder=True)

    torchaudio.save(audio_path, torch.from_numpy(wavs[0]), 24000)

    return audio_path