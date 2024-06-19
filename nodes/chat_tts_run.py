import ChatTTS

import torchaudio,torch

import folder_paths

import os

# 修改模型的本地缓存地址
# os.environ['HF_HOME'] = os.path.join(folder_paths.models_dir,'chat_tts')

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

model_local_path=get_model_dir('chat_tts')

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


def run(audio_file,text,rand_spk,uv_speed=None,uv_oral=None,uv_laugh=None,uv_break=None):
    # 需要运行chat tts 的代码
    
    output_dir = folder_paths.get_output_directory()
    
    counter=get_new_counter(output_dir,audio_file)
    # print('#audio_path',folder_paths, )
    # 添加文件名后缀
    audio_file = f"{audio_file}_{counter:05}.wav"
    
    audio_path=os.path.join(output_dir, audio_file)

    # from IPython.display import Audio
    # print('#audio_path',audio_path)
    chat = ChatTTS.Chat()
    chat.load_models(local_path=model_local_path,compile=False) # 设置为True以获得更快速度

    texts = [text,]

    params_refine_text = {
        'prompt': f''
    } 

    if uv_oral:
        params_refine_text['prompt']+=f'[oral_{uv_oral}]'

    if uv_laugh:
        params_refine_text['prompt']+=f'[laugh_{uv_laugh}]'
    
    if uv_break:
        params_refine_text['prompt']+=f'[break_{uv_break}]'
    
    if uv_speed:
        params_refine_text['prompt']+=f'[speed_{uv_speed}]'

    if rand_spk is None:
        rand_spk = chat.sample_random_speaker()

    params_infer_code = {
    'spk_emb': rand_spk, # add sampled speaker 
    'temperature': .3, # using custom temperature
    'top_P': 0.7, # top P decode
    'top_K': 20, # top K decode
    }

   
    # ChatTTS使用pynini对中英文进行处理，目前在window上安装报错，需要编译环境,
    # 暂时把do_text_normalization关掉
    wavs = chat.infer(texts, 
                      use_decoder=True,
                      do_text_normalization=False,
                      params_refine_text=params_refine_text,
                      params_infer_code=params_infer_code)

    torchaudio.save(audio_path, torch.from_numpy(wavs[0]), 24000)

    return ({
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "prompt":text,
                "audio_path":audio_path
                },rand_spk)