
import os
import sys
from pathlib import Path
import torchaudio
import hashlib
import torch
import folder_paths
import comfy.utils

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# print('current_file_path',current_file_path)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# print('current_directory',current_directory)

# 加载python模块的目录，确认是否有当前插件的nodes路径
# print('sys.path',sys.path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)


# 需要了解python的class是什么意思
class ChatTTSNode:
    def __init__(self):
        self.speaker = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text":  ("STRING", 
                                     {
                                       "default": "[laugh][uv_break]大家好，我是shadow [uv_break]", 
                                       "multiline": True,
                                       "dynamicPrompts": True # comfyui 动态提示
                                       }
                                    ),
                        "random_speaker":("BOOLEAN", {"default": False},), # 是否需要随机发音人
                        }
                }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text,random_speaker):
        # 传入的文本
        print(text)

        audio_file="chat_tts"

        import importlib
        # 模块名称
        module_name = 'chat_tts_run'

        # 动态加载模块
        module = importlib.import_module(module_name)
        
        if random_speaker:
            self.speaker=None

        # 使用加载的模块
        result,rand_spk=module.run(audio_file,text,self.speaker)

        self.speaker=rand_spk
        
        # # 需要运行chat tts 的代码
        # import ChatTTS
        # from IPython.display import Audio

        # chat = ChatTTS.Chat()
        # chat.load_models(compile=False) # 设置为True以获得更快速度

        # texts = [text,]

        # wavs = chat.infer(texts, use_decoder=True)

        # torchaudio.save(audio_file, torch.from_numpy(wavs[0]), 24000)

        return (result,)


import re

def extract_speech(content):
    # 定义正则表达式来捕获人名和讲话内容，使用非贪婪匹配
    pattern = re.compile(r'(\w+)：\[uv_break\](.*?)(?=\n|\Z)', re.DOTALL)
    
    # 查找所有匹配的内容
    matches = pattern.findall(content)
    
    # 构建结果列表
    result = []
    for index, (name, text) in enumerate(matches):
        result.append({
            'name': name.strip(),
            'text': text.strip(),
            'index': index
        })
    
    return result

# 测试内容
content = '''
 [laugh][uv_break]小明：[uv_break]大家好，欢迎收听本周的《AI新动态》。我是主持人小明，今天我们有两位嘉宾，分别是小李和小王。大家跟听众打个招呼吧
小李：[uv_break]大家好，我是小李，很高兴今天能和大家聊聊最新的AI动态。
小王：[uv_break]大家好，我是小王，也很期待今天的讨论。
[uv_break]
'''

# 调用方法并打印结果
speech_list = extract_speech(content)
for speech in speech_list:
    print(speech)



def calculate_tensor_hash(tensor, hash_algorithm='md5'):
    # 将 tensor 转换为字节
    tensor_bytes = tensor.numpy().tobytes()

    # 创建哈希对象
    hash_func = hashlib.new(hash_algorithm)

    # 更新哈希对象
    hash_func.update(tensor_bytes)

    # 返回哈希值的十六进制表示，截取前8个字符
    return hash_func.hexdigest()[:8]


def merge_audio_files(file_list):
    waveforms = []
    sample_rate = None

    # 加载所有音频文件
    for file_path in file_list:
        waveform, current_sample_rate = torchaudio.load(file_path)
        if sample_rate is None:
            sample_rate = current_sample_rate
        else:
            assert sample_rate == current_sample_rate, "采样率不一致"
        
        waveforms.append(waveform)

    # 合并音频文件
    combined_waveform = torch.cat(waveforms, dim=1)

    id=calculate_tensor_hash(combined_waveform)

    output_dir = folder_paths.get_output_directory()
    
    # print('#audio_path',folder_paths, )
    # 添加文件名后缀
    audio_file = f"podcast_{id}.wav"
    
    audio_path=os.path.join(output_dir, audio_file)

    # 保存合并后的音频文件
    torchaudio.save(audio_path, combined_waveform, sample_rate)
    
    return {
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "audio_path":audio_path
                }

# # 示例用法
# file_list = ["audio1.wav", "audio2.wav", "audio3.wav"]
# output_file = "combined_audio.wav"
# merge_audio_files(file_list, output_file)




# 生产多角色的播客节目
class multiPersonPodcast:
    def __init__(self):
        self.speaker={}
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text":  ("STRING", 
                                     {
                                       "default": '''
 [laugh][uv_break]小明：[uv_break]大家好，欢迎收听本周的《AI新动态》。我是主持人小明，今天我们有两位嘉宾，分别是小李和小王。大家跟听众打个招呼吧！
小李：[uv_break]大家好，我是小李，很高兴今天能和大家聊聊最新的AI动态。
小王：[uv_break]大家好，我是小王，也很期待今天的讨论。
[uv_break]
'''.strip(), 
                                       "multiline": True,
                                       "dynamicPrompts": True # comfyui 动态提示
                                       }
                                    ),
                        }
                }
    
    RETURN_TYPES = ("AUDIO","AUDIO",)
    RETURN_NAMES = ("audio_list","audio",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text):
        
        speech_list = extract_speech(text)

        print(speech_list)

        self.speaker = {}
        for speech in speech_list:
            self.speaker[speech['name']] = None
             
        import importlib
        # 模块名称
        module_name = 'chat_tts_run'

        # 动态加载模块
        module = importlib.import_module(module_name)

        podcast=[]
        audio_paths=[]
       
        pbar = comfy.utils.ProgressBar(len(speech_list))

        for speech in speech_list:
            audio_file="chat_tts_"+speech['name']+"_"+str(speech['index'])+"_"
            spk=self.speaker[speech['name']]

            result,rand_spk=module.run(audio_file,speech['text'],spk)

            self.speaker[speech['name']]=rand_spk

            result={**speech, **result}

            podcast.append(result)

            audio_paths.append(result['audio_path'])
            pbar.update(1)

        last_result=merge_audio_files(audio_paths)

        return (podcast,last_result,)
    