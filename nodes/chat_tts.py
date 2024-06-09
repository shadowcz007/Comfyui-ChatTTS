
import os
import sys
from pathlib import Path


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

print('current_file_path',current_file_path)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

print('current_directory',current_directory)

# 加载python模块的目录，确认是否有当前插件的nodes路径
print('sys.path',sys.path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)


# 需要了解python的class是什么意思
class ChatTTSNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text":  ("STRING", 
                                     {
                                       "default": "大家好，我是shadow", 
                                       "multiline": True,
                                       "dynamicPrompts": True # comfyui 动态提示
                                       }
                                    ),
                        }
                }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text):
        # 传入的文本
        print(text)

        audio_file="chat_tts"

        import importlib
        # 模块名称
        module_name = 'chat_tts_run'

        # 动态加载模块
        module = importlib.import_module(module_name)

        # 使用加载的模块
        result=module.run(audio_file,text)

        # # 需要运行chat tts 的代码
        # import ChatTTS
        # from IPython.display import Audio

        # chat = ChatTTS.Chat()
        # chat.load_models(compile=False) # 设置为True以获得更快速度

        # texts = [text,]

        # wavs = chat.infer(texts, use_decoder=True)

        # torchaudio.save(audio_file, torch.from_numpy(wavs[0]), 24000)

        return (result,)
    

