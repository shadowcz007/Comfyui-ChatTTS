
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

    CATEGORY = "♾️Mixlab/Audio"

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


def remove_brackets(text):
    pattern = re.compile(r'\[.*?\]')
    return re.sub(pattern, '', text)

# 示例
# text = "这是一个例子[uv_break]，其中包含[laugh]和[oral]。"
# cleaned_text = remove_brackets(text)
# print(cleaned_text)


def extract_speech(content):
    # 定义正则表达式来捕获人名和讲话内容，使用非贪婪匹配
    # pattern = re.compile(r'(\w+)：\[uv_break\](.*?)(?=\n|\Z)', re.DOTALL)
    pattern = re.compile(r'(\w+)(：|:)(.*?)(?=\n|\Z)', re.DOTALL)

    # 查找所有匹配的内容
    matches = pattern.findall(content)
    # print(matches)
    # 构建结果列表
    result = []
    for index, (name,_, text) in enumerate(matches):
        result.append({
            'name': name.strip().lower(),
            'text': remove_brackets(text).strip(),
            'index': index
        })
    
    return result



# # 测试内容
content = '''
[laugh][uv_break]Alex：[uv_break] 大家好，欢迎收听我们的播客！今天我们要讨论一个非常有趣的话题：SD3 large在制作产品模型方面的表现。我们有幸邀请到了几位专家，一起来探讨这个话题。首先，请大家自我介绍一下。[uv_break]
Jordan：[uv_break]大家好，我是Jordan，[uv_break]一名产品设计师。我一直在寻找新技术来提升我们的设计流程，最近对生成式AI特别感兴趣。
Taylor：[uv_break]大家好，我是Taylor，专注于计算机视觉和机器学习模型的训练和优化。[uv_break]我对SD3 large的技术细节非常感兴趣。[uv_break]
Morgan：[uv_break]大家好，我是Morgan，一名用户体验设计师。[uv_break]我关注的是技术如何能更好地提升用户体验。
Alex：[uv_break]太好了，欢迎大家！[uv_break]我们今天的主题是SD3 large在制作产品模型方面的表现。Jordan，你作为产品设计师，能先分享一下你对SD3 large的初步印象吗？
Jordan：[uv_break]当然。SD3 large在生成产品模型方面表现非常出色，特别是在产品摄影和背景生成上。它能快速生成高质量的模型，大大缩短了我们的设计时间[uv_break]。
Alex：[uv_break]确实如此。Taylor，你作为计算机视觉工程师，[uv_break]能否给我们讲讲SD3 large的技术优势？[uv_break]
Taylor：[uv_break]好的。SD3 large使用了最新的生成式AI技术，能够处理大量参数，生成逼真的产品模型。而且，它在背景生成方面也非常强大，可以根据需求自动调整背景，提高了模型的真实感。[uv_break]
Morgan：[uv_break] 这听起来很棒。我想知道，[uv_break]这些技术如何能提升用户体验？[uv_break]
Jordan：[uv_break] 这是个好问题。[uv_break]通过使用SD3 large，我们可以更快地推出高质量的产品模型，这不仅提高了设计效率，还能更快地回应用户需求，提升用户满意度。
Morgan：[uv_break] 没错，快速响应用户需求是提升用户体验的关键。Alex，[uv_break]你作为生成式AI专家，怎么看待SD3 large在设计领域的未来应用？
Alex：[uv_break]我认为SD3 large在设计领域有非常广阔的应用前景。它不仅能提高设计效率，还能激发设计师的创意。随着技术的不断进步，我们可以期待更多创新的应用场景。[uv_break]特别是虚拟生产和预售产品方面，这些技术可以大大降低成本和时间。
Taylor：[uv_break] 是的，生成式AI的潜力是巨大的。[uv_break]我们可以利用它来创建更加复杂和逼真的模型，甚至在设计阶段就能进行用户测试，进一步优化产品。对于预售产品，生成式AI可以帮助我们在产品正式发布前就进行市场测试，减少失败风险。
Jordan：[uv_break]我完全同意。SD3 large不仅是一个工具，更是一个创意的催化剂。希望未来我们能看到更多这样的技术应用于设计领域。虚拟生产也可以让我们在实际生产前就进行各种测试和调整，确保产品的高质量。
Morgan：[uv_break]另外，虚拟生产还能帮助我们进行可持续设计。通过模拟材料和生产过程，我们可以在设计阶段就考虑环保因素，减少浪费。预售产品的应用也可以让我们更好地了解市场需求，从而做出更符合市场需求的产品。
Alex：[uv_break]为了让我们的听众更好地理解这些技术，我想补充一些实际案例。例如，一些大品牌已经开始使用生成式AI来创建广告素材，这不仅减少了制作时间，还提高了广告的个性化程度。
Morgan：[uv_break]对，这些技术还可以用来进行市场调研。[uv_break]通过生成不同风格的产品模型，我们可以更好地了解消费者的偏好，从而做出更符合市场需求的产品。
Taylor：[uv_break] 另外，生成式AI还能帮助我们进行可持续设计。通过模拟材料和生产过程，我们可以在设计阶段就考虑环保因素，减少浪费。
Jordan：[uv_break] 这些都是非常有启发性的信息。对于设计师来说，生成式AI不仅是一个工具，[uv_break]更是一个新的创意伙伴，帮助我们突破传统设计的局限。[uv_break]
Alex：[uv_break]非常感谢大家的精彩讨论！[uv_break]今天的播客就到这里，希望大家对SD3 large在产品模型制作中的应用有了更深入的了解。感谢各位嘉宾的参与，[uv_break]我们下次再见！
Jordan：[uv_break] 再见！[uv_break]
'''

# # # 调用方法并打印结果
# speech_list = extract_speech(content)
# for speech in speech_list:
#     print(speech)



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




# 生产多角色的语音，可以先听下音色
class CreateSpeakers:
    def __init__(self):
        self.speaker=None
        self.seed=None
        self.last_result=None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text":  ("STRING", 
                                     {
                                       "default":'''小明：大家好，欢迎收听本周的《AI新动态》。我是主持人小明，今天我们有两位嘉宾，分别是小李和小王。大家跟听众打个招呼吧！
                                       小李：大家好，我是小李，很高兴今天能和大家聊聊最新的AI动态。
                                       小王：大家好，我是小王，也很期待今天的讨论。
                                            '''.strip(), 
                                       "multiline": True,
                                       "dynamicPrompts": True # comfyui 动态提示
                                       }
                                    ),
                        
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}), #默认的seed控件
                      
                        }
                }
    
    RETURN_TYPES = ("AUDIO","SPEAKER",)
    RETURN_NAMES = ("audio","speaker",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text,seed):
        
        speech_list = extract_speech(text)

        print('seed',seed,speech_list)

        is_new=False

        if self.seed==None:
            self.seed=seed
            is_new=True
        elif self.seed!=seed:
            self.seed=seed
            is_new=True

        if self.speaker==None:
            self.seed=seed
            is_new=True
        else:
            count=len(self.speaker.keys())
            for speech in speech_list:
                if speech['name'] in  self.speaker:
                    count-=1
            if count>0:
                is_new=True
            print('#count',count)

        if self.last_result==None:
            is_new==True
        
        if is_new==False and self.last_result and self.speaker:
            return (self.last_result ,self.speaker)
        # elif self.last_result==None and is_new==False and self.speaker:

        self.speaker = {}
        for speech in speech_list:
            self.speaker[speech['name']] = None

        import importlib
        # 模块名称
        module_name = 'chat_tts_run'

        # 动态加载模块
        module = importlib.import_module(module_name)

        audio_paths=[]
       
        pbar = comfy.utils.ProgressBar(len(speech_list))

        for name in self.speaker.keys():
            audio_file="chat_tts_"+name+"_"
            spk=self.speaker[name]

            result,rand_spk=module.run(audio_file,
                                       f'Hello 我是{name},你好，欢迎来到mixlab无界社区',
                                       spk,
                                       None,None,None,
                                       3)

            self.speaker[name]=rand_spk

            result={**speech, **result}
 
            audio_paths.append(result['audio_path'])
            pbar.update(1)

        self.last_result = merge_audio_files(audio_paths)

        return (self.last_result ,self.speaker)


def get_speaker_model_path():
    try:
        return folder_paths.get_folder_paths('chat_tts_speaker')[0]
    except:
        return os.path.join(folder_paths.models_dir, "chat_tts_speaker")


def get_files_with_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                file_name = os.path.relpath(file_path, directory)
                file_list.append(file_name)
    return file_list


#todo 保存音色文件，加载音色
class LoadSpeaker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "model":(get_files_with_extension(get_speaker_model_path(),'.pt'),),
                        }
                }
    
    RETURN_TYPES = ("SPEAKER",)
    RETURN_NAMES = ("speaker",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def run(self,model):

        model = model+'.pt'

        model_path=os.path.join(get_speaker_model_path(),model)


        self.speaker={}

        for k,v in torch.load(model_path).items():
            self.speaker[k.lower()]=v

        return {"ui": {"text": self.speaker.keys()}, "result": (self.speaker,)}



class SaveSpeaker:
    def __init__(self):
        self.speaker=None
        
        # 模型位置
        self.model_path=get_speaker_model_path()
        if not os.path.exists(self.model_path):
            # 如果目录不存在，则创建它
            os.makedirs(self.model_path)


    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                         "speaker": ("SPEAKER", {"forceInput": True}), 
                         "filename_prefix":("STRING", {"multiline": False,"default": "mixlab_tts"})
                        }
                }
    
    RETURN_TYPES = ("SPEAKER_FILE",)
    RETURN_NAMES = ("speaker_file",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,speaker,filename_prefix):
        self.speaker=speaker

        # output_dir = folder_paths.get_output_directory()

        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, self.model_path)

        file = f"{filename}_{counter:05}.pt"

        f_path=os.path.join(full_output_folder, file)

        # 保存张量
        torch.save(speaker, f_path)

        return ({
                "filename": file,
                "subfolder": "chat_tts_speaker",
                "type":"model"
            },)
    


# 生产多角色的播客节目
class multiPersonPodcast:
    def __init__(self):
        self.speaker={}
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text":  ("STRING", 
                                     {
                                       "default":'''小明：大家好，欢迎收听本周的《AI新动态》。我是主持人小明，今天我们有两位嘉宾，分别是小李和小王。大家跟听众打个招呼吧！
                                       小李：大家好，我是小李，很高兴今天能和大家聊聊最新的AI动态。
                                       小王：大家好，我是小王，也很期待今天的讨论。
                                            '''.strip(), 
                                       "multiline": True,
                                       "dynamicPrompts": True # comfyui 动态提示
                                       }
                                    ),
                        
                        "uv_speed": ("INT",{
                                "default":0, 
                                "min": 0, #Minimum value
                                "max": 9, #Maximum value
                                "step": 1, #Slider's step
                                "display": "slider" # Cosmetic only: display as "number" or "slider"
                            }),
                        "uv_oral": ("INT",{
                                "default":0, 
                                "min": 0, #Minimum value
                                "max": 9, #Maximum value
                                "step": 1, #Slider's step
                                "display": "slider" # Cosmetic only: display as "number" or "slider"
                            }),
                        "uv_laugh": ("INT",{
                                "default":0, 
                                "min": 0, #Minimum value
                                "max": 9, #Maximum value
                                "step": 1, #Slider's step
                                "display": "slider" # Cosmetic only: display as "number" or "slider"
                            }),
                        "uv_break": ("INT",{
                                "default":0, 
                                "min": 0, #Minimum value
                                "max": 9, #Maximum value
                                "step": 1, #Slider's step
                                "display": "slider" # Cosmetic only: display as "number" or "slider"
                            }), 
                        },
                         "optional":{ 
                                "speaker": ("SPEAKER", {"forceInput": True}), 
                        }
                }
    
    RETURN_TYPES = ("AUDIO","AUDIO",)
    RETURN_NAMES = ("audio_list","audio",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text,uv_speed,uv_oral,uv_laugh,uv_break,speaker=None):
        
        speech_list = extract_speech(text)

        print(speech_list)

        if speaker!=None:
            # 有传入speaker
            self.speaker = speaker
            for speech in speech_list:
                if not speech['name'] in self.speaker:
                    self.speaker[speech['name']] = None
        else:
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
            print('#speaker',speech['name'],not (spk is None))
            result,rand_spk=module.run(audio_file,speech['text'],spk,uv_speed,uv_oral,uv_laugh,uv_break)

            self.speaker[speech['name']]=rand_spk

            result={**speech, **result}

            podcast.append(result)

            audio_paths.append(result['audio_path'])
            pbar.update(1)

        last_result=merge_audio_files(audio_paths)

        return (podcast,last_result,)
    