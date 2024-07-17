import os,re
import sys,time
from pathlib import Path
import torchaudio
import hashlib
import torch
import folder_paths
import comfy.utils

from faster_whisper import WhisperModel

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 添加当前插件的nodes路径，使ChatTTS可以被导入使用
sys.path.append(current_directory)


# ref: https://github.com/jianchang512/ChatTTS-ui/blob/main/uilib/utils.py#L159
# ref: https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
from .zh_normalization import TextNormalizer

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)


def get_lang(text):
    # 定义中文标点符号的模式
    chinese_punctuation = "[。？！，、；：‘’“”（）《》【】…—\u3000]"
    # 使用正则表达式替换所有中文标点为""
    cleaned_text = re.sub(chinese_punctuation, "", text)
    # 使用正则表达式来匹配中文字符范围
    return "zh" if re.search('[\u4e00-\u9fff]', cleaned_text) is not None else "en"

# 数字转为英文读法
def num2text(text):
    numtext=[' zero ',' one ',' two ',' three ',' four ',' five ',' six ',' seven ',' eight ',' nine ']
    point=' point '
    text = re.sub(r'(\d)\,(\d)', r'\1\2', text)
    text = re.sub(r'(\d+)\s*\+', r'\1 plus ', text)
    text = re.sub(r'(\d+)\s*\-', r'\1 minus ', text)
    text = re.sub(r'(\d+)\s*[\*x]', r'\1 times ', text)
    text = re.sub(r'((?:\d+\.)?\d+)\s*/\s*(\d+)', fraction_to_words, text)

    # 取出数字 number_list= [('1000200030004000.123', '1000200030004000', '123'), ('23425', '23425', '')]
    number_list=re.findall('((\d+)(?:\.(\d+))?%?)',text)
    if len(number_list)>0:            
        #dc= ('1000200030004000.123', '1000200030004000', '123','')
        for m,dc in enumerate(number_list):
            if len(dc[1])>16:
                continue
            int_text= num_to_english(dc[1])
            if len(dc)>2 and dc[2]:
                int_text+=point+"".join([numtext[int(i)] for i in dc[2]])
            if dc[0][-1]=='%':
                int_text=f' the pronunciation of  {int_text}'
            text=text.replace(dc[0],int_text)

    return text.replace('1',' one ').replace('2',' two ').replace('3',' three ').replace('4',' four ').replace('5',' five ').replace('6',' six ').replace('7','seven').replace('8',' eight ').replace('9',' nine ').replace('0',' zero ').replace('=',' equals ')


# 针对[uv_break_3][laugh_3][oral_3][break_4][speed_2] 这些中括号包裹的英文+下横线+数字的组合，不需要替换数字为中文。




def fraction_to_words(match):
    numerator, denominator = match.groups()
    # 这里只是把数字直接拼接成了英文分数的形式, 实际上应该使用某种方式将数字转换为英文单词
    # 例如: "1/2" -> "one half", 这里仅为展示目的而直接返回了 "numerator/denominator"
    return numerator + " over " + denominator

# 切分长行 200 150
def split_text_by_punctuation(text):
    # 定义长度限制
    min_length = 150
    punctuation_marks = "。？！，、；：”’》」』）】…—"
    english_punctuation = ".?!,:;)}…"
    
    # 结果列表
    result = []
    # 起始位置
    pos = 0
    
    # 遍历文本中的每个字符
    text_length=len(text)
    for i, char in enumerate(text):
        if char in punctuation_marks or char in english_punctuation:
            if  char=='.' and i< text_length-1 and re.match(r'\d',text[i+1]):
                continue
            # 当遇到标点时，判断当前分段长度是否超过120
            if i - pos > min_length:
                # 如果长度超过120，将当前分段添加到结果列表中
                result.append(text[pos:i+1])
                # 更新起始位置到当前标点的下一个字符
                pos = i+1
    #print(f'{pos=},{len(text)=}')
    
    # 如果剩余文本长度超过120或没有更多标点符号可以进行分割，将剩余的文本作为一个分段添加到结果列表
    if pos < len(text):
        result.append(text[pos:])
    
    return result


# [中英文处理](https://github.com/jianchang512/ChatTTS-ui/blob/main/uilib/utils.py)
# 中英文数字转换为文字，特殊符号处理
def split_text(text_list):
    tx = TextNormalizer()
    haserror=False
    result=[]
    for i,text in enumerate(text_list):
        if get_lang(text)=='zh':
            tmp="".join(tx.normalize(text))
        elif haserror:
            tmp=num2text(text)
        else:
            try:
                # 先尝试使用 nemo_text_processing 处理英文
                from nemo_text_processing.text_normalization.normalize import Normalizer
                from functools import partial
                fun = partial(Normalizer(input_case='cased', lang="en").normalize, verbose=False, punct_post_process=True)
                tmp=fun(text)
                # print(f'使用nemo处理英文ok')
            except Exception as e:
                # print(f"nemo处理英文失败，改用自定义预处理")
                # print(e)
                haserror=True
                tmp=num2text(text)

        if len(tmp)>200:
            tmp_res=split_text_by_punctuation(tmp)
            result=result+tmp_res
        else:
            result.append(tmp)
    print(f'{result=},len={len(result)}')
    return result




# 需要了解python的class是什么意思
class ChatTTSNode:
    def __init__(self):
        self.speaker = None
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
                        "random_speaker":("BOOLEAN", {"default": False},), # 是否需要随机发音人
                        },
                        "optional":{ 
                                    "skip_refine_text":("BOOLEAN", {"default": False},),
                                }
                }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text,random_speaker,skip_refine_text=False):
        # 传入的文本
        # print(text)

        audio_file="chat_tts"

        import importlib
        # 模块名称
        module_name = 'chat_tts_run'

        # 动态加载模块
        module = importlib.import_module(module_name)
        
        if random_speaker:
            self.speaker=None

        do_text=split_text([text])

        # 使用加载的模块
        result,rand_spk=module.run(audio_file,do_text,self.speaker,skip_refine_text=skip_refine_text)

        self.speaker=rand_spk
        
        # # 需要运行chat tts 的代码
        # import ChatTTS
        # from IPython.display import Audio

        # chat = ChatTTS.Chat()
        # chat.load_models(compile=False) # 设置为True以获得更快速度

        # texts = [text,]

        # wavs = chat.infer(texts, use_decoder=True)

        # torchaudio.save(audio_file, torch.from_numpy(wavs[0]), 24000)

        waveform, sample_rate = torchaudio.load(result["audio_path"])

        return ({
            **result,
            "waveform": waveform.unsqueeze(0), 
            "sample_rate": sample_rate
        },)


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
    pattern = re.compile(r'([\w\.\s]+)(：|:)\s*(.*?)(?=\n|\Z)', re.DOTALL)

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



# # # 测试内容
# content = '''
# Dr.Ethan：[uv_break] 大家好，欢迎收听我们的播客！今天我们要讨论一个非常有趣的话题：SD3 large在制作产品模型方面的表现。我们有幸邀请到了几位专家，一起来探讨这个话题。首先，请大家自我介绍一下。[uv_break]
# Jordan：[uv_break]大家好，我是Jordan，[uv_break]一名产品设计师。我一直在寻找新技术来提升我们的设计流程，最近对生成式AI特别感兴趣。
# Taylor：[uv_break]大家好，我是Taylor，专注于计算机视觉和机器学习模型的训练和优化。[uv_break]我对SD3 large的技术细节非常感兴趣。[uv_break]
# Morgan：[uv_break]大家好，我是Morgan，一名用户体验设计师。[uv_break]我关注的是技术如何能更好地提升用户体验。
# Alex：[uv_break]太好了，欢迎大家！[uv_break]我们今天的主题是SD3 large在制作产品模型方面的表现。Jordan，你作为产品设计师，能先分享一下你对SD3 large的初步印象吗？
# Jordan：[uv_break]当然。SD3 large在生成产品模型方面表现非常出色，特别是在产品摄影和背景生成上。它能快速生成高质量的模型，大大缩短了我们的设计时间[uv_break]。
# Alex：[uv_break]确实如此。Taylor，你作为计算机视觉工程师，[uv_break]能否给我们讲讲SD3 large的技术优势？[uv_break]
# Taylor：[uv_break]好的。SD3 large使用了最新的生成式AI技术，能够处理大量参数，生成逼真的产品模型。而且，它在背景生成方面也非常强大，可以根据需求自动调整背景，提高了模型的真实感。[uv_break]
# Morgan：[uv_break] 这听起来很棒。我想知道，[uv_break]这些技术如何能提升用户体验？[uv_break]
# Jordan：[uv_break] 这是个好问题。[uv_break]通过使用SD3 large，我们可以更快地推出高质量的产品模型，这不仅提高了设计效率，还能更快地回应用户需求，提升用户满意度。
# Morgan：[uv_break] 没错，快速响应用户需求是提升用户体验的关键。Alex，[uv_break]你作为生成式AI专家，怎么看待SD3 large在设计领域的未来应用？
# Alex：[uv_break]我认为SD3 large在设计领域有非常广阔的应用前景。它不仅能提高设计效率，还能激发设计师的创意。随着技术的不断进步，我们可以期待更多创新的应用场景。[uv_break]特别是虚拟生产和预售产品方面，这些技术可以大大降低成本和时间。
# Taylor：[uv_break] 是的，生成式AI的潜力是巨大的。[uv_break]我们可以利用它来创建更加复杂和逼真的模型，甚至在设计阶段就能进行用户测试，进一步优化产品。对于预售产品，生成式AI可以帮助我们在产品正式发布前就进行市场测试，减少失败风险。
# Jordan：[uv_break]我完全同意。SD3 large不仅是一个工具，更是一个创意的催化剂。希望未来我们能看到更多这样的技术应用于设计领域。虚拟生产也可以让我们在实际生产前就进行各种测试和调整，确保产品的高质量。
# Morgan：[uv_break]另外，虚拟生产还能帮助我们进行可持续设计。通过模拟材料和生产过程，我们可以在设计阶段就考虑环保因素，减少浪费。预售产品的应用也可以让我们更好地了解市场需求，从而做出更符合市场需求的产品。
# Alex：[uv_break]为了让我们的听众更好地理解这些技术，我想补充一些实际案例。例如，一些大品牌已经开始使用生成式AI来创建广告素材，这不仅减少了制作时间，还提高了广告的个性化程度。
# Morgan：[uv_break]对，这些技术还可以用来进行市场调研。[uv_break]通过生成不同风格的产品模型，我们可以更好地了解消费者的偏好，从而做出更符合市场需求的产品。
# Taylor：[uv_break] 另外，生成式AI还能帮助我们进行可持续设计。通过模拟材料和生产过程，我们可以在设计阶段就考虑环保因素，减少浪费。
# Jordan：[uv_break] 这些都是非常有启发性的信息。对于设计师来说，生成式AI不仅是一个工具，[uv_break]更是一个新的创意伙伴，帮助我们突破传统设计的局限。[uv_break]
# Alex：[uv_break]非常感谢大家的精彩讨论！[uv_break]今天的播客就到这里，希望大家对SD3 large在产品模型制作中的应用有了更深入的了解。感谢各位嘉宾的参与，[uv_break]我们下次再见！
# Jordan：[uv_break] 再见！[uv_break]
# '''

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

# silence_duration 的单位是秒 (seconds)
def merge_audio_files(file_list, silence_duration=0.5, target_sample_rate=24000):
    waveforms = []

    # 加载所有音频文件并调整采样率
    for file_path in file_list:
        waveform, current_sample_rate = torchaudio.load(file_path)
        if current_sample_rate != target_sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=current_sample_rate, new_freq=target_sample_rate)
            waveform = resample_transform(waveform)
        
        waveforms.append(waveform)

    # 创建静音间隔
    silence_samples = int(silence_duration * target_sample_rate)
    silence_waveform = torch.zeros(1, silence_samples)

    # 合并音频文件并添加静音间隔
    combined_waveform = waveforms[0]
    for waveform in waveforms[1:]:
        combined_waveform = torch.cat((combined_waveform, silence_waveform, waveform), dim=1)

    id = calculate_tensor_hash(combined_waveform)

    output_dir = folder_paths.get_output_directory()
    
    # 添加文件名后缀
    audio_file = f"podcast_{id}.wav"
    
    audio_path = os.path.join(output_dir, audio_file)

    # 保存合并后的音频文件
    torchaudio.save(audio_path, combined_waveform, target_sample_rate)
    
    return {
        "filename": audio_file,
        "subfolder": "",
        "type": "output",
        "audio_path": audio_path
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
                                       "default":'''小明：大家好，欢迎收听本周的《AI新动态》。我是主持人小明，今天我们有两位嘉宾，分别是小李和小王。'''.strip(), 
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

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text,seed):
        
        speech_list = extract_speech(text)

        # print('seed',seed,speech_list)

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
                                       [f'Hello 我是{name},你好，欢迎来到mixlab无界社区'],
                                       spk,
                                       None,None,None,
                                       3)

            self.speaker[name]=rand_spk

            result={**speech, **result}
 
            audio_paths.append(result['audio_path'])
            pbar.update(1)

        self.last_result = merge_audio_files(audio_paths,0)

        return (self.last_result ,self.speaker)


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
                        "model":(get_files_with_extension(get_model_dir("chat_tts_speaker"),'.pt'),),
                        }
                }
    
    RETURN_TYPES = ("SPEAKER",)
    RETURN_NAMES = ("speaker",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def run(self,model):

        model = model+'.pt'

        model_path=os.path.join(get_model_dir("chat_tts_speaker"),model)


        self.speaker={}

        tensor=torch.load(model_path)

        if isinstance(tensor, dict):
            for k,v in tensor.items():
                self.speaker[k.lower()]=v
        else:
            self.speaker['mixlab']=tensor

        return {"ui": {"text": list(self.speaker.keys())}, "result": (self.speaker,)}



class SaveSpeaker:
    def __init__(self):
        self.speaker=None
        
        # 模型位置
        self.model_path=get_model_dir("chat_tts_speaker")
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

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_NODE = True
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
    
class MergeSpeaker:
    def __init__(self):
        self.speaker=None
       
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                         "speaker1": ("SPEAKER", {"forceInput": True}), 
                         "speaker2": ("SPEAKER", {"forceInput": True}), 
                        }
                }
    
    RETURN_TYPES = ("SPEAKER",)
    RETURN_NAMES = ("speakers",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,speaker1,speaker2):
        # print(speaker1,speaker2)
        speaker1.update(speaker2)

        self.speaker=speaker1
        
        return {"ui": {"text": list(self.speaker.keys()),"input":[list(speaker1.keys()),list(speaker2.keys())]}, "result": (self.speaker,)}



class RenameSpeaker:
    def __init__(self):
        self.speaker=None
       
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                         "speaker": ("SPEAKER", {"forceInput": True}), 
                         "name":("STRING", {"multiline": False,"default": "mixlab"})
                        }
                }
    
    RETURN_TYPES = ("SPEAKER",)
    RETURN_NAMES = ("speaker",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,speaker,name):
         
        self.speaker={}

        self.speaker[name.strip().lower()]= list(speaker.values())[0]

        return (self.speaker,)
    


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
                                "skip_refine_text":("BOOLEAN", {"default": False},),
                                "silence_duration":("FLOAT",{
                                        "default":0.5, 
                                        "min": 0, #Minimum value
                                        "max": 100, #Maximum value
                                        "step": 0.01, #Slider's step
                                        "display": "number" # Cosmetic only: display as "number" or "slider"
                                    }),
                                
                        }
                }
    
    RETURN_TYPES = ("AUDIO","AUDIO",)
    RETURN_NAMES = ("audio_list","audio",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,text,uv_speed,uv_oral,uv_laugh,uv_break,speaker=None,skip_refine_text=False,silence_duration=0.5):
        
        speech_list = extract_speech(text)

        # print(speech_list)

        if speaker!=None:
            # 有传入speaker
            self.speaker = {}
            for k,v in speaker.items():
                self.speaker[k.strip().lower()]=v

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
       
        sum=len(speech_list)
        pbar = comfy.utils.ProgressBar(sum)

        for speech in speech_list:
            audio_file="chat_tts_"+speech['name']+"_"+str(speech['index'])+"_"
            spk=self.speaker[speech['name']]
            # print('#speaker',speech['name'],not (spk is None))
            print('\033[93m#speaker', speech['name'], not (spk is None), '\033[0m')

            do_text=split_text([speech['text']])

            result,rand_spk=module.run(audio_file,do_text,spk,uv_speed,uv_oral,uv_laugh,uv_break,skip_refine_text)

            self.speaker[speech['name']]=rand_spk

            result={**speech, **result}

            waveform, sample_rate = torchaudio.load(result["audio_path"])

            podcast.append({
                **result,
                "waveform": waveform.unsqueeze(0), 
                "sample_rate": sample_rate
                })

            audio_paths.append(result['audio_path'])
            pbar.update(1)

        last_result=merge_audio_files(audio_paths,silence_duration )


        texts=["".join(split_text([s['text']])) for s in speech_list]
        last_result["prompt"]="".join(texts)


        waveform, sample_rate = torchaudio.load(last_result["audio_path"])

        return (podcast,{
            **last_result,
            "waveform": waveform.unsqueeze(0), 
            "sample_rate": sample_rate
        },)
    


    
whisper_model=get_model_dir('whisper')




class LoadWhisperModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_size": ([
                d for d in os.listdir(whisper_model) if os.path.isdir(
                    os.path.join(whisper_model, d)
                    ) and os.path.isfile(os.path.join(os.path.join(whisper_model, d), "config.json"))
                    
                    ],),
            "device": (["auto","cpu"],),
                             },
                }
    
    RETURN_TYPES = ("WHISPER",)
    RETURN_NAMES = ("whisper_model",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio/Whisper"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
    global model
    model = None
    def run(self,model_size,device):
        global model

        if device=="auto":
            device="cuda" if torch.cuda.is_available() else "cpu"

        # device="cpu"
        # "cuda" if torch.cuda.is_available() else 

        model = WhisperModel(os.path.join(whisper_model, model_size), device=device)
        # if model.device=='cuda':
        #     model.model.to('cpu')

        return (model,)
    

class WhisperTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "whisper_model": ("WHISPER",),
                                "audio": ("AUDIO",),
                             },
                }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Audio/Whisper"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
    
    def run(self,whisper_model,audio):

        audio_path=audio['audio_path']
       
        segments, info = whisper_model.transcribe(audio_path, beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        # Function to format time for SRT
        def format_time(seconds):
            millis = int((seconds - int(seconds)) * 1000)
            hours, remainder = divmod(int(seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

        # Prepare SRT content as a string
        srt_content = ""
        for i, segment in enumerate(segments):
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            srt_content += f"{i + 1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{segment.text}\n\n"

        return (srt_content,)


class OpenVoiceClone:
    def __init__(self):
        self.speaker = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "reference_audio":  ("AUDIO", ),
                        "source_audio":("AUDIO", ), 
                        },
                "optional":{
                    "whisper":("WHISPER",)
                },
                }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    FUNCTION = "ov_run"

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,) #list 列表 [1,2,3]
  
    def ov_run(self,reference_audio,source_audio,whisper=None):

        # 判断是否是 Tensor 类型
        is_dict =  isinstance(reference_audio, dict)
        # print('#判断是否是 Tensor 类型',is_tensor,audio)
        if is_dict and 'waveform' in reference_audio and 'sample_rate' in reference_audio:
            audio_path=os.path.join(folder_paths.get_temp_directory(),"reference_audio_"+str(int(time.time()))+'.wav')
            torchaudio.save(audio_path, reference_audio['waveform'].squeeze(0), reference_audio["sample_rate"])
            reference_audio["audio_path"]=audio_path

        # 判断是否是 Tensor 类型
        is_dict =  isinstance(source_audio, dict)
        # print('#判断是否是 Tensor 类型',is_tensor,audio)
        if is_dict and 'waveform' in source_audio and 'sample_rate' in source_audio:
            audio_path=os.path.join(folder_paths.get_temp_directory(),"source_audio_"+str(int(time.time()))+'.wav')
            torchaudio.save(audio_path, source_audio['waveform'].squeeze(0), source_audio["sample_rate"])
            source_audio["audio_path"]=audio_path


        # 传入的文本
        import importlib
        # 模块名称
        module_name = 'openvoice_run'

        # 动态加载模块
        module = importlib.import_module(module_name)

        
        output_dir = folder_paths.get_output_directory()

        (full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path("openvoice", output_dir)

        
        # print('#audio_path',folder_paths, )
        # 添加文件名后缀
        audio_file = f"openvoice_{counter:05}.wav"
        
        save_path=os.path.join(output_dir, audio_file)
 
        module.run(reference_audio['audio_path'],source_audio['audio_path'],save_path,whisper)

        waveform, sample_rate = torchaudio.load(save_path)
        audio = {
            "filename": audio_file,
            "subfolder": "",
            "type": "output",
            "audio_path":save_path,
            "waveform": waveform.unsqueeze(0), 
            "sample_rate": sample_rate}
        
        return (audio,)




class OpenVoiceCloneBySpeaker:
    def __init__(self):
        self.speaker=None
       
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                         "audio_list":("AUDIO",),
                         "reference_speaker": ("SPEAKER", {"forceInput": True}),
                         "reference_speaker_name":("STRING", {"multiline": False,"default": "mixlab"}),#参考reference_speaker里的哪个角色的音色
                         "source_speaker_name":("STRING", {"multiline": False,"default": "opus"}),#audio_list 里的哪个角色需要更换音色
                         "silence_duration":("FLOAT",{
                                        "default":0.5, 
                                        "min": 0, #Minimum value
                                        "max": 100, #Maximum value
                                        "step": 0.01, #Slider's step
                                        "display": "number" # Cosmetic only: display as "number" or "slider"
                                    })
                        },
                 "optional":{
                    "whisper":("WHISPER",),
                     
                },
                }
    
    RETURN_TYPES = ("AUDIO","AUDIO",)
    RETURN_NAMES = ("audio_list","audio",)

    FUNCTION = "chat_tts_run"

    CATEGORY = "♾️Mixlab/Audio/ChatTTS"

    INPUT_IS_LIST = False
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,False,) #list 列表 [1,2,3]
  
    def chat_tts_run(self,audio_list,reference_speaker,reference_speaker_name,source_speaker_name,silence_duration=0.5,whisper=None):
        name=reference_speaker_name.strip().lower()
        # print(name,speaker,silence_duration,audio_list,whisper)
        s_name=source_speaker_name.strip().lower()

        # 音色
        spk=reference_speaker[name]

        # 创建声音文件
        import importlib
        # 模块名称
        module_name = 'chat_tts_run'

        # 动态加载模块
        module = importlib.import_module(module_name)

        audio_file="chat_tts_"+name+"_"

        reference_audio,rand_spk=module.run(audio_file,
                                            [f'Hello 我是{name},你好，欢迎来到mixlab无界社区'],
                                            spk,
                                            None,None,None,3)

        
        # 动态加载模块
        openvoice_run = importlib.import_module('openvoice_run')

        output_dir = folder_paths.get_output_directory()

        def clone_voice(source_audio):
            (full_output_folder,
                filename,
                counter,
                subfolder,
                _,
            ) = folder_paths.get_save_image_path("openvoice", output_dir)

            # 添加文件名后缀
            audio_file = f"openvoice_clone_voice_{counter:05}.wav"
            save_path=os.path.join(output_dir, audio_file)
    
            openvoice_run.run(reference_audio['audio_path'],source_audio,save_path,whisper)
            waveform, sample_rate = torchaudio.load(save_path)
            audio = {
                "filename": audio_file,
                "subfolder": "",
                "type": "output",
                "audio_path":save_path,
                "waveform": waveform.unsqueeze(0), 
                "sample_rate": sample_rate}
            
            return audio
        
        audio_paths=[]
        for index in range(len(audio_list)):
            audio=audio_list[index]
            #目标角色更换音色
            a_name=audio['name'].strip().lower()
            if a_name==s_name:
                audio_list[index]=clone_voice(audio['audio_path'])
            audio_paths.append(audio_list[index]['audio_path'])

        last_result = merge_audio_files(audio_paths,silence_duration)

        waveform, sample_rate = torchaudio.load(last_result["audio_path"])

        return (audio_list,{
            **last_result,
            "waveform": waveform.unsqueeze(0), 
            "sample_rate": sample_rate
        },)
    