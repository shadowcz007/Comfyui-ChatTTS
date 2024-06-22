from openvoice import se_extractor
from openvoice.api import ToneColorConverter

import comfy.model_management as mm

import folder_paths

import os,torch

# 修改模型的本地缓存地址
# os.environ['HF_HOME'] = os.path.join(folder_paths.models_dir,'chat_tts')

def get_model_dir(m):
    try:
        return folder_paths.get_folder_paths(m)[0]
    except:
        return os.path.join(folder_paths.models_dir, m)

ckpt_converter=get_model_dir('open_voice')

# device = mm.get_torch_device()
device="cuda:0" if torch.cuda.is_available() else "cpu"

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

def run(reference_speaker="",src_path="",save_path=""):

    if reference_speaker != "" and src_path!="":
        # Run the base speaker tts
        print("Ready for voice cloning!")
        
        target_dir=os.path.join(ckpt_converter,'processed')
        source_se, audio_name = se_extractor.get_se(src_path, tone_color_converter, target_dir=target_dir, vad=True)

        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir=target_dir, vad=True)

        # Run the tone color converter
        # convert from file
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path)
      