import torch
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import numpy as np
import os
import folder_paths 
import node_helpers


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def create_temp_file(image):
    output_dir = folder_paths.get_temp_directory()

    (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path('material', output_dir)

    
    image=tensor2pil(image)
 
    image_file = f"{filename}_{counter:05}.png"
     
    image_path=os.path.join(full_output_folder, image_file)

    image.save(image_path,compress_level=4)

    return (image_path,[{
    "filename": image_file,
    "subfolder": subfolder,
    "type": "temp"
    }])


# image - tensor - 文件路径
# loadImage的方法（ 文件路径 - image-mask ）


class EditMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",), # 表示一个张量
                     
                     },

                      "optional":{
                            "image_update": ("IMAGE_",)
                        },
                   
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    OUTPUT_NODE = True

    def load_image(self, image,image_update=None):

        image_path=None
        print('#image_update',image_update)
        if image_update==None:
            print('--')
        else:
            if 'images' in image_update:
                images=image_update['images']
                filename=images[0]['filename']
                subfolder=images[0]['subfolder']
                name, base_dir=folder_paths.annotated_filepath(filename)
                base_dir = folder_paths.get_input_directory()  
                print(base_dir,subfolder, name)
                image_path = os.path.join(base_dir,subfolder, name)
        
        if image_path==None:
            image_path,images=create_temp_file(image)
        print('#image_path',image_path)
        # image_path = folder_paths.get_annotated_filepath(image) #文件名
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return {"ui":{"images": images},"result": (output_image, output_mask)}

        # return (output_image, output_mask)
    

class AudioPlayNode_TEST:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                }, 
                }
    
    RETURN_TYPES = ()
  
    FUNCTION = "run"

    CATEGORY = "♾️Mixlab_Test_ChatTTS"

    INPUT_IS_LIST = False
    # OUTPUT_IS_LIST = False

    OUTPUT_NODE = True #当前节点需要运行一次
  
    def run(self,audio):

        print('#audio',audio)
        #py 列表 [ ]  js   数组Array [ ] 
        return { "ui": { "audio1":[audio,1],"tes2":[333] } }
    