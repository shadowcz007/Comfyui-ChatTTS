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
    