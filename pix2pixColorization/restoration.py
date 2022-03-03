import torch
from networks import define_G
#Load model

class RestorationModel:
    def __init__(self, url, gpu_ids = []):
        device = (torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu"))
        model = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=True,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=gpu_ids,
    )
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        model.load_state_dict(state_dict)
    
    def __call__(self, image):
        return self.model(image)