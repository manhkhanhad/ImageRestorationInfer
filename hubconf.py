#from pix2pixColorization.networks import defind_G
from pix2pixColorization import networks
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torchvision.models.resnet import resnet18
import sys
import torch
sys.path.insert(0, 'pix2pixColorization')


def pix2pixColorization_unet256():
    gpu_ids = []
    url = "https://github.com/manhkhanhad/ImageRestorationInfer/releases/download/Pretrained/latest_net_G.pth"


    model = networks.define_G(
            input_nc=1,
            output_nc=2,
            ngf=64,
            netG="unet_256",
            norm="batch",
            use_dropout=True,
            init_type="normal",
            init_gain=0.02,
            gpu_ids=gpu_ids,
        )
    # get device name: CPU or GPU
    device = (
        torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu")
    )

    state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)
    
    return model

def pix2pixColorization_resnet9b():
    gpu_ids = []
    url = "https://github.com/manhkhanhad/ImageRestorationInfer/releases/download/Pretrained/latest_net_G_resnet9b.pth"


    model = networks.define_G(
            input_nc=1,
            output_nc=2,
            ngf=64,
            netG="resnet_9blocks",
            norm="batch",
            use_dropout=True,
            init_type="normal",
            init_gain=0.02,
            gpu_ids=gpu_ids,
        )
    # get device name: CPU or GPU
    device = (
        torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu")
    )

    state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)
    
    return model

def DeOldifyColorization():
    gpu_ids = []
    url = "https://github.com/manhkhanhad/ImageRestorationInfer/releases/download/Pretrained/latest_net_G_deoldify_newWeight.pth"

    body = create_body(resnet18, pretrained=False, n_in=3, cut=-2)
    model = DynamicUnet(body, 3, (256, 256),self_attention=True)

    device = (
        torch.device("cuda:{}".format(gpu_ids[0])) if gpu_ids else torch.device("cpu")
    )

    state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)
    
    return model


# if __name__ == '__main__':
#     pix2pixColorization()
