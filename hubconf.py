from pix2pixColorization.colorization import ColorizationModel

def pix2pixColorization():
    gpu_ids = []
    url = "https://drive.google.com/file/d/1LjM-ml5SqJUcgmg7GtfODTcTqVGirJgM"


    gen = define_G(
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