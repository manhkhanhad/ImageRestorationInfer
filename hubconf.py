from pix2pixColorization.colorization import ColorizationModel

def pix2pixColorization():
    url = ""
    model = ColorizationModel()
    model.train(url)
    return model