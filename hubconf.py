from pix2pixColorization.colorization import ColorizationModel

def pix2pixColorization():
    url = ""
    model = ColorizationModel()
    model.train("https://drive.google.com/file/d/1LjM-ml5SqJUcgmg7GtfODTcTqVGirJgM")
    return model