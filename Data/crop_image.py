import os
from PIL import Image

for image_name in os.listdir("./image/"):

    im = Image.open("./image/" + image_name)
    imResize = im.crop((32, 32, 32, 32))
    os.remove("./image/" + image_name)
    imResize.save("./image/" + image_name, 'PNG', quality=120)