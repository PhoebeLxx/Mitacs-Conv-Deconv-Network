import os
from PIL import Image

for filename in os.listdir("./raw/"):
    name = "./raw/" + filename
    image_name = "./image/" + filename.replace(".raw", ".png")
    isovalue = int(filename.split('_')[1].strip('.raw'))
    os.system(f"python render.py {name} 128,128,128 {image_name} {float(isovalue) / 2.}")

    # im = Image.open(image_name)
    # imResize = im.resize((64, 64), Image.ANTIALIAS).convert('RGB')
    # os.remove(image_name)
    # imResize.save(image_name, 'PNG', quality=150)

    im = Image.open(image_name)
    imResize = im.resize((128, 128), Image.ANTIALIAS).convert('RGB')
    os.remove(image_name)
    imResize.save(image_name, 'PNG', quality=90)