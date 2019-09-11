import os
import numpy.random as nprd
import random
from PIL import Image

shape_option = ['cube', 'sphere', 'prism', 'ellipse']
list = []

for i in range(5200, 8400):
    shape = random.choices(shape_option, weights=[0.25, 0.25, 0.25, 0.25])
    num_item = random.randint(1, 3)
    iso_dens = random.randint(96, 169)
    rotation = random.choices([0, 1], [0.35, 0.65])[0]
    smooth = random.randint(5, 10)

    min_x = round(nprd.normal(loc=0.18, scale=0.1, size=None), 3)
    min_y = round(nprd.normal(loc=0.18, scale=0.1, size=None), 3)
    min_z = round(nprd.normal(loc=0.18, scale=0.1, size=None), 3)
    max_x = round(nprd.normal(loc=0.45, scale=0.08, size=None), 3)
    max_y = round(nprd.normal(loc=0.45, scale=0.08, size=None), 3)
    max_z = round(nprd.normal(loc=0.45, scale=0.08, size=None), 3)

    list.append(shape[0])
    name = "./raw/" + shape[0] + str(i) + "_" + str(iso_dens) + ".raw"
    image_name = "./image/" + shape[0] + str(i) + "_" + str(iso_dens) + ".png"

    os.system(
        f"./volumegen_2 -d 128,128,128 -s "
        f"{shape[0]},{num_item},{min_x},{min_y},{min_z},{max_x},{max_y},{max_z},{iso_dens},{rotation} -o {name} -smooth {smooth}")

    os.system(f"python render.py {name} 128,128,128 {image_name} {float(iso_dens) / 2.}")

    im = Image.open(image_name)
    imResize = im.resize((128, 128), Image.ANTIALIAS).convert('RGB')
    os.remove(image_name)
    imResize.save(image_name, 'PNG', quality=90)

    print(image_name, "isovalue:" + str(iso_dens), "num_item:" + str(num_item),
          "rotation:" + str(rotation), "smooth:" + str(smooth))
    print()
    print(image_name, "min_x:" + str(min_x), "max_x:" + str(max_x))
    print(image_name, "min_y:" + str(min_y), "max_y:" + str(max_y))
    print(image_name, "min_z:" + str(min_z), "max_z:" + str(max_z))
    print()

print('sphere:', list.count('sphere'))
print('cube:', list.count('cube'))
print('prism:', list.count('prism'))
print('ellipse:', list.count('ellipse'))
