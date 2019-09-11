import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pandas as pd

torch.set_printoptions(edgeitems=40)
desired_width = 320
pd.set_option('display.width', desired_width)


def save_output(tensor, dir, name):
    if os.path.exists(dir + name.replace('.raw', '.png')):
        os.remove(dir + name.replace('.raw', '.png'))
    save_image(tensor, dir + name.replace('.raw', '.png'))


def read_data(raw_dir, img_dir):
    name = []
    difference = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform_t = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    for filename in os.listdir(raw_dir):
        if os.path.isfile(img_dir + filename.replace('.raw', '.png')):
            item = filename.replace('.raw', '.png')
            im = transform(Image.open(img_dir + item))
            im_t = transform_t(Image.open(img_dir + item))
            difference.append((im_t - im).float())
            # print(filename)
            # print(difference[difference != 1])
            # save_output(im, result_dir, filename)
        name.append(filename)
    return name, difference


def read_raw(start, end=7800):
    l_data = []
    l_label = []
    r_name = []

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for filename in os.listdir(raw_dir):
        r_name.append(filename)

        # if filename == "cube25_115.raw":
        #     print(r_name.index(filename))

        if 7700 <= r_name.index(filename) < 7800:
            print(filename)

        if start <= r_name.index(filename) < end:
            # iso = int(filename.split('_')[1].strip('.raw'))
            # raw = np.fromfile(raw_dir + filename, dtype='uint8')
            # raw_p = (raw.astype('float') - iso / 2) / 255
            # raw_img = torch.tensor(raw_p, dtype=torch.float16).reshape([1, 128, 128, 128])
            # print(filename)
            # print(torch.max(raw_img.to('cuda')))
            # print(torch.min(raw_img.to('cuda')))
            # l_data.append(raw_img.detach())
            # del raw_img
            # gc.collect()

            if os.path.isfile(result_dir + filename.replace('.raw', '.png')):
                item = filename.replace('.raw', '.png')
                im = transform(Image.open(result_dir + item))
                im_o = transform(Image.open(img_dir + item))

                if filename == 'ellipse2997_162.raw':
                    save_image(im, os.path.join(data_dir, '2997/') + str(1) + ".png")

                # if filename.replace('.raw', '.png') == 'cube6234_130.png':
                #     print(im_o[im_o != 0])
                #     print()
                #     print("result tensor")
                #     print(im[im != 0])
                #
                #     difference = (im_o - im)[(im_o - im) != 0]
                #     print(difference)
                #     print(len(im[im == 1]))

                # print(torch.max(im.to('cuda')))
                # print()
                # l_label.append(im)
        elif r_name.index(filename) < start:
            continue
        else:
            break

    return l_data, l_label, r_name[start:end]


data_dir = './data/'
raw_dir = os.path.join(data_dir, 'raw/')
img_dir = os.path.join(data_dir, 'image/')
result_dir = os.path.join(data_dir, 'result/')
network_path = './data/network.pkl'

# f_name, diff = read_data(raw_dir, img_dir)
# f_name = f_name[:10]
# diff = diff[:10]
#
# for i in range(len(f_name)):
#     print(f_name[i])
#     print(diff[i])
# np.savetxt('test.out', diff[0], delimiter=',')


l, z, r = read_raw(0)
