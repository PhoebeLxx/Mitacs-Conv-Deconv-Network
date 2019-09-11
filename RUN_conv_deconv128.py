import os
import numpy as np
from matplotlib import pyplot as plt
import visdom_plot
from PIL import Image
import gc

import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
import time
import Tool.pytorch_ssim as ssim
import Tool.radam as radam

import Network.Network128NoPool as Network

torch.set_printoptions(linewidth=30)
torch.set_grad_enabled(True)
torch.set_printoptions(edgeitems=20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_average(num):
    return sum(num) / len(num)


def check_tensor():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            del obj
            torch.cuda.empty_cache()


def read_data(start, end):
    l_data = []
    l_label = []
    r_name = []

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for filename in os.listdir(raw_dir):
        r_name.append(filename)

        if start <= r_name.index(filename) < end:
            iso = int(filename.split('_')[1].strip('.raw'))
            raw = np.fromfile(raw_dir + filename, dtype='uint8')
            raw_p = (raw.astype('float') - iso / 2) / 255 * 2
            raw_img = torch.tensor(raw_p, dtype=torch.float16).reshape([1, 128, 128, 128]).transpose(2, 3)
            l_data.append(raw_img.detach())
            del raw_img
            gc.collect()

            if os.path.isfile(img_dir + filename.replace('.raw', '.png')):
                item = filename.replace('.raw', '.png')
                im = transform(Image.open(img_dir + item))
                l_label.append(im)
        elif r_name.index(filename) < start:
            continue
        else:
            break

    return l_data, l_label, r_name[start:end]


def visualize_output(batch):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(25, 25))
    for ax in axes.flatten():
        ax.axis('off')

    for i, img in enumerate(batch):
        axes[i // 4, i % 4].imshow(img.permute(1, 2, 0))


def draw_image(prediction, image):
    visualize_output(prediction.detach())
    visualize_output(image.detach())


def generate_sample(rate):
    valid_start = int(len(list_data) - int(len(list_data) / (rate * 100)) * 100)

    global raw_name
    raw_name = raw_name[valid_start:]

    train_v = torch.stack(list_data[:valid_start])
    train_i = torch.stack(list_label[:valid_start])

    valid_v = torch.stack(list_data[valid_start:])
    valid_i = torch.stack(list_label[valid_start:])

    print()
    print('-' * 15, 'Epoch {}/{} -- New Group {}'.format(epoch, num_epochs, g), '-' * 15)
    print('    Training: {}, {}'.format(train_v.shape, train_i.shape))
    print('    Validation: {}, {}'.format(valid_v.shape, valid_i.shape))

    train = data.TensorDataset(train_v, train_i)
    valid = data.TensorDataset(valid_v, valid_i)

    t_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)

    return t_loader, v_loader


def cross_validation_split(data_set,
                           sample_size,
                           val_split,
                           shuffle=True):
    random_seed = 42

    indices = list(range(sample_size))
    split = int(np.floor(val_split * sample_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, valid_indices = indices[split:], indices[:split]

    train_sampler = data.SubsetRandomSampler(train_indices)
    valid_sampler = data.SubsetRandomSampler(valid_indices)

    t_loader = data.DataLoader(data_set,
                               batch_size=batch_size,
                               sampler=train_sampler)
    v_loader = data.DataLoader(data_set,
                               batch_size=batch_size,
                               sampler=valid_sampler)

    return t_loader, v_loader


def save_output(tensor):
    for index in range(tensor.size()[0]):
        if raw_name[index] == 'cube25_115.raw':
            save_image(tensor[index], result_dir + str(epoch) + raw_name[index].replace('.raw', '.png'))
        elif raw_name[index] == 'prism2345_145.raw':
            save_image(tensor[index], result_dir + str(epoch) + raw_name[index].replace('.raw', '.png'))
        elif raw_name[index] == 'sphere2870_121.raw':
            save_image(tensor[index], result_dir + str(epoch) + raw_name[index].replace('.raw', '.png'))
        elif raw_name[index] == 'ellipse4313_165.raw':
            save_image(tensor[index], result_dir + str(epoch) + raw_name[index].replace('.raw', '.png'))
        else:
            if os.path.exists(result_dir + raw_name[index].replace('.raw', '.png')):
                os.remove(result_dir + raw_name[index].replace('.raw', '.png'))
            # img = transforms.ToPILImage()(tensor[index])
            # img.save(result_dir + raw_name[index].replace('.raw', '.png'))
            save_image(tensor[index], result_dir + raw_name[index].replace('.raw', '.png'))


def train_model():
    scheduler.step(epoch)
    print("    learning rate: {}".format(scheduler.get_lr()[0]))
    total_loss = 0

    for volume, image in train_loader:
        torch.cuda.empty_cache()
        prediction = network(volume.float())

        optimizer.zero_grad()
        ssim_out = - ssim_loss(prediction, image)
        ssim_value = - ssim_out.item()
        ssim_out.backward()
        optimizer.step()

        total_loss += ssim_value

    final_loss = total_loss / len(train_loader)

    return final_loss


def evaluate_model():
    scheduler.step(epoch)
    total_loss = 0
    result = torch.Tensor()

    with torch.no_grad():
        for volume, image in valid_loader:
            torch.cuda.empty_cache()
            prediction = network(volume.float())

            ssim_out = - ssim_loss(prediction, image)
            ssim_value = - ssim_out.item()

            total_loss += ssim_value

            if output:
                result = torch.cat((result, prediction), 0)

        loss = total_loss / len(valid_loader)

        if output:
            save_output(result)

    return loss


data_dir = './Data/'
raw_dir = os.path.join(data_dir, 'raw/')
img_dir = os.path.join(data_dir, 'image/')
result_dir = os.path.join(data_dir, 'result/')
network_path = './Data/network.pkl'

data_total = 7200
group_num = 6
group_size = data_total / group_num

num_epochs = 36
batch_size = 2
learning_rate = 0.0001

# network = Network.Network()
# network.load_state_dict(torch.load(network_path))

fold_train = []
fold_valid = []

network = Network.Network()
optimizer = radam.RAdam(network.parameters(), lr=learning_rate)
ssim_loss = ssim.SSIM()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

since = time.time()
plotter = visdom_plot.VisdomLinePlotter(env_name='Conv Deconv')
torch.cuda.empty_cache()

for group in range(int(num_epochs / group_num) + 1):
    train_loss = []
    valid_loss = []

    for g in range(group_num):
        epoch = group * group_num + g + 1

        if epoch > num_epochs:
            break

        list_data, list_label, raw_name = read_data(int(g * group_size), int((g + 1) * group_size))
        train_loader, valid_loader = generate_sample(12)

        with torch.cuda.device(0):
            if epoch % 6 == 0:
                output = True
            else:
                output = False

            network.train()
            t_loss = train_model()
            print("    Train Loss:", t_loss)
            train_loss.append(t_loss)
            plotter.plot('loss', 'train', '%s Epochs Loss' % num_epochs, epoch, t_loss)

            network.eval()
            v_loss = evaluate_model()
            print("    Valid Loss:", v_loss)
            valid_loss.append(v_loss)
            plotter.plot('loss', 'valid', '%s Epochs Loss' % num_epochs, epoch, v_loss)

            fold_train.append(t_loss)
            fold_valid.append(v_loss)

            torch.save(network.state_dict(), network_path)

        del list_data, list_label, raw_name

time_elapsed = time.time() - since
print('{} Epoch Time Total: {:.0f}m {:.0f}s'.format(num_epochs, time_elapsed // 60, time_elapsed % 60))

print()
print(fold_train)
print(fold_valid)
