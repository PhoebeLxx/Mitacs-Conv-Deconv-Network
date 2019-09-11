import os
import numpy as np
from matplotlib import pyplot as plt
import visdom_plot
from PIL import Image
import gc

from sklearn.model_selection import KFold
import torch
import torch.nn.functional as func
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
import Tool.radam as radam
import time

import Network as Network

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


def read_data(cross=True):
    l_data = []
    l_label = []
    r_name = []

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    for filename in os.listdir(raw_dir):
        r_name.append(filename)
        iso = int(filename.split('_')[1].strip('.raw'))

        raw = np.fromfile(raw_dir + filename, dtype='uint8')
        raw_p = (raw.astype('float') - iso / 2) / 255
        # raw_img = torch.tensor(raw_p, dtype=torch.float16).reshape([1, 128, 128, 128])
        raw_img = torch.tensor(raw_p, dtype=torch.float16).reshape([1, 64, 64, 64])
        l_data.append(raw_img.detach())
        del raw_img
        gc.collect()

        if os.path.isfile(img_dir + filename.replace('.raw', '.png')):
            item = filename.replace('.raw', '.png')
            im = transform(Image.open(img_dir + item))
            l_label.append(im)

        if len(l_data) == 1000:
            print("{} volumes read in the list.".format(len(l_data)))
            break

    if cross:
        # return l_data[:1600], l_label[:1600], l_data[1600:1800], l_label[1600:1800], r_name[1600:1800]
        return l_data[:800], l_label[:800], l_data[800:1000], l_label[800:1000], r_name[800:1000]
    else:
        tensor_data = torch.stack(list_data)
        tensor_label = torch.stack(list_label)
        return data.TensorDataset(tensor_data, tensor_label), r_name


def visualize_output(batch):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(25, 25))
    for ax in axes.flatten():
        ax.axis('off')

    for i, img in enumerate(batch):
        axes[i // 4, i % 4].imshow(img.permute(1, 2, 0))


def draw_image(prediction, image):
    visualize_output(prediction.detach())
    visualize_output(image.detach())


def generate_sample(train_idx, valid_idx):
    train_v = torch.stack(list(list_data[i] for i in train_idx))
    train_i = torch.stack(list(list_label[i] for i in train_idx))

    valid_v = torch.stack(list(list_data[i] for i in valid_idx))
    valid_i = torch.stack(list(list_label[i] for i in valid_idx))

    test_v = torch.stack(test_data)
    test_i = torch.stack(test_label)

    print()
    print()
    print('-' * 15, " New Fold %s" % fold_num, '-' * 15)

    train = data.TensorDataset(train_v, train_i)
    valid = data.TensorDataset(valid_v, valid_i)
    test = data.TensorDataset(test_v, test_i)

    t_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    v_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)
    te_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return t_loader, v_loader, te_loader


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
        if os.path.exists(result_dir + raw_name[index].replace('.raw', '.png')):
            os.remove(result_dir + raw_name[index].replace('.raw', '.png'))
        # print()
        # print(name[index].replace('.raw', '.png'))
        # print(tensor[index][0][32])
        # print(output[0][32])
        save_image(tensor[index], result_dir + raw_name[index].replace('.raw', '.png'))


def train_model():
    scheduler.step(epoch)
    print(scheduler.get_lr())
    total_loss = 0

    for volume, image in train_loader:
        print(volume.shape)
        torch.cuda.empty_cache()

        # with torch.no_grad():
        #     image.to(device)
        #     volume.to(device)
        optimizer.zero_grad()
        prediction = network(volume.float())
        loss = func.mse_loss(prediction, image)
        # loss = F.smooth_l1_loss(prediction, image)

        loss.backward()
        optimizer.step()
        total_loss += loss.detach()

    loss = total_loss.item() / len(train_loader)

    return loss


def evaluate_model():
    scheduler.step(epoch)
    total_loss = 0

    with torch.no_grad():
        for volume, image in valid_loader:
            torch.cuda.empty_cache()
            prediction = network(volume.float())
            loss_valid = func.mse_loss(prediction, image)
            # loss_valid = F.smooth_l1_loss(prediction, image)
            total_loss += loss_valid.detach()

        loss = total_loss.item() / len(valid_loader)

    return loss


def predict_model():
    result = torch.Tensor()
    with torch.no_grad():
        for volume, image in test_loader:
            prediction = network(volume.float())
            result = torch.cat((result, prediction), 0)

    return result


data_dir = './Data/'
raw_dir = os.path.join(data_dir, 'raw64/')
img_dir = os.path.join(data_dir, 'image64/')
result_dir = os.path.join(data_dir, 'result/')
network_path = './Data/network.pkl'

list_data, list_label, test_data, test_label, raw_name = read_data(cross=True)

num_epochs = 1
batch_size = 10
learning_rate = 0.0001

# network = Network.Network()
# network.load_state_dict(torch.load(network_path))

skf = KFold(n_splits=8, shuffle=True, random_state=0)

fold_train = {}
fold_valid = {}

fold_num = 1
for t_idx, v_idx in skf.split(list_data, list_label):
    train_loss = []
    valid_loss = []
    plotter = visdom_plot.VisdomLinePlotter(env_name='Volume to Image: 8 fold, 30 epochs each')

    train_loader, valid_loader, test_loader = generate_sample(t_idx, v_idx)

    since = time.time()

    with torch.cuda.device(0):
        network = Network.Network()
        optimizer = radam.RAdam(network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            network.train()
            t_loss = train_model()
            print("     Train Loss:", t_loss)
            train_loss.append(t_loss)
            plotter.plot('loss', 'train', 'Fold %s Loss' % fold_num, epoch, t_loss)

            network.eval()
            v_loss = evaluate_model()
            print("     Valid Loss:", v_loss)
            valid_loss.append(v_loss)
            plotter.plot('loss', 'valid', 'Fold %s Loss' % fold_num, epoch, v_loss)

            print('-' * 40)

            if epoch % 5 == 0:
                torch.save(network.state_dict(), network_path)
                pred = predict_model()
                save_output(pred)

        fold_train["Fold %s" % fold_num] = train_loss
        fold_valid["Fold %s" % fold_num] = valid_loss

        print('-' * 40)
        time_elapsed = time.time() - since
        print('{} Epoch Time Total: {:.0f}m {:.0f}s'.format(num_epochs, time_elapsed // 60, time_elapsed % 60))
        print('{} Epoch Train Loss Average: {}'.format(num_epochs, calculate_average(train_loss)))
        print('{} Epoch Valid Loss Average: {}'.format(num_epochs, calculate_average(valid_loss)))

    fold_num += 1
    break

print(fold_train)
print(fold_valid)
