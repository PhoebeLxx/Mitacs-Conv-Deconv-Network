import torch
import torch.nn as nn


def flatten_indices(indices, stride_num):
    indices = indices[:, :, 0, :, :]
    bound = indices.size()[2] * indices.size()[3] * (stride_num ** 2)
    return (check_index(indices, bound)).long()


def check_index(indices, bound):
    indices = indices.int() - ((indices >= bound).int() * bound)
    if (indices >= bound).byte().any():
        indices = check_index(indices, bound)
    return indices


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Convolution 1
        self.conv1 = nn.Conv3d(1, 40, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.conv1.weight.data)

        # Convolution 2
        self.conv2 = nn.Conv3d(40, 60, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.conv2.weight.data)

        # Convolution 3
        self.conv3 = nn.Conv3d(60, 80, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.conv3.weight.data)

        # Convolution 4
        self.conv4 = nn.Conv3d(80, 100, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.conv4.weight.data)

        # Convolution 5
        self.conv5 = nn.Conv3d(100, 128, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.conv5.weight.data)

        # Fully Connected / Dense Layer 1
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 128 * 4 * 4)
        self.drop = nn.Dropout(0.2)

        # De Convolution 1
        self.deconv0 = torch.nn.ConvTranspose2d(128, 100, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.deconv0.weight.data)

        # De Convolution 1
        self.deconv1 = torch.nn.ConvTranspose2d(100, 80, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.deconv1.weight.data)

        # De Convolution 2
        self.deconv2 = torch.nn.ConvTranspose2d(80, 60, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.deconv2.weight.data)

        # De Convolution 3
        self.deconv3 = torch.nn.ConvTranspose2d(60, 40, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.deconv3.weight.data)

        # De Convolution 4
        self.deconv4 = torch.nn.ConvTranspose2d(40, 3, kernel_size=2, stride=2)
        nn.init.xavier_uniform_(self.deconv4.weight.data)

    def forward(self, data):
        out = torch.relu(self.conv1(data))
        out = torch.relu(self.conv2(out))
        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))
        out = torch.relu(self.conv5(out))

        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = out.view(2, 128, 4, 4)
        out = self.drop(out)

        out = torch.relu(self.deconv0(out))
        out = torch.relu(self.deconv1(out))
        out = torch.relu(self.deconv2(out))
        out = torch.relu(self.deconv3(out))
        out = self.deconv4(out)

        return out
