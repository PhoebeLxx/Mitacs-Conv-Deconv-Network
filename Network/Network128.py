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
        self.conv1 = nn.Conv3d(1, 25, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data)
        self.max1 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                 stride=(2, 2, 2),
                                 return_indices=True)

        # Convolution 2
        self.conv2 = nn.Conv3d(25, 50, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight.data)
        self.max2 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                 stride=(2, 2, 2),
                                 return_indices=True)

        # Convolution 3
        self.conv3 = nn.Conv3d(50, 80, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight.data)
        self.max3 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                 stride=(2, 2, 2),
                                 return_indices=True)

        # Convolution 4
        self.conv4 = nn.Conv3d(80, 100, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight.data)
        self.max4 = nn.MaxPool3d(kernel_size=(4, 4, 4),
                                 stride=(4, 4, 4),
                                 return_indices=True)

        # Fully Connected / Dense Layer 1
        self.fc1 = nn.Linear(100 * 4 * 4 * 4, 100 * 4 * 4)
        self.drop = nn.Dropout(0.2)

        # De Convolution 1
        self.maxUn1 = torch.nn.MaxUnpool2d(4, stride=4)
        self.deconv1 = torch.nn.ConvTranspose2d(100, 80, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight.data)

        # De Convolution 2
        self.maxUn2 = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = torch.nn.ConvTranspose2d(80, 50, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv2.weight.data)

        # De Convolution 3
        self.maxUn3 = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = torch.nn.ConvTranspose2d(50, 25, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv3.weight.data)

        # De Convolution 4
        self.maxUn4 = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv4 = torch.nn.ConvTranspose2d(25, 3, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv4.weight.data)

    def forward(self, data):
        out = torch.relu(self.conv1(data))
        out, indices1 = self.max1(out)

        out = torch.relu(self.conv2(out))
        out, indices2 = self.max2(out)

        out = torch.relu(self.conv3(out))
        out, indices3 = self.max3(out)

        out = torch.relu(self.conv4(out))
        out, indices4 = self.max4(out)

        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = out.view(2, 100, 4, 4)
        out = self.drop(out)

        indices1 = flatten_indices(indices1, 2)
        indices2 = flatten_indices(indices2, 2)
        indices3 = flatten_indices(indices3, 2)
        indices4 = flatten_indices(indices4, 4)

        out = self.maxUn1(out, indices4)
        out = torch.relu(self.deconv1(out))

        out = self.maxUn2(out, indices3)
        out = torch.relu(self.deconv2(out))

        out = self.maxUn3(out, indices2)
        out = torch.relu(self.deconv3(out))

        out = self.maxUn4(out, indices1)
        out = self.deconv4(out)

        return out
