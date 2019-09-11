import torch
import torch.nn as nn
import torch.nn.functional as func


def flatten_indices(indices):
    indices = indices[:, :, 0, :, :]
    bound = indices.size()[2] * indices.size()[3] * 4
    return (indices.int() - ((indices >= bound).int() * bound)).long()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Convolution 1
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.prelu1 = nn.PReLU()
        self.max1 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                 stride=(2, 2, 2),
                                 return_indices=True)

        # Convolution 2
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.prelu2 = nn.PReLU()
        self.max2 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                 stride=(2, 2, 2),
                                 return_indices=True)

        # Convolution 3
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.prelu3 = nn.PReLU()
        self.max3 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                 stride=(2, 2, 2),
                                 return_indices=True)

        # Convolution 4
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.prelu4 = nn.PReLU()
        self.max4 = nn.MaxPool3d(kernel_size=(2, 2, 2),
                                 stride=(2, 2, 2),
                                 return_indices=True)

        # Fully Connected / Dense Layer 1
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 128 * 4 * 4)
        self.drop = nn.Dropout(0.2)

        # De Convolution 1
        self.maxUn1 = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.prelu5 = nn.PReLU()

        # De Convolution 2
        self.maxUn2 = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.prelu6 = nn.PReLU()

        # De Convolution 3
        self.maxUn3 = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.prelu7 = nn.PReLU()

        # De Convolution 4
        self.maxUn4 = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv4 = torch.nn.ConvTranspose2d(16, 3, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv4.weight)

    def forward(self, data):
        out = self.prelu1(self.conv1(data))
        out, indices1 = self.max1(out)

        out = self.prelu2(self.conv2(out))
        out, indices2 = self.max2(out)

        out = self.prelu3(self.conv3(out))
        out, indices3 = self.max3(out)

        out = self.prelu4(self.conv4(out))
        out, indices4 = self.max4(out)

        out = out.view(out.size(0), -1)
        out = func.leaky_relu(self.fc1(out))
        out = out.view(10, 128, 4, 4)
        out = self.drop(out)

        indices1 = flatten_indices(indices1)
        indices2 = flatten_indices(indices2)
        indices3 = flatten_indices(indices3)
        indices4 = flatten_indices(indices4)

        out = self.maxUn1(out, indices4)
        out = self.prelu5(self.deconv1(out))

        out = self.maxUn2(out, indices3)
        out = self.prelu6(self.deconv2(out))

        out = self.maxUn3(out, indices2)
        out = self.prelu7(self.deconv3(out))

        out = self.maxUn4(out, indices1)
        out = self.deconv4(out)

        return out
