from torch_geometric.nn import voxel_grid, max_pool, SplineConv, max_pool_x
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net_n_caltech101(nn.Module):
    def __init__(self, n_class=101):
        super(Net_n_caltech101, self).__init__()
        self.conv0 = SplineConv(1, 64, dim=2, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.conv1 = ResidualBlock(64, 128)
        self.conv2 = ResidualBlock(128, 256)
        self.conv3 = ResidualBlock(256, 512)

        self.fc1 = torch.nn.Linear(512*16, 1024)
        self.bn = torch.nn.BatchNorm1d(1024)
        self.drop_out = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(1024, n_class)

    def forward(self, data):
        data.x = F.elu(self.bn1(self.conv0(data.x, data.edge_index, data.edge_attr)))
        cluster = voxel_grid(data.pos, data.batch, size=[4,3])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))


        data = self.conv1(data)
        cluster = voxel_grid(data.pos, data.batch, size=[16,12])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.conv2(data)
        cluster = voxel_grid(data.pos, data.batch, size=[30,23])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.conv3(data)
        cluster = voxel_grid(data.pos, data.batch, size=[60,45])
        x = max_pool_x(cluster, data.x, data.batch, size=16)
        # x = max_pool_x(cluster, data.x, data.batch)

        x = x[0].view(-1, self.fc1.weight.size(1))
        x = self.fc1(x)
        x = F.elu(x)
        x = self.bn(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class Net_n_mnist(nn.Module):
    def __init__(self, n_class=10):
        super(Net_n_mnist, self).__init__()

        self.conv1 = ResidualBlock(1, 32)

        self.conv2 = ResidualBlock(32, 64)

        self.conv3 = ResidualBlock(64, 128)

        self.fc1 = torch.nn.Linear(128*25, 128)
        self.bn = torch.nn.BatchNorm1d(128)
        self.drop_out = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128, n_class)

    def forward(self, data):
        data = self.conv1(data)
        cluster = voxel_grid(data.pos, data.batch, size=2)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.conv2(data)
        cluster = voxel_grid(data.pos, data.batch, size=4)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.conv3(data)
        cluster = voxel_grid(data.pos, data.batch, size=7)
        x = max_pool_x(cluster, data.x, data.batch, size=25)
        # x = max_pool_x(cluster, data.x, data.batch)

        x = x[0].view(-1, self.fc1.weight.size(1))
        x = self.fc1(x)
        x = F.elu(x)
        x = self.bn(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = SplineConv(in_channel, out_channel, dim=2, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = SplineConv(out_channel, out_channel, dim=2, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)

        self.shortcut_conv = SplineConv(in_channel, out_channel, dim=2, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)

    def forward(self, data):

        data.x = F.elu(
            self.left_bn2(
                self.left_conv2(
                    F.elu(
                        self.left_bn1(
                            self.left_conv1(data.x, data.edge_index, data.edge_attr)
                        )
                    ),
                    data.edge_index,
                    data.edge_attr,
                )
            )
            + self.shortcut_bn(
                self.shortcut_conv(data.x, data.edge_index, data.edge_attr)
            )
        )

        return data


if __name__ == '__main__':
    pass
    # net = Plain_3D(20)
    # net = Residual_3D(20)
    # # print(net.named_modules)
    # summary(net, input_size=(128, 8, 32, 32), batch_size=1, device='cuda')
