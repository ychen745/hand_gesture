import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import adjacency_handpose21

class GraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_keypoints,
                 temporal_kernel_size,
                 stride=1,
                 dropout=0):
        super().__init__()

        self.kernel_size = 3
        self.num_keypoints = num_keypoints
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.kernel_size,
            kernel_size=1,
            bias=True)

        padding = (temporal_kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (temporal_kernel_size, 1),
                (stride, 1),
                (padding, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x, A):
        x = self.conv(x)
        x = torch.transpose(x, 1, 3)
        xs = torch.chunk(x, self.kernel_size, dim=3)  # [1, 21, 45, 32]
        x = torch.cat(xs, dim=1)
        x = torch.nn.functional.conv2d(x, A.view(self.num_keypoints, self.num_keypoints*self.kernel_size, 1, 1))
        x = torch.transpose(x, 1, 3)
        return self.tcn(x)

class STGCN(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                num_keypoints,
                temporal_kernel_size,
                stride=1,
                dropout=0,
                residual=True):
        super().__init__()
        assert temporal_kernel_size % 2 == 1
        self.gcn = GraphConv(in_channels, out_channels, num_keypoints, temporal_kernel_size, stride=stride, dropout=dropout)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = x + res
        return self.relu(x)

class STGCNAction(nn.Module):
    def __init__(self, in_channels, num_keypoints, num_class, dropout=0.0, export=False):
        super().__init__()
        temporal_kernel_size = 9
        A = adjacency_handpose21()
        A = A.permute(2, 0, 1)
        self.register_buffer('A', A)
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)  # along temporal dimension
        self.st_gcn_networks = nn.ModuleList((
            STGCN(in_channels, 8, num_keypoints, temporal_kernel_size, 1, residual=False),
            STGCN(8, 16, num_keypoints, temporal_kernel_size, 1, dropout=dropout),
            STGCN(16, 32, num_keypoints, temporal_kernel_size, 1, dropout=dropout),
            STGCN(32, 64, num_keypoints, temporal_kernel_size, 2, dropout=dropout),
            STGCN(64, 64, num_keypoints, temporal_kernel_size, 1, dropout=dropout),
            STGCN(64, 128, num_keypoints, temporal_kernel_size, 2, dropout=dropout),
            STGCN(128, 256, num_keypoints, temporal_kernel_size, 1, dropout=dropout),
        ))

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        self.export = export

    def forward(self, x):
        # batch-normalize input along T dimension
        #  N, T, C, V - >  N, C, V, T
        if self.export:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 3, 1)

        N, C, V, T = x.size()
        x = x.view(N, V * C, T)
        x = x.view(N, C, V, T)
        x = x.transpose(2, 3)

        # forward
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        x = F.avg_pool2d(x, x.size()[2:])
        
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
    
        if self.export: 
            x = F.softmax(x, dim=1)  # only add this when export model (coreml wants softmaxed output)

        return x
