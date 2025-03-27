import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        """
        U-Net 需要输入噪声 xt 和时间步 t
        """
        t = t[:, None, None, None].expand(x.shape)  # 扩展时间步维度
        x = torch.cat([x, t], dim=1)  # 将时间信息拼接到输入
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
