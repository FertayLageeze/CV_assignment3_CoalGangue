import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes  # 添加 num_classes 属性
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.pool(nn.ReLU()(self.conv1(x)))  # [B, 64, 240, 424]
        x2 = self.pool(nn.ReLU()(self.conv2(x1)))  # [B, 128, 120, 212]
        x3 = self.pool(nn.ReLU()(self.conv3(x2)))  # [B, 256, 60, 106]

        x = nn.ReLU()(self.upconv1(x3))  # [B, 128, 120, 212]
        x = self._crop_and_concat(x, x2)  # [B, 256, 120, 212]
        x = nn.ReLU()(self.conv4(x))  # [B, 128, 120, 212]

        x = nn.ReLU()(self.upconv2(x))  # [B, 64, 240, 424]
        x = self._crop_and_concat(x, x1)  # [B, 128, 240, 424]
        x = nn.ReLU()(self.conv5(x))  # [B, 64, 240, 424]

        x = self.conv6(x)  # [B, num_classes, 240, 424]
        x = nn.functional.interpolate(x, size=(480, 852), mode='bilinear', align_corners=True)  # 上采样到原始尺寸
        return x

    def _crop_and_concat(self, upsampled, bypass):
        # 裁剪特征图以匹配上采样后的尺寸
        _, _, h, w = upsampled.size()
        bypass = self._crop(bypass, h, w)
        return torch.cat((upsampled, bypass), dim=1)

    def _crop(self, tensor, target_h, target_w):
        _, _, h, w = tensor.size()
        delta_h = (h - target_h) // 2
        delta_w = (w - target_w) // 2
        return tensor[:, :, delta_h:delta_h + target_h, delta_w:delta_w + target_w]
