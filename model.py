import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        pool: nn.MaxPool2d,
    ) -> None:
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            pool,
            DoubleConv(in_channels, out_channels, kernel_size, padding),
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, padding: int):
        super(Up, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.up_conv = DoubleConv(in_channels, out_channels, kernel_size, padding)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)

        target_dim = x1.size()[2]
        origin_dim = x2.size()[2]
        padding = (origin_dim - target_dim) // 2

        x2 = x2[:, :, padding : origin_dim - padding, padding : origin_dim - padding]
        x = torch.cat([x2, x1], dim=1)

        return self.up_conv(x)


class WireUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super(WireUNet, self).__init__()
        self.pool11 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.pool02 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input = DoubleConv(in_channels, 32, 3, 0)
        self.down1 = Down(32, 64, 3, 0, self.pool02)
        self.down2 = Down(64, 128, 3, 0, self.pool02)
        self.down3 = Down(128, 256, 3, 1, self.pool02)
        self.down4 = Down(256, 512, 7, 0, self.pool11)

        self.up1 = Up(512, 256, 3, 0)
        self.up2 = Up(256, 128, 3, 0)
        self.up3 = Up(128, 64, 3, 0)
        self.up4 = Up(64, 32, 3, 0)

        self.output = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.output(x)

        return x


from time import time


def test():
    start = time()
    batch_size = 20

    x = torch.randn((batch_size, 1, 252, 252), dtype=torch.float16, device=device)
    model = WireUNet(1, 3, [64, 128, 128, 128]).to(device)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(100):
            preds = model(x)
    print(preds.shape)
    print(x.shape)
    print(preds.dtype)
    print(f"Time elapsed: {time() - start} seconds")
    summary(model, input_size=(1, 252, 252), batch_size=1)


if __name__ == "__main__":
    test()
