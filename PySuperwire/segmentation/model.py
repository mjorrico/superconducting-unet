import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchinfo import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DoubleConv, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Up, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.up_conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.up_conv(x)


class WireUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[64, 128, 256, 512]):
        super(WireUNet, self).__init__()

        self.contractive = nn.ModuleList()
        self.expansive = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2, 2)

        for f in features:
            self.contractive.append(DoubleConv(in_channels, f, 3))
            in_channels = f

        self.bottleneck = DoubleConv(features[-1], 2 * features[-1], 3)

        for f in reversed(features):
            self.expansive.append(Up(2 * f, f, 3))

        self.output = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        skip_values = []
        for doubleconv in self.contractive:
            x = doubleconv(x)
            skip_values.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)

        skip_values = skip_values[::-1]
        for i, combine in enumerate(self.expansive):
            x = combine(x, skip_values[i])

        x = self.output(x)

        return x


from time import time

def test():
    start = time()
    batch_size = 10

    x = torch.randn((batch_size, 1, 240, 240), dtype=torch.float16, device="cuda")
    model = WireUNet(1, 4, [16, 32, 64, 128]).to("cuda")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in range(1):
            preds = model(x)
    print(preds.shape)
    print(f"Time elapsed: {time() - start} seconds")
    summary(model, input_size=(10, 1, 240, 240))


if __name__ == "__main__":
    test()
