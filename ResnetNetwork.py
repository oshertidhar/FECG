from ResnetBasics import *

class ResNetEncoder(nn.Module):

    def __init__(self, in_channels = 1,activation='leaky_relu'):
        super().__init__()

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch = nn.BatchNorm1d(16)
        self.relu = activation_func(activation)

        self.block1 = ResNetBasicBlockEncoder(16, 32, downsampling=2)
        self.block2 = ResNetBasicBlockEncoder(32, 32)

        self.block3 = ResNetBasicBlockEncoder(32, 64)
        self.block4 = ResNetBasicBlockEncoder(64, 64)

        self.block5 = ResNetBasicBlockEncoder(64, 128, downsampling=2)
        self.block6 = ResNetBasicBlockEncoder(128, 128)

        self.block7 = ResNetBasicBlockEncoder(128, 256)
        self.block8 = ResNetBasicBlockEncoder(256, 256)

        self.block9 = ResNetBasicBlockEncoder(256, 512, downsampling=2)
        self.block10 = ResNetBasicBlockEncoder(512, 512)

        self.block11 = ResNetBasicBlockEncoder(512, 1024)
        self.block12 = ResNetBasicBlockEncoder(1024, 1024)

        self.block13 = ResNetBasicBlockEncoder(1024, 2048, downsampling=2)
        self.block14 = ResNetBasicBlockEncoder(2048, 2048)


    def forward(self, x):


        x = self.conv1(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        #x = self.dropout1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        #x = self.dropout2(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        return x


class ResnetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.conv_out = nn.ConvTranspose1d(16, out_channels, kernel_size=3, stride=2, padding=2, output_padding=0,
                                           bias=False)
        self.batch_norm = nn.BatchNorm1d(16)

        self.block1 = ResNetBasicBlockDecoder(1024, 1024)
        self.block2 = ResNetBasicBlockDecoder(1024, 512, upsampling=2)

        self.block3 = ResNetBasicBlockDecoder(512, 512)
        self.block4 = ResNetBasicBlockDecoder(512, 256)

        self.block5 = ResNetBasicBlockDecoder(256, 256)
        self.block6 = ResNetBasicBlockDecoder(256, 128, upsampling=2)

        self.block7 = ResNetBasicBlockDecoder(128, 128)
        self.block8 = ResNetBasicBlockDecoder(128, 64, upsampling=2)

        self.block9 = ResNetBasicBlockDecoder(64, 64)
        self.block10 = ResNetBasicBlockDecoder(64, 32, upsampling=2)

        self.block11 = ResNetBasicBlockDecoder(32, 32)
        self.block12 = ResNetBasicBlockDecoder(32, 16)

    def forward(self, x):

        one_before_last = x
        x = self.block1(x)
        x = self.block2(x)[:, :, :-1]
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)[:, :, :-1]
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)[:, :, :-1]
        x = self.block11(x)
        x = self.block12(x)
        x = self.batch_norm(x)
        x = self.conv_out(x)[:, :, :-1]
        return x, one_before_last

class ResNet(nn.Module):
    def __init__(self, in_channels,*args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.Mdecoder = ResnetDecoder()
        self.Fdecoder = ResnetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        latent_half = x.size()[1] // 2
        m = x[:, :latent_half, :]
        f = x[:, latent_half:, :]
        m_out, one_before_last_m = self.Mdecoder(m)
        f_out, one_before_last_f = self.Fdecoder(f)
        return m_out, one_before_last_m, f_out, one_before_last_f