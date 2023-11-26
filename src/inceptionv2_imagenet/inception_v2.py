import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class conv_block(nn.Module): # Create a class for the convolutional block
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchNormalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.batchNormalization(out)
        out = self.activation(out)
        
        return out

class stem_block(nn.Module): # Create a class for the stem block
    def __init__(self, in_channels):
        super(stem_block, self).__init__()

        self.stem = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=3, stride=2, padding=0), # 3x3 conv
            conv_block(32, 32, kernel_size=3, stride=1, padding=0), # 3x3 conv
            conv_block(32, 32, kernel_size=3, stride=1, padding=1), # 3x3 conv
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 3x3 max pooling
            conv_block(32, 80, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(80, 192, kernel_size=1, stride=1, padding=0), # 1x1 conv
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # 3x3 max pooling
        )

    def forward(self, x):
        out = self.stem(x)

        return out

class inception_A_block(nn.Module): # Create a class for the inception A block which contain 4 branches
    def __init__(self, in_channels):
        super(inception_A_block, self).__init__()

        self.branch1 = nn.Sequential( 
            conv_block(in_channels, 48, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(48, 64, kernel_size=5, stride=1, padding=2), # 5x5 conv 
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 48, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(48, 64, kernel_size=3, stride=1, padding=1), # 3x3 conv
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1), # 3x3 avg pooling
            conv_block(in_channels, 64, kernel_size=1, stride=1, padding=0), # 1x1 conv
        )

        self.branch4 = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=1, stride=1, padding=0), # 1x1 conv
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        out = torch.cat([out1, out2, out3, out4], 1)

        return out
    
class inception_restnet_A_block(nn.Module): # Create a class for the inception A block which contain 3 branches and a conv
    def __init__(self, in_channels):
        super(inception_restnet_A_block, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(32, 48, kernel_size=3, stride=1, padding=0), # 3x3 conv
            conv_block(48, 64, kernel_size=3, stride=1, padding=1), # 3x3 conv
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(32, 32, kernel_size=3, stride=1, padding=1), # 3x3 conv
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, 32, kernel_size=1, stride=1, padding=0), # 1x1 conv
        )

        self.conv = conv_block(128, 320, kernel_size=1, stride=1, padding=0) # 1x1 conv

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        out = torch.cat([out1, out2, out3], 1)
        out = self.conv(out)

        out += x

        return out
    
class inception_resnet_b_block(nn.Module):
    def __init__(self, in_channels):
        super(inception_resnet_b_block, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, 128, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), # 1x7 conv
            conv_block(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), # 7x1 conv
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1, stride=1, padding=0), # 1x1 conv
        )

        self.conv = conv_block(384, 1088, kernel_size=1, stride=1, padding=0) # 1x1 conv

        def forward(self, x):
            out1 = self.branch1(x)
            out2 = self.branch2(x)

            out = torch.cat([out1, out2], 1)

            out = self.conv(out)
            out += x

            return out

class inception_resnet_c_block(nn.Module):
    def __init__(self, in_channels):
        super(inception_resnet_c_block, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), # 1x3 conv
            conv_block(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)), # 3x1 conv
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1, stride=1, padding=0), # 1x1 conv
        )

        self.conv = conv_block(448, 2080, kernel_size=1, stride=1, padding=0) # 1x1 conv

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        out = torch.cat([out1, out2], 1)

        out = self.conv(out)
        out += x

        return out

class reduction_A_block(nn.Module): # Create a class for the reduction A block which contain 3 branches
    def __init__(self, in_channels):
        super(reduction_A_block, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(256, 256, kernel_size=3, stride=1, padding=1), # 3x3 conv
            conv_block(256, 384, kernel_size=3, stride=2, padding=0), # 3x3 conv
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, 384, kernel_size=3, stride=2, padding=0), # 3x3 conv
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # 3x3 max pooling
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        out = torch.cat([out1, out2, out3], 1)

        return out

class reduction_B_block(nn.Module):
    def __init__(self, in_channels):
        super(reduction_B_block, self).__init__()
    
        self.branch1 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(256, 288, kernel_size=3, stride=1, padding=1), # 3x3 conv
            conv_block(288, 320, kernel_size=3, stride=2, padding=0), # 3x3 conv
        )
        
        self.branch2 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(256, 384, kernel_size=3, stride=2, padding=0), # 3x3 conv
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, 256, kernel_size=1, stride=1, padding=0), # 1x1 conv
            conv_block(256, 288, kernel_size=3, stride=2, padding=0), # 3x3 conv
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # 3x3 max pooling
        )

        def forward(self, x):
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out3 = self.branch3(x)
            out4 = self.branch4(x)

            out = torch.cat([out1, out2, out3, out4], 1)

            return out
        

class inception_v2(nn.Module): # Create a class for the inception v2 model which contain 9 blocks (stem, 5 inception blocks (inception_A, 10 inception ResNet_A, 20 inception ResNet_B, 10 inception ResNet_C), 2 reduction blocks (A and B), 1 conv block, 1 global avg pooling, 2 fully connected layers)
    def __init__(self):
        super(inception_v2, self).__init__()

        self.stem = stem_block(3)

        self.inceptionBlock = inception_A_block(192)

        self.resnetABlock = inception_restnet_A_block(320)

        self.reductionABlock = reduction_A_block(320)

        self.resnetBBlock = inception_resnet_b_block(1088)

        self.reductionBBlock = reduction_B_block(1088)

        self.resnetCBlock = inception_resnet_c_block(2080)

        self.conv = conv_block(2080, 1536, kernel_size=1, stride=1, padding=0) # 1x1 conv
        self.globalAvgPool = conv_block(1536, 1536, kernel_size=8, stride=1, padding=0) # 8x8 avg pooling

        self.fc1 = nn.Linear(1536, 1536) 
        self.fc2 = nn.Linear(1536, 1000)

        def forward(self,x):
            out = self.stem(x)
            
            out = self.inceptionBlock(out)

            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)
            out = self.resnetABlock(out)

            out = self.reductionABlock(out)

            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)
            out = self.resnetBBlock(out)

            out = self.reductionBBlock(out)

            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)
            out = self.resnetCBlock(out)

            out = self.conv(out)

            out = self.globalAvgPool(out)

            out = out.reshape(out.size(0), -1)

            out = self.fc1(out)
            out = nn.ReLU

            out = self.fc2(out)
            out = nn.Softmax(out)

            return out