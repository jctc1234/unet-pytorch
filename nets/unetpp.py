import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        # self.up     = nn.ConvTranspose2d(kernel_size=4,stride=2,padding=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self,num, inputs1, inputs2,inputs3=None,inputs4=None,inputs5=None):
        if num==2:
            outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        elif num==3:
            outputs = torch.cat([inputs1,inputs2, self.up(inputs3)], 1)
        elif num == 4:
            outputs = torch.cat([inputs1,inputs2,inputs3 ,self.up(inputs4)], 1)
        elif num == 5:
            outputs = torch.cat([inputs1, inputs2, inputs3,inputs4 ,self.up(inputs5)], 1)
        else:print('cuo')
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg',batch_norm=False):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained,batch_norm=batch_norm)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)
        
        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True



class NestedUNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg', deep_supervision=False):
        super(NestedUNet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_1 = [192, 384, 768, 1024]
            in_2 = [256, 512, 1024]
            in_3 = [320, 640,]
            in_4 = [384]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]


        self.up_concat4 = unetUp(in_1[3], out_filters[3])
        self.up_concat3 = unetUp(in_1[2], out_filters[2])
        self.up_concat2 = unetUp(in_1[1], out_filters[1])
        self.up_concat1 = unetUp(in_1[0], out_filters[0])

        self.up_concat22 = unetUp(in_2[2], out_filters[2])
        self.up_concat12 = unetUp(in_2[1], out_filters[1])
        self.up_concat02 = unetUp(in_2[0], out_filters[0])

        self.up_concat13 = unetUp(in_3[1], out_filters[1])
        self.up_concat03 = unetUp(in_3[0], out_filters[0])

        self.up_concat04 = unetUp(in_4[0], out_filters[0])

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None


        self.backbone = backbone
        if self.deep_supervision:
            self.final1 = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)


    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat00, feat10, feat20, feat30, feat40] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat00, feat10, feat20, feat30, feat40] = self.resnet.forward(inputs)

        feat01= self.up_concat1(2,feat00, feat10)
        feat11 = self.up_concat2(2, feat10, feat20)
        feat21 = self.up_concat3(2, feat20, feat30)
        feat31 = self.up_concat4(2,feat30, feat40)

        feat02 = self.up_concat02(3, feat00, feat01,feat11)
        feat12 = self.up_concat12(3, feat10, feat11, feat21)
        feat22 = self.up_concat22(3, feat20, feat21, feat31)

        feat03 = self.up_concat03(4, feat00, feat01, feat02,feat12)
        feat13 = self.up_concat13(4, feat10, feat11, feat12, feat22)

        feat04 = self.up_concat04(5, feat00, feat01, feat02, feat03, feat13)

        if self.up_conv != None:
            feat04 = self.up_conv(feat04)


        if self.deep_supervision:
            output1 = self.final1(feat01)
            output2 = self.final2(feat02)
            output3 = self.final3(feat03)
            output4 = self.final4(feat04)
            return [output1, output2, output3, output4]

        else:
            output = self.final(feat04)
            return output



    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
