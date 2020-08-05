import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet
import torch.nn.functional as F
import pdb

class CelebNet(nn.Module):
    def __init__(self, num_classes, num_features=0, dropout=0.6, set_pooling='max'):
        super(CelebNet, self).__init__()
        self.dropout = dropout
        self.num_class = num_classes
        self.num_features = num_features
        self.has_embedding = num_features > 0
        self.set_pooling = set_pooling

        img_base = resnet.resnet50(pretrained=True) # image branch
        mask_base = resnet.ResNet(resnet.Bottleneck, [1, 1, 1, 1], num_classes=1000, zero_init_residual=True)
        mask_base.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'),
                                  strict=False)

        # fixed_names = []
        # for name, module in img_base._modules.items():
        #     if name == "layer3":
        #         print("break at layer3...")
        #         break
        #     fixed_names.append(name)
        #     for param in module.parameters():
        #         param.requires_grad = False
        #
        # fixed_names = []
        # for name, module in mask_base._modules.items():
        #     if name == "layer3":
        #         print("break at layer3...")
        #         break
        #     fixed_names.append(name)
        #     for param in module.parameters():
        #         param.requires_grad = False

        img_modules = list(img_base.children())[:-2]
        self.img_base = nn.Sequential(*img_modules)

        mask_modules = list(mask_base.children())[:-2]
        self.mask_base = nn.Sequential(*mask_modules)

        num_ftrs = img_base.fc.in_features
        self.attenblock0 = CrossAttention(in_channels=num_ftrs, out_channels=num_ftrs//8)
        # self.attenblock0 = CrossAttentionShort(in_channels=num_ftrs, out_channels=num_ftrs)
        self.conv_layer1 = BasicResLayer(num_ftrs, num_ftrs // 2, stride=2)  # 1024
        self.mask_layer1 = MaskConvBlock(num_ftrs, num_ftrs // 2, padding=1)  # 1024
        self.glob_layer1 = BasicResLayer(num_ftrs, num_ftrs // 2, stride=2)
        self.attenblock1 = CrossAttention(in_channels=num_ftrs // 2, out_channels=num_ftrs // 8)
        # self.attenblock1 = CrossAttentionShort(in_channels=num_ftrs // 2, out_channels=num_ftrs//2)

        num_ftrs = num_ftrs // 2
        self.conv_layer2 = BasicResLayer(num_ftrs, num_ftrs // 2, stride=2)  # 512
        self.mask_layer2 = MaskConvBlock(num_ftrs, num_ftrs // 2, padding=1)  # 512
        self.glob_layer2 = BasicResLayer(num_ftrs, num_ftrs // 2, stride=2)
        self.attenblock2 = CrossAttention(in_channels=num_ftrs//2, out_channels=num_ftrs//8)
        # self.attenblock2 = CrossAttentionShort(in_channels=num_ftrs//2, out_channels=num_ftrs//2)
        num_ftrs = num_ftrs // 2

        if self.has_embedding:
            self.fc_img = nn.Linear(num_ftrs, self.num_features)
            self.img_feat_bn = nn.BatchNorm1d(self.num_features)
            self.fc_mask = nn.Linear(num_ftrs, self.num_features)
            self.mask_feat_bn = nn.BatchNorm1d(self.num_features)
            self.fc_glob = nn.Linear(num_ftrs, self.num_features)
            self.glob_feat_bn = nn.BatchNorm1d(self.num_features)

            nn.init.kaiming_normal_(self.fc_img.weight, mode='fan_out')
            nn.init.constant_(self.fc_img.bias, 0)
            nn.init.constant_(self.img_feat_bn.weight, 1)
            nn.init.constant_(self.img_feat_bn.bias, 0)

            nn.init.kaiming_normal_(self.fc_mask.weight, mode='fan_out')
            nn.init.constant_(self.fc_mask.bias, 0)
            nn.init.constant_(self.mask_feat_bn.weight, 1)
            nn.init.constant_(self.mask_feat_bn.bias, 0)

            nn.init.kaiming_normal_(self.fc_glob.weight, mode='fan_out')
            nn.init.constant_(self.fc_glob.bias, 0)
            nn.init.constant_(self.glob_feat_bn.weight, 1)
            nn.init.constant_(self.glob_feat_bn.bias, 0)
        else:
            self.num_features=num_ftrs

        # if self.dropout:
        #     self.concat_drop = nn.Dropout(self.dropout)
        # if self.num_class:
        #     self.concat_cls = nn.Linear(self.num_features*3, self.num_class)
        #     nn.init.normal_(self.concat_cls.weight, std=0.001)
        #     nn.init.constant_(self.concat_cls.bias, 0)

        if self.dropout > 0:
            self.conv_drop = nn.Dropout(self.dropout)
            self.mask_drop = nn.Dropout(self.dropout)
            self.glob_drop = nn.Dropout(self.dropout)
        if self.num_class > 0:
            self.conv_cls = nn.Linear(self.num_features, self.num_class)
            self.mask_cls = nn.Linear(self.num_features, self.num_class)
            self.glob_cls = nn.Linear(self.num_features, self.num_class)

            nn.init.normal_(self.conv_cls.weight, std=0.001)
            nn.init.constant_(self.conv_cls.bias, 0)
            nn.init.normal_(self.mask_cls.weight, std=0.001)
            nn.init.constant_(self.mask_cls.bias, 0)
            nn.init.normal_(self.glob_cls.weight, std=0.001)
            nn.init.constant_(self.glob_cls.bias, 0)

    def forward(self, vids, masks, out_feature=False):
        if len(vids.shape) == 5:
            n_id, n_fr, c, h, w = vids.size()
            n_spl = n_id
        else:
            n_id, n_seq, n_fr, c, h, w = vids.size()
            n_spl = n_id * n_seq

        x = vids.view(-1, c, h, w)

        if len(masks.shape) == 4:
            m = masks.view(-1, h, w).unsqueeze(1).repeat(1, 3, 1, 1)
        elif len(masks.shape) == 5 and masks.shape[0] == 1:
            m = masks.view(-1, h, w).unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            m = masks.view(-1, 1, h, w).repeat(1, 3, 1, 1)
        # pdb.set_trace()
        x = self.img_base(x)
        m = self.mask_base(m)
        g = vid_max_pooling(self.attenblock0(x, m), n_spl, n_fr)

        x = self.conv_layer1(x)
        m = self.mask_layer1(m)
        g = self.glob_layer1(g) + vid_max_pooling(self.attenblock1(x, m), n_spl, n_fr)

        x = self.conv_layer2(x)
        m = self.mask_layer2(m)
        g = self.glob_layer2(g) + vid_max_pooling(self.attenblock2(x, m), n_spl, n_fr)

        x = F.relu(x)
        m = F.relu(m)
        g = F.relu(g)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        m = F.adaptive_avg_pool2d(m, (1, 1)).view(m.size(0), -1)
        g = F.adaptive_avg_pool2d(g, (1, 1)).view(g.size(0), -1)

        x = self.fc_img(x)
        x = self.img_feat_bn(x)
        m = self.fc_mask(m)
        m = self.mask_feat_bn(m)
        g = self.fc_glob(g)
        g = self.glob_feat_bn(g)
        # pdb.set_trace()

        x = x.view(n_spl, n_fr, x.shape[1])
        m = m.view(n_spl, n_fr, m.shape[1])

        if self.set_pooling == 'max':
            x = torch.max(x, dim=1)[0]
            m = torch.max(m, dim=1)[0]

        elif self.set_pooling == 'mean':
            x = torch.mean(x, dim=1)
            m = torch.mean(m, dim=1)
        else:
            x = x.view(n_spl * n_fr, -1)
            m = m.view(n_spl * n_fr, -1)

        # out = g
        out = torch.cat((x, m, g), 1)
        # out = torch.cat((x, g), 1)
        # out = x + g
        em = F.normalize(out)
        if out_feature:
            return em

        # if self.dropout:
        #     out = self.concat_drop(out)
        # if self.concat_cls:
        #     out = self.concat_cls(out)
        # return out, em


        if self.dropout:
            x = self.conv_drop(x)
            m = self.mask_drop(m)
            g = self.glob_drop(g)
        if self.num_class > 0:
            x = self.conv_cls(x)
            m = self.mask_cls(m)
            g = self.glob_cls(g)
        return x, m, g, em


class BasicResLayer(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(BasicResLayer, self).__init__()
        downsample = DownSample(inplanes, planes, stride)
        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        layers.append(BasicBlock(planes, planes))
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DownSample(nn.Module):
    def __init__(self, inplanes, planes,stride):
        super(DownSample, self).__init__()
        self.downsample = nn.Sequential(
            resnet.conv1x1(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        out = self.downsample(x)
        return out


class MaskConvBlock(nn.Module):
    def __init__(self, inplanes, planes, ksize=3, padding=0):
        super(MaskConvBlock, self).__init__()
        block = []
        block.append(ConvPoolBlock(inplanes, planes, ksize, padding=padding, pooling=False))
        block.append(ConvPoolBlock(planes, planes, ksize, padding=padding, pooling=True))
        self.mask_block = nn.Sequential(*block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mask_block(x)
        return x


class ConvPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, pooling=False):
        super(ConvPoolBlock, self).__init__()
        self.pooling = pooling
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if self.pooling:
            self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        if self.pooling:
            x = self.maxpool2d(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttention, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.m = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.v = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        # pdb.set_trace()
        N, C, H, W = x.shape
        # if len(torch.squeeze(mask).shape) == 3:
        #     mask = mask.unsqueeze(dim=1).repeat(1, C, 1, 1)
        proj_f = self.f(x).view(N, -1, H * W).permute(0, 2, 1)  # N * HW * C
        proj_g = self.g(x).view(N, -1, H * W)  # N * C * HW
        energy = torch.bmm(proj_f, proj_g)  # N * HW * HW
        proj_h = self.h(mask).view(N, -1, H * W)

        attention = F.softmax(energy, dim=-1)
        mask_energy = torch.bmm(proj_h, attention.permute(0, 2, 1))  # N * C * HW
        mask_atten = F.softmax(mask_energy, dim=-1)

        proj_m = self.m(x)

        out =proj_m * mask_atten.view(N, -1, H, W)
        out = self.v(out)
        out = self.gamma * out + x
        return out


class CrossAttentionShort(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttentionShort, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.m = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        # pdb.set_trace()
        N, C, H, W = x.shape
        # if len(torch.squeeze(mask).shape) == 3:
        #     mask = mask.unsqueeze(dim=1).repeat(1, C, 1, 1)
        proj_f = self.f(x).view(N, -1, H * W).permute(0, 2, 1)  # N * HW * C
        proj_g = self.g(x).view(N, -1, H * W)  # N * C * HW
        energy = torch.bmm(proj_f, proj_g)  # N * HW * HW
        proj_h = self.h(mask).view(N, -1, H * W)

        attention = F.softmax(energy, dim=-1)
        mask_energy = torch.bmm(proj_h, attention.permute(0, 2, 1))  # N * C * HW
        mask_atten = F.softmax(mask_energy, dim=-1)

        proj_m = self.m(x)

        out =proj_m * mask_atten.view(N, C, H, W)
        out = self.gamma * out + x
        return out


class SpaceTimeAtten(nn.Module):
    def __init__(self, in_channels, out_channels, bn_layer=True):
        super(SpaceTimeAtten, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.h = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.m = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if bn_layer:
            self.W_z = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1),
                                     nn.BatchNorm3d(in_channels))
        else:
            self.W_z = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        """
        args:
            x: (N, T, C, H, W)
        """
        N, C, T, H, W = x.shape
        proj_f = self.h(x).view(N, -1, T*H*W).permute(0, 2, 1) # N* （T*H*W）* C
        proj_g = self.g(x).view(N, -1, T*H*W)
        energy = torch.bmm(proj_f, proj_g)  # N * THW * THW
        proj_h = self.h(mask).view(N, -1, T * H * W)

        attention = F.softmax(energy, dim=-1)
        mask_energy = torch.bmm(proj_h, attention.permute(0, 2, 1))  # N * C * THW
        mask_atten = F.softmax(mask_energy, dim=-1)

        proj_m = self.m(x)

        out = proj_m * mask_atten.view(N, -1, T, H, W)

        # out = torch.bmm(proj_m, mask_atten.permute(0, 2, 1))
        out = out.contiguous()
        # out = out.view(N, -1, T, H, W)
        W_y = self.W_z(x)
        out = self.gamma * out + W_y
        return out


def vid_max_pooling(x, n_id, n_fr):
    if len(x.shape) == 4:
        x = x.view(n_id, n_fr, x.shape[1], x.shape[2], x.shape[3])
    out = torch.max(x, 1)[0]
    return out


