import torch.nn as nn
import torch
import torch.nn.functional as F

class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)

        fpp = spat_att * fp
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm 
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction_ratio = 1, kernel_cbam = 3, use_cbam = False):
        super(BasicBlock, self).__init__()
        self.use_cbam = use_cbam
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if self.use_cbam:
            self.cbam = CBAM(n_channels_in = self.expansion*planes, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #cbam
        if self.use_cbam:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, reduction_ratio = 1, kernel_cbam = 3, use_cbam = False):
        super(Bottleneck, self).__init__()
        self.use_cbam = use_cbam

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        if self.use_cbam:
            self.cbam = CBAM(n_channels_in = self.expansion*planes, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        #cbam
        if self.use_cbam:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, reduction_ratio = 1, kernel_cbam = 3, use_cbam_block= True, use_cbam_class = True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.reduction_ratio = reduction_ratio
        self.kernel_cbam = kernel_cbam
        self.use_cbam_block = use_cbam_block
        self.use_cbam_class = use_cbam_class
        
        self.conv1 = nn.Conv2d(256*3, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)



        if self.use_cbam_class:
            self.cbam = CBAM(n_channels_in = 256*block.expansion, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        
        out = out  + self.cbam(out)
        
        out = F.avg_pool2d(out, 1)
        # out = out.view(out.size(0), -1)

        # out = self.linear(out)

        return out




def ResNet18(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = True, use_cbam_class = True):
    # print(kernel_cbam)
    return ResNet(
                BasicBlock, 
                [2,2,2,2], 
                reduction_ratio= reduction_ratio,
                kernel_cbam = kernel_cbam,
                use_cbam_block= use_cbam_block,
                use_cbam_class = use_cbam_class
                )


# Our Model!!----------------------------------------------------------------------------------------------------------------------------------------


# class HeightCompression(nn.Module):
#     def __init__(self, model_cfg, **kwargs):
#         super().__init__()
#         self.net = ResNet18()
#         self.model_cfg = model_cfg
#         self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

#         fc1 = []

#         fc1.extend([
#             nn.Conv2d(256 * 2, 256, kernel_size = (1, 1), stride = (1, 1), padding = 0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#         ])
    
#         self.fc_layer1 = nn.Sequential(*fc1)


#         fc2 = []

#         fc2.extend([
#             nn.Conv2d(256 * 3, 256, kernel_size = (1, 1), stride = (1, 1), padding = 0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#         ])
    
#         self.fc_layer2 = nn.Sequential(*fc2)

#         fc3 = []

#         fc3.extend([
#             nn.Linear(3, 5), 
#             nn.ReLU(),
#             nn.Linear(5, 3),                           
#         ])
    
#         self.fc_layer3 = nn.Sequential(*fc3)

#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#         Returns:
#             batch_dict:
#                 spatial_features:

#         """
#         encoded_spconv_tensor1 = batch_dict['encoded_spconv_tensor3']
#         encoded_spconv_tensor2 = batch_dict['encoded_spconv_tensor4']

#         spatial_features1 = encoded_spconv_tensor1.dense()
#         spatial_features2 = encoded_spconv_tensor2.dense()  # torch.Size([batch, 128, 2, 200, 176])

#         N, C, D, H, W = spatial_features2.shape

#         spatial_features1 = spatial_features1.view(N, C * D, H, W) # torch.Size([batch, 256, 200, 176])
#         spatial_features2 = spatial_features2.view(N, C * D, H, W)

#         spatial_features3 = torch.stack([spatial_features1, spatial_features2], dim=1) # torch.Size([3, 3, 256, 200, 176])
#         spatial_features3 = spatial_features3.view(N, C * D * 2, H, W) # 3 is number of spatial_features  # torch.Size([3, 768, 200, 176])

#         spatial_features3 = self.fc_layer1(spatial_features3) 


#         # Convolution Method
#         # alpha = torch.tensor([1.0, 1.0, 1.0], device='cuda:0')
#         # alpha = self.fc_layer3(alpha)
#         spatial_features = torch.stack([spatial_features1, spatial_features2, spatial_features3], dim=1)
#         # spatial_features = torch.stack([spatial_features1, spatial_features2, spatial_features3], dim=1)

#         spatial_features = spatial_features.view(N, -1, H, W)

#         spatial_features = self.net(spatial_features)
    
#         batch_dict['spatial_features'] = spatial_features

#         batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

#         return batch_dict






# Original Model!!----------------------------------------------------------------------------------------------------------------------------------------


import torch.nn as nn
class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
