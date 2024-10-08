import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class VNet(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(VNet, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

        self.temperature = 0.1
        self.proto_m = 0.99
        self.low_dim = 128

        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(), nn.Linear(128, self.low_dim))
        self.register_buffer("prototypes", torch.zeros(n_outputs, self.low_dim))

    @torch.no_grad()
    def init_prototypes(self, features, labels):
        # features, labels are lists of tensors
        assert set(range(len(self.prototypes))) == set(labels.cpu().numpy()), \
            f'The missing labels are {set(range(len(self.prototypes))) - set(labels.cpu().numpy())}'
        for i in range(self.prototypes.shape[0]):
            self.prototypes[i] = features[labels == i].mean(dim=0)
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    @torch.no_grad()
    def update_prototypes(self, features, labels):
        avg_features = torch.zeros_like(self.prototypes).to(self.prototypes.device)
        for i in range(avg_features.shape[0]):
            avg_features[i] = features[labels == i].mean(dim=0)
        avg_features = F.normalize(avg_features, p=2, dim=1)
        for i in range(avg_features.shape[0]):
            if not torch.isnan(avg_features[i]).any():
                self.prototypes[i] = self.proto_m * self.prototypes[i] + (1 - self.proto_m) * avg_features[i]
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    def forward(self, x, forward_pass='default', sharpen=True):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        features = h.view(h.size(0), h.size(1))
        out=self.l_c1(features)
        if forward_pass == 'default':
            return out

        elif forward_pass == 'backbone':
            return features

        elif forward_pass == 'cls':
            return out

        elif forward_pass == 'proj':
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            return q

        elif forward_pass == 'cls_proj':
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            return out, q

        elif forward_pass == 'all':
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            prototypes = self.prototypes.clone().detach()
            if sharpen:
                logits_proto = torch.mm(q, prototypes.t()) / self.temperature
            else:
                logits_proto = torch.mm(q, prototypes.t())
            return out, logits_proto, q
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return logit


class CNN_bak(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25):
        self.dropout_rate = dropout_rate
        super(CNN_bak, self).__init__()

        #block1
        self.conv1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(128)

        #block2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4=nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5=nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6=nn.BatchNorm2d(256)

        #block3
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.bn7=nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.bn8=nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.bn9=nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        
        self.fc=nn.Linear(128,n_outputs)
        self.temperature = 0.1
        self.proto_m = 0.99
        self.low_dim = 128

        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(), nn.Linear(128, self.low_dim))
        self.register_buffer("prototypes", torch.zeros(n_outputs, self.low_dim))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def init_prototypes(self, features, labels):
        # features, labels are lists of tensors
        assert set(range(len(self.prototypes))) == set(labels.cpu().numpy()), \
            f'The missing labels are {set(range(len(self.prototypes))) - set(labels.cpu().numpy())}'
        for i in range(self.prototypes.shape[0]):
            self.prototypes[i] = features[labels == i].mean(dim=0)
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    @torch.no_grad()
    def update_prototypes(self, features, labels):
        avg_features = torch.zeros_like(self.prototypes).to(self.prototypes.device)
        for i in range(avg_features.shape[0]):
            avg_features[i] = features[labels == i].mean(dim=0)
        avg_features = F.normalize(avg_features, p=2, dim=1)
        for i in range(avg_features.shape[0]):
            if not torch.isnan(avg_features[i]).any():
                self.prototypes[i] = self.proto_m * self.prototypes[i] + (1 - self.proto_m) * avg_features[i]
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

    def forward(self, x, forward_pass='default', sharpen=True):
        
        #block1
        x=F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x=F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x=F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x=self.pool(x)
        x=F.dropout2d(x, p=self.dropout_rate)

        #block2
        x=F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x=F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01)
        x=F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.01)
        x=self.pool(x)
        x=F.dropout2d(x, p=self.dropout_rate)

        #block3
        x=F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.01)
        x=F.leaky_relu(self.bn8(self.conv8(x)), negative_slope=0.01)
        x=F.leaky_relu(self.bn9(self.conv9(x)), negative_slope=0.01)
        x=self.avgpool(x)

        features = x.view(x.size(0), x.size(1))
        out=self.fc(features)
        if forward_pass == 'default':
            return out

        elif forward_pass == 'backbone':
            return features

        elif forward_pass == 'cls':
            return out

        elif forward_pass == 'proj':
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            return q

        elif forward_pass == 'cls_proj':
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            return out, q

        elif forward_pass == 'all':
            q = self.projector(features)
            q = F.normalize(q, dim=1)
            prototypes = self.prototypes.clone().detach()
            if sharpen:
                logits_proto = torch.mm(q, prototypes.t()) / self.temperature
            else:
                logits_proto = torch.mm(q, prototypes.t())
            return out, logits_proto, q
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=14):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model