import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import cv2 
from PIL import Image

class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.fc_angles(x)
        return x

class AlexNet(nn.Module):
    # AlexNet laid out as a Hopenet - classify Euler angles in bins and
    # regress the expected value.
    def __init__(self, num_bins):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc_yaw = nn.Linear(4096, num_bins)
        self.fc_pitch = nn.Linear(4096, num_bins)
        self.fc_roll = nn.Linear(4096, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll
    

class headPose:
    def __init__(self, model_path):
        self.model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        saved_state_dict = torch.load(model_path)
        self.model.load_state_dict(saved_state_dict)
        
        
        self.transformations = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.gpu_id = 0
        self.model.cuda(self.gpu_id)
        self.model.eval()
        # straight face
        # -20 < yaw_predicted  < 20
        # -20 <pitch_predicted < 20
        # -20 <roll_predicted < 20
        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpu_id)
        
    def extractFace_origin(self, bbox, frame):
        x_min = max(bbox[0], 0)
        y_min = max(bbox[1], 0)
        x_max = min(frame.shape[1], bbox[2])
        y_max = min(frame.shape[0], bbox[3])

        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        if bbox_width > 70 and bbox_height > 70:                 
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
            #print("x_min = {}, x_max = {}, y_min = {}, y_max = {}".format(x_min, x_max, y_min, y_max))
            # Crop image
            face = frame[int(y_min):int(y_max),int(x_min):int(x_max)]
            #cv2.imshow("face",img)
            size = bbox_height/2
            tdx = (x_min + x_max) / 2
            tdy = (y_min + y_max) / 2
            return face, (tdx, tdy, size)
        else:
            return None, None
        
    def extractFace(self, bbox, frame):
        x_min = max(bbox[0], 0)
        y_min = max(bbox[1], 0)
        x_max = min(frame.shape[1], bbox[2])
        y_max = min(frame.shape[0], bbox[3])

        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        if bbox_width > 70 and bbox_height > 70:                 
            x_min -= int(bbox_width * 0.2)
            x_max += int(bbox_width * 0.2)
            y_min -= int(bbox_height* 0.2)
            y_max += int(bbox_height* 0.2)
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
            #print("x_min = {}, x_max = {}, y_min = {}, y_max = {}".format(x_min, x_max, y_min, y_max))
            # Crop image
            face = frame[int(y_min):int(y_max),int(x_min):int(x_max)]
            #cv2.imshow("face",img)
            size = bbox_height/2
            tdx = (x_min + x_max) / 2
            tdy = (y_min + y_max) / 2
            return face, (tdx, tdy, size)
        else:
            return None, (None, None, None)
        
    def getHeadPose(self, bbox, frame):
        face, (tdx, tdy, size) =  self.extractFace(bbox, frame)     
        if face is None:
            return None, None, None, None
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face)
        
        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(self.gpu_id)

        yaw, pitch, roll = self.model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99
        
        return yaw_predicted, pitch_predicted, roll_predicted, (tdx, tdy, size)
    


        
