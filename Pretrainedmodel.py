import torch.nn as nn
from src.pretrain.reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

class Pretrainedmodel(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.model.fc = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            in_features=2048,
            out_features=num_classes
        )

        # Sigmoid to get a simple binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        x4 = self.global_pool(x4)
        x4 = x4.view(x4.size(0), -1)

        y = self.fc(x4)
        
        # Apply sigmoid to get a simple binary classification
        y = self.sigmoid(y)
        return y

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1):
        original_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)
        encoder = original_model.model.vision_encoder
        # print(f'Loaded vision encoder from {model_name}')
        # print(encoder)
        return cls(model=encoder, num_classes=num_classes)

