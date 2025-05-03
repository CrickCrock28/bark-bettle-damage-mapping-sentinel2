import torch.nn as nn
from model.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

class Pretrainedmodel(nn.Module):
    """Pretrained model with custom head for binary classification."""
    def __init__(self, model, num_classes):
        """Initialize the model."""
        super().__init__()
        self.model = model
        self.model.fc = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Custom head for binary classification
        self.fc_layers = nn.Linear(
            in_features=2048,
            out_features=num_classes
        )

    def forward(self, x):
        """Forward pass through the model."""
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

        # Forward through custom head
        y = self.fc_layers(x4)

        return y

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1):
        """Load a pretrained model from the BigEarthNetv2.0 publication."""
        original_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)
        encoder = original_model.model.vision_encoder
        # print(encoder)
        return cls(model=encoder, num_classes=num_classes)
