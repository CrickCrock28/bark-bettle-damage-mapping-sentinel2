import rasterio

from src.pretrain.reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
import torch.nn as nn

class Pretrainedmodel(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.model.fc = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ti lascio un fully connected layer come esempio per una classificazione binaria come finale
        self.fc = nn.Linear(
            in_features=2048,
            out_features=num_classes,
        )

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

        # da qui in poi dovrai implementare la tua logica (te lo lascio solo come esempio)
        y = self.fc(x4)
        return y

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1):
        original_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)
        encoder = original_model.model.vision_encoder
        print(f'Loaded vision encoder from {model_name}')
        print(encoder)
        return cls(model=encoder, num_classes=num_classes)


def reorder_select_channels (image, current_channels, new_channels):
    channel_to_index = {channel: idx for idx, channel in enumerate(current_channels)}
    new_indices = [channel_to_index[channel] for channel in new_channels]
    reordered_image = image[new_indices, :, :]
    return reordered_image

def __main__():
    # così puoi caricare il codice
    model = Pretrainedmodel.from_pretrained('BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0')

    # esempio invece di caricamento di una immagine (mantenere questo ordine perchè è richiesto dal pretrained che stiamo utilizzando)
    current_channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    new_channels = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]
    image_path = 'data/2019-09-01 - 2019-09-30/sentinel_2/geojson_205232.tif'
    with rasterio.open(image_path) as src:
        image = src.read()
        image = reorder_select_channels(image, current_channels, new_channels)


    # TODO il caricamento delle immagini nel modello prevede un ingresso a 224x224 per 10 canali
    # TODO implementare quindi il caricamento delle immagini (se più piccole di 224x224, aggiungere padding a zero)