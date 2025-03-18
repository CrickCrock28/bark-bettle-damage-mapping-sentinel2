paths = {
    "train_mask_dir": "data/masks/train",
    "test_mask_dir": "data/masks/test",
    "sentinel_data_dirs": [
        "data/2019-09-01 - 2019-09-30/sentinel_2",
        "data/2020-09-01 - 2020-09-30/sentinel_2"
    ]
}

filenames = {
    "masks": "mask_{id}.tif",
    "images": "geojson_{id}.tif"
}

channels = {
    "current": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    "selected": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"]
}

training = {
    "target_size": [224, 224],
    "batch_size": 8,
    "epochs": 2,
    "learning_rate": 0.1
}

model = {
    "pretrained_name": "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
}
