import torch
from torch.utils.data import DataLoader, Subset
from config.config_loader import Config
from data.dataset import NPZSentinelDataset
from data.preprocess import preprocess_images, load_data
from model.trainer import Trainer
import os
from datetime import datetime
from model.tester import ModelTester


class Pipeline:
    def __init__(self, config_path):
        
        self.config = Config(config_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.data_loaders = {}
        self.datasets = {}

    def _get_paths(self, year):
        """Returns the correct Sentinel and preprocessed directories"""
        return {
            "sentinel_data_dir": self.config.paths[f"sentinel_data_dir_{year}"],
            "preprocessed_train_dir": self.config.paths[f"preprocessed_train_dir_{year}"],
            "preprocessed_test_dir": self.config.paths[f"preprocessed_test_dir_{year}"]
        }

    def _get_filename(self, filtered):
        if filtered:
            return self.config.filenames["preprocessed_filtered"]
        else:
            return self.config.filenames["preprocessed_all"]

    def preprocess(self):
        """Preprocess the dataset"""

        now = datetime.now()
        config = self.config

        # Load channels order
        current_channels = config.channels["current"]
        selected_channels = config.channels["selected"]
        channels_order = [current_channels.index(c) + 1 for c in selected_channels]
        
        for year in [2019, 2020]:
            for filtered in [True, False]:

                # Get paths and filenames
                paths = self._get_paths(year)
                filename = self._get_filename(filtered)
                forest_dir = config.paths["forest_mask_dir"] if filtered else None
                forest_format = config.filenames["forest_masks"] if filtered else None

                # Train
                train_images, train_masks, train_forests = load_data(
                    sentinel_data_dir=paths["sentinel_data_dir"],
                    image_filename_format=config.filenames["images"],
                    mask_dir=config.paths["train_mask_dir"],
                    mask_filename_format=config.filenames["masks"],
                    forest_mask_dir=forest_dir,
                    forest_mask_filename_format=forest_format
                )
                preprocess_images(
                    channels_order,
                    paths["preprocessed_train_dir"],
                    filename,
                    config.dataset["radius"],
                    train_images,
                    train_masks,
                    train_forests
                )

                # Test
                test_images, test_masks, test_forests = load_data(
                    sentinel_data_dir=paths["sentinel_data_dir"],
                    image_filename_format=config.filenames["images"],
                    mask_dir=config.paths["test_mask_dir"],
                    mask_filename_format=config.filenames["masks"],
                    forest_mask_dir=forest_dir,
                    forest_mask_filename_format=forest_format
                )
                preprocess_images(
                    channels_order,
                    paths["preprocessed_test_dir"],
                    filename,
                    config.dataset["radius"],
                    test_images,
                    test_masks,
                    test_forests
                )

        total_time = datetime.now() - now
        print(f"\nPreprocessing completed. Total time: {str(total_time).split('.')[0]}")

    def load_dataset(self, year, filtered, is_train):
        paths = self._get_paths(year)
        filename = self._get_filename(filtered)
        data_dir = paths["preprocessed_train_dir"] if is_train else paths["preprocessed_test_dir"]

        return NPZSentinelDataset(
            data_dir=data_dir,
            target_size=self.config.dataset["target_size"],
            resize_mode=self.config.dataset["resize_mode"],
            preprocessed_filename=filename
        )

    def setup_datasets(self):
        """Load all 4 combinations of train/test datasets"""
        for year in [2019, 2020]:
            for filtered in [True, False]:
                key = f"{year}_{'filtered' if filtered else 'all_data'}"

                # Train Dataset + split
                full_train_dataset = self.load_dataset(year, filtered, is_train=True)
                train_len = int(0.8 * len(full_train_dataset))
                train_subset = Subset(full_train_dataset, range(0, train_len))
                val_subset = Subset(full_train_dataset, range(train_len, len(full_train_dataset)))

                self.datasets[f"{key}_full_train"] = full_train_dataset
                self.data_loaders[f"{key}_train"] = DataLoader(
                    train_subset,
                    batch_size=self.config.training["batch_size"],
                    shuffle=True,
                    num_workers=self.config.dataset["num_workers"],
                    pin_memory=True
                )
                self.data_loaders[f"{key}_val"] = DataLoader(
                    val_subset,
                    batch_size=self.config.training["batch_size"],
                    shuffle=False,
                    num_workers=self.config.dataset["num_workers"],
                    pin_memory=True
                )

                # Test dataset
                test_dataset = self.load_dataset(year, filtered, is_train=False)
                self.datasets[f"{key}_test"] = test_dataset
                self.data_loaders[f"{key}_test"] = DataLoader(
                    test_dataset,
                    batch_size=self.config.training["batch_size"],
                    shuffle=False,
                    num_workers=self.config.dataset["num_workers"],
                    pin_memory=True
                )

    def train(self):
        """Train the model using the config-specified dataset"""
        year = self.config.dataset["year"]
        filtered = self.config.dataset["use_forest_masks"]
        key = f"{year}_{'filtered' if filtered else 'all_data'}"

        trainer = Trainer(config=self.config)
        trainer.train_model(
            train_loader=self.data_loaders[f"{key}_train"],
            val_loader=self.data_loaders[f"{key}_val"],
            test_loader=self.data_loaders[f"{key}_test"],
            config=self.config
        )

    def test(self):
        """Run test only"""
        tester = ModelTester(self.config)
        tester.run_damage_detection()

        
    # def test(self):
    #     """Test the model"""
    #     tester = ModelTester(
    #         config=self.config,
    #         test_loaders={
    #             key: loader for key, loader in self.data_loaders.items() if key.endswith("_test")
    #         }
    #     )
    #     tester.run_damage_detection()

    def run(self, do_preprocess=False, do_train=False, do_test=False):
        """Run the pipeline with the specified options"""
        if do_preprocess:
            self.preprocess()
        if do_train or do_test:
            self.setup_datasets()
        if do_train:
            self.train()
        if do_test:
            self.test()
