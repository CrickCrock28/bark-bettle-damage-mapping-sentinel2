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

        if self.config.dataset["year"] == 2019:
            self.sentinel_data_dir = self.config.paths["sentinel_data_dir_2019"]
            self.preprocessed_train_dir = self.config.paths["preprocessed_train_dir_2019"]
            self.preprocessed_test_dir = self.config.paths["preprocessed_test_dir_2019"]
        elif self.config.dataset["year"] == 2020:
            self.sentinel_data_dir = self.config.paths["sentinel_data_dir_2020"]
            self.preprocessed_train_dir = self.config.paths["preprocessed_train_dir_2020"]
            self.preprocessed_test_dir = self.config.paths["preprocessed_test_dir_2020"]
        else:
            raise ValueError("Invalid year specified in the configuration.")
        
        if self.config.dataset["use_forest_masks"]:
            self.preprocessed_filename = self.config.filenames["preprocessed_all"]
        else:
            self.preprocessed_filename = self.config.filenames["preprocessed_filtered"]

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def preprocess(self):
        """Preprocess the dataset"""

        now = datetime.now()
        config = self.config

        # Load configuration parameters
        current_channels = config.channels["current"]
        selected_channels = config.channels["selected"]
        channels_order = [current_channels.index(c) + 1 for c in selected_channels]
        
        # Load train dataset
        train_images, train_masks, train_forests = load_data(
            sentinel_data_dir=self.sentinel_data_dir,
            image_filename_format=config.filenames["images"],
            mask_dir=config.paths["train_mask_dir"],
            mask_filename_format=config.filenames["masks"],
            forest_mask_dir=config.paths["forest_mask_dir"] if config.dataset["use_forest_masks"] else None,
            forest_mask_filename_format=config.filenames["forest_masks"] if config.dataset["use_forest_masks"] else None
        )
        preprocess_images(
            channels_order,
            self.preprocessed_train_dir,
            self.preprocessed_filename,
            self.config.dataset["radius"],
            train_images,
            train_masks,
            train_forests
        )

        # Test
        test_images, test_masks, test_forests = load_data(
            sentinel_data_dir=self.sentinel_data_dir,
            image_filename_format=config.filenames["images"],
            mask_dir=config.paths["test_mask_dir"],
            mask_filename_format=config.filenames["masks"],
            forest_mask_dir=config.paths["forest_mask_dir"] if config.dataset["use_forest_masks"] else None,
            forest_mask_filename_format=config.filenames["forest_masks"] if config.dataset["use_forest_masks"] else None
        )
        preprocess_images(
            channels_order,
            self.preprocessed_test_dir,
            self.preprocessed_filename,
            self.config.dataset["radius"],
            test_images,
            test_masks,
            test_forests
        )

        total_time = datetime.now() - now
        print(f"\nPreprocessing completed. Total time: {str(total_time).split('.')[0]}")

    def setup_datasets(self):
        """Load datasets"""

        config = self.config
        target_size = tuple(config.dataset["target_size"])

        # Load train dataset
        full_train_dataset = NPZSentinelDataset(
            data_dir=self.preprocessed_train_dir,
            target_size=target_size,
            resize_mode=config.dataset["resize_mode"],
            preprocessed_filename=self.preprocessed_filename
        )

        # Split full training set into train and validation sets
        train_len = int(0.8 * len(full_train_dataset))
        train_dataset = Subset(full_train_dataset, range(0, train_len))
        val_dataset = Subset(full_train_dataset, range(train_len, len(full_train_dataset)))

        # Load full test dataset and split into validation and test sets
        test_dataset = NPZSentinelDataset(
            data_dir=self.preprocessed_test_dir,
            target_size=target_size,
            resize_mode=config.dataset["resize_mode"],
            preprocessed_filename=self.preprocessed_filename
        )

        # Create DataLoader for train, validation, and test datasets
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training["batch_size"],
            shuffle=True,
            num_workers=config.dataset["num_workers"],
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training["batch_size"],
            shuffle=False,
            num_workers=config.dataset["num_workers"],
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.training["batch_size"],
            shuffle=False,
            num_workers=config.dataset["num_workers"],
            pin_memory=True
        )

        self.full_train_dataset = full_train_dataset
        self.test_dataset = test_dataset

    def train(self):
        """Train the model"""        

        trainer = Trainer(
            config=self.config
        )

        trainer.train_model(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            config=self.config
        )

    def test(self):
        """Run test only"""
        tester = ModelTester(self.config)
        tester.run_damage_detection()

    def run(self, do_preprocess=False, do_train=False, do_test=False):
        """Run the pipeline with the specified options"""
        if do_preprocess:
            self.preprocess()
        if do_train:
            self.setup_datasets()
            self.train()
            self.full_train_dataset.clear_memory()
            self.test_dataset.clear_memory()
        if do_test:
            self.test()
