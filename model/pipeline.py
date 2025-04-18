import torch
from torch.utils.data import DataLoader, Subset
from config.config_loader import Config
from data.dataset import NPZSentinelDataset
from data.preprocess import preprocess_images, load_data
from model.utils import build_optimizer, build_scheduler, log_epoch_results
from model.model import Pretrainedmodel
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

        self.model = None
        self.trainer = None
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
            sentinel_data_dir=config.paths["sentinel_data_dir"],
            image_filename_format=config.filenames["images"],
            mask_dir=config.paths["train_mask_dir"],
            mask_filename_format=config.filenames["masks"],
            forest_mask_dir=config.paths["forest_mask_dir"] if config.dataset["use_forest_masks"] else None,
            forest_mask_filename_format=config.filenames["forest_masks"] if config.dataset["use_forest_masks"] else None
        )
        preprocess_images(config, channels_order, config.paths["preprocessed_train_dir"], train_images, train_masks, train_forests)

        # Test
        test_images, test_masks, test_forests = load_data(
            config.paths["sentinel_data_dir"],
            config.filenames["images"],
            config.paths["test_mask_dir"],
            config.filenames["masks"],
            forest_mask_dir=config.paths["forest_mask_dir"] if config.dataset["use_forest_masks"] else None,
            forest_mask_filename_format=config.filenames["forest_masks"] if config.dataset["use_forest_masks"] else None
        )
        preprocess_images(config, channels_order, config.paths["preprocessed_test_dir"], test_images, test_masks, test_forests)

        total_time = datetime.now() - now
        print(f"\nPreprocessing completed. Total time: {str(total_time).split('.')[0]}")

    def setup_datasets(self):
        """Load datasets"""

        config = self.config
        
        # Load train dataset
        target_size = tuple(config.dataset["target_size"])
        train_data_dir = config.paths["preprocessed_train_dir"]
        test_data_dir = config.paths["preprocessed_test_dir"]

        # Load train dataset
        full_train_dataset = NPZSentinelDataset(
            data_dir=train_data_dir,
            target_size=target_size,
            resize_mode=config.dataset["resize_mode"],
            preprocessed_filename=config.filenames["preprocessed"]
        )

        # Split full training set into train and validation sets
        train_len = int(0.8 * len(full_train_dataset))
        train_dataset = Subset(full_train_dataset, range(0, train_len))
        val_dataset = Subset(full_train_dataset, range(train_len, len(full_train_dataset)))

        # Load full test dataset and split into validation and test sets
        test_dataset = NPZSentinelDataset(
            data_dir=test_data_dir,
            target_size=target_size,
            resize_mode=config.dataset["resize_mode"],
            preprocessed_filename=config.filenames["preprocessed"]
        )

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

    def setup_model(self):
        """Load model and set up training components"""

        config = self.config
        
        # Load the model
        model = Pretrainedmodel.from_pretrained(
            config.model["pretrained_name"],
            num_classes=2
        ).to(self.device)

        # Loss function, optimizer, scheduler
        class_weights = torch.tensor(config.training["class_weights"]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = build_optimizer(config, model.parameters())
        scheduler = build_scheduler(config, optimizer, len(self.train_loader), 0)

        # Set up the model
        self.model = model

        # Trainer setup
        self.trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            results_dir=config.paths["results_dir"],
            results_filename=config.filenames["results"],
            experiment_name=config.training["experiment_name"],
            scheduler=scheduler
        )

    def train(self):
        """Train the model"""

        config = self.config
        trainer = self.trainer
        model = self.model

        # Training parameters
        patience = config.training["patience"]
        best_model_path = os.path.join(config.paths["results_dir"], config.training["experiment_name"])
        best_f1_val = 0.0
        no_improve_epochs = 0

        # Training loop
        now = datetime.now()
        for epoch in range(config.training["epochs"]):
            print(f"\nEpoch [{epoch+1}/{config.training['epochs']}] starting...")

            # Train and validate
            train_metrics = trainer.train_epoch(self.train_loader)
            val_metrics = trainer.validate_epoch(self.val_loader)
            test_metrics = trainer.test_epoch(self.test_loader)

            log_epoch_results(
                epoch+1,
                train_metrics,
                val_metrics,
                test_metrics,
                config.training["experiment_name"],
                results_path=config.paths["results_dir"]
            )

            # Early stopping and model saving
            if val_metrics["F1_1"] > best_f1_val:
                best_f1_val = val_metrics["F1_1"]
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        total_time = datetime.now() - now
        print(f"\nTraining complete. Total time: {str(total_time).split('.')[0]}")

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
            self.setup_model()
            self.train()
            self.full_train_dataset.clear_memory()
            self.test_dataset.clear_memory()
        if do_test:
            self.test()
