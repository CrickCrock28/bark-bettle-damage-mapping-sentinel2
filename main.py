from model.pipeline import Pipeline
import argparse
import warnings

warnings.filterwarnings("ignore", message="Keyword 'img_size' unknown*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/config.yaml", help='Path to the config file')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--train', action='store_true', help='Train the model specified in the config file')
    parser.add_argument('--test', action='store_true', help='Test the model specified in the config file')
    parser.add_argument('--damage_detection', action='store_true', help='Run damage detection using the model specified in the config file')
    args = parser.parse_args()

    pipeline = Pipeline(config_path=args.config)
    pipeline.run(
        do_preprocess=args.preprocess,
        do_train=args.train,
        do_test=args.test,
        do_damage_detection=args.damage_detection
    )
