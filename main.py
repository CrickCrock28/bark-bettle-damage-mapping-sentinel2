from model.pipeline import Pipeline
import argparse
import warnings

# FIXME workaround for the warning message
warnings.filterwarnings("ignore", message="Keyword 'img_size' unknown*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/config.yaml")
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    pipeline = Pipeline(config_path=args.config)
    pipeline.run(
        do_preprocess=args.preprocess,
        do_train=args.train,
        do_test=args.test
    )
