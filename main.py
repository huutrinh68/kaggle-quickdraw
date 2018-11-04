from comet_ml import Experiment
from data_loader.simple_data_loader import MobileNetDataLoader
from models.simple_model import MobileNetModel
from trainers.simple_trainer import MobileNetModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args, gpu_limitation


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        gpu_control = gpu_limitation()
    except:
        print("Missing or invalid arguments")
        exit(0)

    print("Create the experiments dirs.")
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print("Create the data generator.")
    data_loader = MobileNetDataLoader(config)

    print("Create the model.")
    model = MobileNetModel(config)

    print("Create the trainer.")
    trainer = MobileNetModelTrainer(model.model, data_loader.get_train_data(), data_loader.get_test_data(), config)

    print("Start training the model.")
    trainer.train()


if __name__ == "__main__":
    main()
