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

    # validation
    valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
    map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)

    # create submission
    test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
    test.head()
    x_test = df_to_image_array_xd(test, size)
    test_predictions = model.predict(x_test, batch_size=128, verbose=1)
    top3 = preds2catids(test_predictions)
    cats = Simplified().list_all_categories()
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
    top3cats = top3.replace(id2cat)
    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = test[['key_id', 'word']]
    submission.to_csv('gs_mn_submission_{}.csv'.format(int(map3 * 10**4)), index=False)

    # show process time
    end = dt.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))



if __name__ == "__main__":
    main()
