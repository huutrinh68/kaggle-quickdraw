from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard

class MobileNetModelTrainer(BaseTrain):
    def __init__(self, model, data_train, data_test, config):
        super(MobileNetModelTrainer, self).__init__(model, data_train, data_test, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        if hasattr(self.config,"comet_ml"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_ml.comet_api_key, project_name=self.config.comet_ml.project_name, workspace=self.config.comet_ml.workspace)
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        history = self.model.fit_generator(
            generator=self.data_train,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch = self.config.trainer.steps_per_epoch,
            validation_data=self.data_test,
            validation_steps=self.config.trainer.validation_steps,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
