from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

class MobileNetModel(BaseModel):
    def __init__(self, config):
        super(MobileNetModel, self).__init__(config)
        self.build_model()
    
    def build_model(self):
        img_size = self.config.trainer.img_size
        self.model = MobileNet(input_shape=(img_size, img_size, 1), alpha=1., weights=None, classes=340)
        self.model.compile(
            optimizer=Adam(lr=0.002), 
            loss='categorical_crossentropy',
            metrics=['acc', categorical_crossentropy, categorical_accuracy, top_3_accuracy]
        )