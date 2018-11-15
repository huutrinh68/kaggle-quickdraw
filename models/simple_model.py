from base.base_model import BaseModel
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, GlobalAveragePooling2D
from keras.applications import MobileNet, NASNetLarge
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

class MobileNetModel(BaseModel):
    def __init__(self, config):
        super(MobileNetModel, self).__init__(config)
        self.num_classes = config.data_attr.num_classes
        self.build_model()
    
    def build_model(self):
        img_size = self.config.trainer.img_size
        num_classes = self.config.data_attr.num_classes
        input_tensor = Input(shape=(img_size, img_size, 3))
        self.model = MobileNet(
            include_top=False,
            input_shape=(img_size, img_size, 3), 
            alpha=self.config.model.alpha, 
            weights='imagenet'
        )
        bn = BatchNormalization()(input_tensor)
        x = self.model(bn)
        x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='sigmoid')(x)
        self.model = Model(input_tensor, output)
        
        self.model.compile(
            optimizer=Adam(lr=self.config.model.learning_rate), 
            loss=self.config.trainer.loss,
            metrics=["acc", categorical_crossentropy, categorical_accuracy, top_3_accuracy]
        )