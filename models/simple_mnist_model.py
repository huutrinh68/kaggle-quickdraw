from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet

class SimpleMnistModel(BaseModel):
    def __init__(self, config):
        super(SimpleMnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=(28 * 28,)))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config.model.optimizer,
            metrics=['acc'],
        )

class MobileNetModel(BaseModel):
    def __init_(self, config):
        super(MobileNetModel, self).__init__(config)
        self.build_model()
    
    def build_model(self):
        img_size = self.config.trainer.img_size
        self.model = MobileNet(input_shape=(img_size, img_size, 1), alpha=1., weights=None, classes=340)
        # self.model = ResNet50(
        #     include_top=False, 
        #     weights='imagenet', 
        #     input_tensor=None, 
        #     input_shape=(img_size, img_size, 1)
        # )
        
        # for layer in self.model.layers:
        #     layer.trainable=False
        
        # x = self.model.output
        # x = GlobalAveragePooling2D(data_format='channels_last')(x)
        # predictions = Dense(self.config.data_attr.num_classes, activation='softmax')(x)
        # self.model = Model(self.model.input, predictions)
        self.model.compile(optimizer=Adam(
            lr=0.002), 
            loss='categorical_crossentropy',
            metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy]
        )