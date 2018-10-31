from common import *
from params import *
from utils import *


def load_model():
    
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(size, size, 3))
    for layer in model.layers:
        layer.trainable=False
    
    x = model.output
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    predictions = Dense(NCATS, activation='softmax')(x)
    updated_model = Model(model.input, predictions)

    updated_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])

    # model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
    # model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
    #             metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
    return updated_model

