from common import *
from params import *
from utils import *

def load_model():
    model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
    model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
                metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
    return model

