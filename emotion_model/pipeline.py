from sklearn.pipeline import Pipeline
import keras
from keras.wrappers.scikit_learn import KerasClassifier
# from emotion_model.config.core import config
# from emotion_model.utils.utils import load_obj
from config.core import config
from utils.utils import load_obj

def create_model():
    model = keras.Sequential()
    model.add(
        load_obj(config.layer1.class_name)(**config.layer1.params)
    )
    model.add(
        load_obj(config.layer2.class_name)(**config.layer2.params)
    )
    model.add(
        load_obj(config.layer3.class_name)(**config.layer3.params)
    )
    model.add(
        load_obj(config.layer4.class_name)(**config.layer4.params)
    )
    model.add(
        load_obj(config.layer5.class_name)(**config.layer5.params)
    )
    model.add(
        load_obj(config.layer6.class_name)()
    )
    model.add(
        load_obj(config.layer7.class_name)(**config.layer7.params)
    )
    model.add(
        load_obj(config.layer8.class_name)(**config.layer8.params)
    )

    model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)
    return model

clf = KerasClassifier(build_fn=create_model, verbose=0)

emotion_pipe = Pipeline(
    [
        ('clf', clf)
    ]
)