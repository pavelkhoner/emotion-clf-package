from sklearn.pipeline import Pipeline
import keras
from scikeras.wrappers import KerasClassifier
from emotion_model.config.core import config
from emotion_model.utils.utils import load_obj

def create_model():
    model = keras.Sequential()
    model.add(
        load_obj(config.nn_config.layer1['class_name'])(**config.nn_config.layer1['params'])
    )
    model.add(
        load_obj(config.nn_config.layer2['class_name'])(**config.nn_config.layer2['params'])
    )
    model.add(
        load_obj(config.nn_config.layer3['class_name'])(**config.nn_config.layer3['params'])
    )
    model.add(
        load_obj(config.nn_config.layer4['class_name'])(**config.nn_config.layer4['params'])
    )
    model.add(
        load_obj(config.nn_config.layer5['class_name'])(**config.nn_config.layer5['params'])
    )
    model.add(
        load_obj(config.nn_config.layer6['class_name'])()
    )
    model.add(
        load_obj(config.nn_config.layer7['class_name'])(**config.nn_config.layer7['params'])
    )
    model.add(
        load_obj(config.nn_config.layer8['class_name'])(**config.nn_config.layer8['params'])
    )

    model.compile(optimizer=config.nn_config.optimizer, loss=config.nn_config.loss, metrics=config.nn_config.metrics)
    return model

clf = KerasClassifier(build_fn=create_model, epochs=config.nn_config.epochs, batch_size=config.nn_config.batch_size, verbose=1)

emotion_pipe = Pipeline(
    [
        ('clf', clf)
    ]
)