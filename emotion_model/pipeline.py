from sklearn.pipeline import Pipeline
import keras
#from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
# from emotion_model.config.core import config
# from emotion_model.utils.utils import load_obj
from config.core import config
from utils.utils import load_obj

def create_model():
    model = keras.Sequential()
    model.add(
        load_obj(config.model_config.layer1.class_name)(**config.model_config.layer1.params)
    )
    # model.add(
    #     load_obj(config.model_config.layer1.class_name)(
    #         config.model_config.layer1.params.filters,
    #         config.model_config.layer1.params.kernel_size,
    #         activation=config.model_config.layer1.params.activation,
    #         input_shape=config.model_config.layer1.params.input_shape
    #         )
    # )
    model.add(
        load_obj(config.model_config.layer2.class_name)(**config.model_config.layer2.params)
    )
    model.add(
        load_obj(config.model_config.layer3.class_name)(**config.model_config.layer3.params)
    )
    model.add(
        load_obj(config.model_config.layer4.class_name)(**config.model_config.layer4.params)
    )
    model.add(
        load_obj(config.model_config.layer5.class_name)(**config.model_config.layer5.params)
    )
    model.add(
        load_obj(config.model_config.layer6.class_name)()
    )
    model.add(
        load_obj(config.model_config.layer7.class_name)(**config.model_config.layer7.params)
    )
    model.add(
        load_obj(config.model_config.layer8.class_name)(**config.model_config.layer8.params)
    )

    model.compile(optimizer=config.model_config.optimizer, loss=config.model_config.loss, metrics=config.model_config.metrics)
    return model

k = create_model()
k.summary()
clf = KerasClassifier(build_fn=create_model, verbose=0)

emotion_pipe = Pipeline(
    [
        ('clf', clf)
    ]
)