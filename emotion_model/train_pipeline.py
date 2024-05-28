import logging
from pathlib import Path

import numpy as np
from config.core import LOG_DIR, config
from pipeline import emotion_pipe
from processing.data_manager import load_dataset, save_pipeline
# from processing.validation import get_first_cabin, get_title у нас нет такого модуля 
# from sklearn.metrics import accuracy_score, f1_score может заместо него используем который ниже ?
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from emotion_model import __version__ as _version
from emotion_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def run_training() -> None:
    """Train the model."""
    # Update logs
    log_path = Path(f"{LOG_DIR}/log_{_version}.log")
    if Path.exists(log_path):
        log_path.unlink()
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    emotions = config.nn_config.emotions

    X, y = load_dataset(emotions)
    # input_shape = X[0].shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.nn_config.test_size,
                                                        random_state=config.nn_config.random_state)
    # X_train = X_train.reshape(-1, 96, 96, 1)
    # X_test = X_test.reshape(-1, 96, 96, 1)
    X_train = X_train.reshape(*config.nn_config.image_reshape_params)
    X_test = X_test.reshape(*config.nn_config.image_reshape_params)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    emotion_pipe.fit(X_train, y_train)

    # Предсказание вероятностей принадлежности к каждому классу на обучающем наборе
    class_ = emotion_pipe.predict(X_train)
    pred = np.argmax(class_, axis=1)

    print(classification_report(class_, pred, target_names=emotions))
    print()

    logging.info(f"train metrics: {classification_report(class_, pred, target_names=emotions)}")

    class_ = emotion_pipe.predict(X_test)
    pred = np.argmax(class_, axis=1)

    print(classification_report(y_test, pred, target_names=emotions))
    print()

    logging.info(f"test metrics: {classification_report(y_test, pred, target_names=emotions)}")

    # persist trained model
    save_pipeline(pipeline_to_persist=emotion_pipe)


if __name__ == "__main__":
    run_training()
