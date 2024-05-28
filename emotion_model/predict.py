import typing as t
import cv2

from emotion_model import __version__ as _version
from emotion_model.config.core import config
from emotion_model.processing.data_manager import load_pipeline, solo_image_generator
#from emotion_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(
    *,
    image_path: str,
) -> dict:
    """Make a prediction using a saved model pipeline."""

    # Загрузка и предобработка изображения
    image = solo_image_generator(image_path)

    # Предсказание с использованием модели
    preds = _emotion_pipe.predict(image)
    probs = _emotion_pipe.predict_proba(image)

    # Формирование словаря с результатами
    results: t.Dict[str, t.Any] = {
        "preds": [int(np.argmax(pred)) for pred in preds],
        "probs": [float(np.max(prob)) for prob in probs],
        "version": _version,
        "errors": None
    }

    return results
