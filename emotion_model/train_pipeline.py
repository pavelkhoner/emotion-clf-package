import logging
from pathlib import Path

from config.core import LOG_DIR, config
from pipeline import emotion_pipe
from processing.data_manager import load_dataset, save_pipeline
# from processing.validation import get_first_cabin, get_title у нас нет такого модуля 
# from sklearn.metrics import accuracy_score, f1_score может заместо него используем который ниже ?
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from emotion_model import __version__ as _version
from emotion_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def image_generator(input_path, image_size): # нужна ли эта функция или в датасете ее пропишем и от туда будем брать данные ?
  """Image and labels generator"""
  emotions = ['surprise', 'neutral', 'sad', 'happy', 'anger']
    for index, emotion in enumerate(emotions):
        for filename in os.listdir(os.path.join(input_path, emotion)):
            img = cv2.imread(os.path.join(input_path, emotion, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0
            img = img.flatten()
            yield img, index


def load_images(input_path, emotions, image_size): # нужна ли эта функция или в датасете ее пропишем и от туда будем брать данные ?
    X, y = [], []
    for img, label in image_generator1(input_path, emotions, image_size):
        X.append(img)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


def run_training() -> None:
  """Train the model."""
    # Update logs
  log_path = Path(f"{LOG_DIR}/log_{_version}.log")
  if Path.exists(log_path):
      log_path.unlink()
  logging.basicConfig(filename=log_path, level=logging.DEBUG)
    
    #
    # read training data
    # data = load_dataset(file_name=config.app_config.training_data_file)

    # data["Cabin"] = data["Cabin"].apply(get_first_cabin)
    # data["Title"] = data["Name"].apply(get_title)

    # cast numerical variables as floats
    # data["Fare"] = data["Fare"].astype("float")
    # data["Age"] = data["Age"].astype("float")

    # data.drop(labels=config.model_config.variables_to_drop, axis=1, inplace=True)
    #
  
    X, y = load_images1(INPUT_PATH, EMOTIONS, IMAGE_SIZE) #тут надо подумать как сделать сбор констант из конфига
    input_shape = X[0].shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(-1, 96, 96, 1) # эти параметры мы тоже вроде должны брать из конфига
    X_test = X_test.reshape(-1, 96, 96, 1) # эти параметры мы тоже вроде должны брать из конфига
    

    #
    # divide train and test
    #X_train, X_test, y_train, y_test = train_test_split(
    #    data[config.model_config.features],  # predictors
    #    data[config.model_config.target],
    #    test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
    #    random_state=config.model_config.random_state,
    # )

    # fit model
    #titanic_pipe.fit(X_train, y_train)
    #

    emotion_pipe.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)) # Эпохи тоже вроде должны из конфига брать

    #
    # make predictions for train set
    #class_ = titanic_pipe.predict(X_train)
    #pred = titanic_pipe.predict_proba(X_train)[:, 1]
    #


    # Предсказание вероятностей принадлежности к каждому классу на обучающем наборе
    class_ = model.predict(X_train)
    pred = np.argmax(y_train_probs, axis=1)

    #
    # determine train accuracy and roc-auc
    #train_accuracy = accuracy_score(y_train, class_)
    #train_roc_auc = roc_auc_score(y_train, pred)

    #print(f"train accuracy: {train_accuracy}")
    #print(f"train roc-auc: {train_roc_auc}")
    #print()
    #

    emotions = ['surprise', 'neutral', 'sad', 'happy', 'anger']
    print(classification_report(class_, pred, target_names=emotions))
    print()

    logging.info(f"train metrics: {classification_report(class_, pred, target_names=emotions)}")
    
    #
    # make predictions for test set
    #class_ = titanic_pipe.predict(X_test)
    #pred = titanic_pipe.predict_proba(X_test)[:, 1]
    #

    class_ = emotion_pipe.predict(X_test)
    pred = np.argmax(class_, axis=1)

    #
    # determine test accuracy and roc-auc
    #test_accuracy = accuracy_score(y_test, class_)
    #test_roc_auc = roc_auc_score(y_test, pred)
    #

    print(classification_report(y_test, pred, target_names=emotions))
    print()

    logging.info(f"test metrics: {classification_report(y_test, pred, target_names=emotions)}")

    #
    #print(f"test accuracy: {test_accuracy}")
    #print(f"test roc-auc: {test_roc_auc}")
    #print()

    #logging.info(f"test accuracy: {test_accuracy}")
    #logging.info(f"test roc-auc: {test_roc_auc}")
    #

    # persist trained model
    save_pipeline(pipeline_to_persist=emotion_pipe)


if __name__ == "__main__":
    run_training()
