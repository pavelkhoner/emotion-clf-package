# Emotion model

## Установка библиотеки

Чтобы установить библиотеку, выполните:

```
pip install emotion-model
```

После этого сделайте вызов функции на вашем изображении, чтобы получить предсказание:

```
from emotion_model.predict import make_prediction

result = make_prediction(<Путь к вашему изображению>)

print(result)
```

## Ссылки
Библиотека выполнена на основе статьи Хонера П. Д. "Классификация эмоций на лице человека при помощи компьютерного зрения" (https://www.researchgate.net/publication/379937357_Klassifikacia_emocij_na_lice_celoveka_pri_pomosi_komputernogo_zrenia). Также выражаем благодарность Тагиеву Э. Р., поскольку его репозитории https://github.com/Emilien-mipt/titanic-package и https://github.com/Emilien-mipt/fer-pytorch использовались в качестве примера.
