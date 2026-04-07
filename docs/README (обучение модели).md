## КАК РАБОТАТЬ
### ПЕРВИЧНОЕ ОБУЧЕНИЕ (train)
(когда нет моделей, или хотим, чтобы с 0 обучить модель)

`python app.py train arg1 arg2 input.csv output.csv "arg3"`

`arg1` — 0 если хотим бинарную модель, 1 если качественную модель

`arg2` — 1 если хотим обучить на всех данных, Х на каком-то количестве данных (если Х = 0.8, значит 80% будет для обучения, 20% отсеется и выйдет в output.csv

`arg3` — это название целевого столбца (target), то есть колонка, которую модель должна предсказывать.
Примеры запросов:
1) `python app.py train 0 0.6 data/table_NaN_binary.csv binary_test.csv "Оценка (0 или 1)"`
2) `python app.py train 1 1.0 data/table_NaN_quality.csv quality_test.csv "ОЦЕНКА_NUM"`
### ДООБУЧЕНИЕ (retrain)
(когда модели есть и хотим их улучшить новыми данными. Данные поглощаются полностью)

`python app.py retrain arg1 input.csv “arg3”`

Пример запросов:
1) `python app.py retrain 0 new_binary.csv "Оценка (0 или 1)"`
2) `python app.py retrain 1 new_quality.csv "ОЦЕНКА_NUM"`

## ПРЕДСКАЗАНИЕ (predict) 
(когда модели есть и хотим предсказания, занимает очень много времени из-за SHAP и вывода факторов влияния, у меня это по 30 минут каждый, можно спокойно идти обедать)

`python app.py predict arg1 input.csv output.csv`
Пример запросов:
1) `python app.py predict 0 binary_test.csv binary_output.csv`
2) `python app.py predict 1 quality_test.csv quality_output.csv`

