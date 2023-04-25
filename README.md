# Пайплайн обучения ML-модели.

1. шаблон DAG’а (dags/hw_dag.py),
2. готовый код ML-модели (modules/pipeline.py),
3. шаблон скрипта для прогноза моделью (modules/predict.py),
4. данные для обучения и тестирования (data/train, data/test),
5. пустые папки под сохранение ML-модели и предсказаний.

# Запуск

Клонировать папку airflow_hw  и открыть её в Pycharm.

Запустить пайплайн с моделью локально и в Airflow, это обучит и сохранит объект с пайплайном лучшей модели в pickle формате:

Выполнить комманду в терминале локально: python3 modules/pipeline.py (из терминала Pycharm).

в Airflow: скопировать файл hw_dag.py в папку $AIRFLOW_HOME/dags.

# После этого в интерфейсе отобразится новый DAG:

Код в файле modules/predict.py, который при вызове функции predict():
загружает обученную модель, делает предсказания для всех объектов в папке data/test,
объединяет предсказания в один Dataframe и сохраняет их в csv-формате в папку data/predictions.

Проверить корректность кода, запустив его локально: python3 modules/predict.py (из терминала Pycharm)

pipeline — здесь выполняется функция pipeline.

predict — здесь делается предикт для всех объектов и сохраняется в папку data/predictions.

Запустить пайплайн в интерфейсе Airflow и получить предикты модели.
