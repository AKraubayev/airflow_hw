import json
import pandas as pd
import dill
import os
import glob
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


os.environ["MLFLOW_REGISTRY_URI"] = "/home/mlops/project/mlflow/"
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("predict_of_model")

def predict():
    path = os.path.expanduser('~/airflow_hw')
    with open(f'{path}/data/models/cars_pipe_202303221454.pkl', 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])

    path_files = path + '/data/test/*json'
    for json_files_path in glob.iglob(path_files):
        with open(json_files_path) as fin:
            form = json.load(fin)
            #print(form)
            df = pd.DataFrame.from_dict([form])
            pred = model.predict(df)
            x = {'id': df.id, 'pred': pred}
            #y = model.predict(df)
            data = pd.DataFrame(x)
            print(data)
            data.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')

            
if __name__ == '__main__':
    predict()
    
    
with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path=path"/modules/predict.py",
                        artifact_path="predict_of_model code")
    mlflow.end_run()
