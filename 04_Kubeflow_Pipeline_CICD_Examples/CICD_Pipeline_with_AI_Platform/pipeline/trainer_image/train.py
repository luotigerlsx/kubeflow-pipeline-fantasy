import os
import subprocess
import sys

import fire
import numpy as np
import pandas as pd
import pickle
import hypertune

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow import gfile

def train_evaluate(job_dir, training_dataset_path, validation_dataset_path,
                   alpha, max_iter, hptune):
    with gfile.Open(training_dataset_path, 'r') as f:
        # Assume there is no header
        df_train = pd.read_csv(f)

    with gfile.Open(validation_dataset_path, 'r') as f:
        # Assume there is no header
        df_validation = pd.read_csv(f)

    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    numeric_feature_indexes = slice(0, 10)
    categorical_feature_indexes = slice(10, 12)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_feature_indexes),
            ('cat', OneHotEncoder(), categorical_feature_indexes)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SGDClassifier(loss='log'))
    ])

    num_features_type_map = {feature: 'float64' for feature in
                             df_train.columns[numeric_feature_indexes]}
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)

    print('Starting training: alpha={}, max_iter={}'.format(alpha, max_iter))
    X_train = df_train.drop('Cover_Type', axis=1)
    y_train = df_train['Cover_Type']

    pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
    pipeline.fit(X_train, y_train)

    if hptune:
        X_validation = df_validation.drop('Cover_Type', axis=1)
        y_validation = df_validation['Cover_Type']
        accuracy = pipeline.score(X_validation, y_validation)
        print('Model accuracy: {}'.format(accuracy))
        # Log it with hypertune
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=accuracy
        )

    # Save the model
    if not hptune:

        model_filename = 'model.pkl'
        gcs_model_path = "{}/{}".format(job_dir, model_filename)

        if gfile.Exists(gcs_model_path):
            gfile.Remove(gcs_model_path)

        with gfile.Open(gcs_model_path, 'w') as wf:
            pickle.dump(pipeline, wf)

        print("Saved model in: {}".format(gcs_model_path))
    
if __name__ == "__main__":
    fire.Fire(train_evaluate)
