# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kubeflow pipeline using BigQuery for preprocessing and AutoML Tables for modelling."""
import os
import time
from typing import NamedTuple

import kfp
from google.cloud import bigquery
from jinja2 import Template
from kfp.components import func_to_container_op

BASE_IMAGE = os.getenv('BASE_IMAGE')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')

FEATURE_TABLE_ID = 'model_features_{}'
AUTOML_DATASET_NAME = 'cover_dataset_{}'
AUTOML_MODEL_NAME = 'cover_model_{}'

component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX]
)

bigquery_query_op = component_store.load_component('bigquery/query')
automl_create_dataset_op = component_store.load_component('automl/create_dataset_for_tables')
automl_import_data_from_bq_op = component_store.load_component('automl/import_data_from_bigquery')
automl_create_model_op = component_store.load_component('automl/create_model_for_tables')
automl_split_dataset_table_column_names_op = component_store.load_component('automl/split_dataset_table_column_names')


def prepare_query(project_id, dataset_id):
    """Prepare the preprocessing query."""

    query_template = """
        SELECT *
        FROM `{{ project_id }}.{{ dataset_id }}.covertype`
        WHERE Soil_Type IN (
          SELECT Soil_Type FROM `{{ project_id }}.{{ dataset_id }}.covertype`
          GROUP BY 1
          HAVING COUNT(*) > 500
        )
        """

    return Template(query_template).render(
        project_id=project_id, dataset_id=dataset_id)


def create_bq_job_config():
    """Create the BigQuery job configuration."""

    bq_job_config = bigquery.QueryJobConfig()
    bq_job_config.create_disposition = bigquery.job.CreateDisposition.CREATE_IF_NEEDED
    bq_job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

    return bq_job_config.to_api_repr()


def automl_data_statistics(project_id: str,
                           region: str,
                           dataset_path: str
                           ) -> NamedTuple('Outputs', [('max_data_missing_rate', float),
                                                       ('mlpipeline_ui_metadata', 'UI_metadata')]):
    """Extract the AutoML Tables dataset statistics.

    Args:
        project_id (str): The project hosting the AutoML dataset.
        region (str): The AutoML Tables region.
        dataset_path (str): The AutoML Tables dataset path.

    Returns:
        A NamedTuple contains the maximum missing data rate across all features and
        the missing counts for all features in KFP UI metadata table format
    """
    from collections import namedtuple
    import json
    import logging
    from google.cloud import automl_v1beta1 as automl

    logging.basicConfig(level=logging.INFO)
    client = automl.TablesClient(project=project_id, region=region)
    column_specs = client.list_column_specs(dataset_name=dataset_path)

    max_data_missing_rate = 0.0
    data = ''
    for spec in column_specs:
        data += "'{}',{},{}\n".format(spec.display_name,
                                      spec.data_stats.null_value_count,
                                      spec.data_stats.valid_value_count)
        max_data_missing_rate = max(max_data_missing_rate,
                                    float(spec.data_stats.null_value_count) / spec.data_stats.valid_value_count)

    headers = ['Feature Name', 'Missing Count', 'Total Count']
    metadata = {
        'outputs': [{
            'type': 'table',
            'storage': 'inline',
            'format': 'csv',
            'header': headers,
            'source': data
        }]
    }

    stats_output = namedtuple('Outputs', ['max_data_missing_rate', 'mlpipeline_ui_metadata'])
    return stats_output(max_data_missing_rate, json.dumps(metadata))


def retrieve_classification_metrics(project_id: str,
                                    region: str,
                                    model_path: str,
                                    metric_name: str) -> NamedTuple('Outputs', [('metric_value', float)]):
    """Retrieve the specified evaluation metric from AutoML Tables model evaluation.

    Args:
        project_id (str): The project hosting the AutoML dataset.
        region (str): The AutoML Tables region.
        model_path (str): The AutoML Tables model path.
        metric_name (str): The metric name. Valid values are ['au_prc', 'au_roc', 'log_loss']. Defaults to 'log_loss'.

    Returns:
        NamedTuple of the metric value.
    """

    from google.cloud import automl_v1beta1 as automl

    client = automl.TablesClient(project=project_id, region=region)
    evaluation = None
    for ev in client.list_model_evaluations(model_name=model_path):
        if ev.display_name == '':
            evaluation = ev
            break

    return (getattr(evaluation.classification_evaluation_metrics, metric_name),)


def automl_deploy_model(project_id: str,
                        region: str,
                        model_path: str,
                        evaluation_metrics: str,
                        deployment_threshold: float,
                        metric_value: float):
    """Deploy the AutoML Tables model.

    Args:
        project_id (str): The project hosting the AutoML dataset.
        region (str): The AutoML Tables region.
        model_path (str): The AutoML Tables model path.
        evaluation_metrics (str): The evaluation metric value to check before deployment.
            Valid values are ['au_prc', 'au_roc', 'log_loss']. Defaults to 'log_loss'.
        deployment_threshold (float): The evaluation threshold requirement for model deployment.
            If the evaluation metrics is 'log_loss', the model will be deployed only if the metric value
            is less than the threshold; otherwise the metric value must be greater than the threshold
            in order to deploy.
        metric_value (float): The evaluation metric value requirement.
    """
    import logging
    from google.cloud import automl_v1beta1 as automl
    from google.cloud.automl_v1beta1 import enums

    logging.basicConfig(level=logging.INFO)
    client = automl.TablesClient(project=project_id, region=region)

    model = client.get_model(model_name=model_path)
    if model.deployment_state != enums.Model.DeploymentState.DEPLOYED:
        if evaluation_metrics == 'log_loss':
            deploy = metric_value < deployment_threshold
        else:
            deploy = metric_value > deployment_threshold

        if deploy:
            logging.info('Starting model deployment: {}'.format(model_path))
            response = client.deploy_model(model_name=model_path)
            response.result()  # Wait for operation to complete
            logging.info('Deployment completed')
        else:
            logging.error('Fail to deploy model')
            raise ValueError('Cannot meet evaluation requirement, abort deployment.')
    else:
        logging.info('Model already deployed')


def automl_export_model(model_path: str,
                        gcs_destination: str):
    """Export the AutoML Tables model.

    Args:
        model_path (str): The AutoML Tables model path.
        gcs_destination (str): The GCS location for the exported model.
    """
    import logging
    from google.cloud import automl_v1beta1 as automl

    logging.basicConfig(level=logging.INFO)

    if gcs_destination is not None:
        logging.info('Export model to path: {}'.format(gcs_destination))
        client = automl.AutoMlClient()
        output_config = {'model_format': 'tf_saved_model',
                         'gcs_destination': {'output_uri_prefix': gcs_destination}}
        client.export_model(model_path, output_config)


# Convert the functions to KFP components
automl_data_statistics_op = func_to_container_op(automl_data_statistics, base_image=BASE_IMAGE)
retrieve_classification_metrics_op = func_to_container_op(retrieve_classification_metrics, base_image=BASE_IMAGE)
automl_deploy_model_op = func_to_container_op(automl_deploy_model, base_image=BASE_IMAGE)
automl_export_model_op = func_to_container_op(automl_export_model, base_image=BASE_IMAGE)


@kfp.dsl.pipeline(
    name='BigQuery & AutoML Cover Pipeline',
    description='The pipeline to train an AutoML Tables model based on BigQuery dataset',
)
def bq_automl_pipeline(project_id,
                       region: str,
                       dataset_id: str,
                       dataset_location: str = "US",
                       data_missing_threshold: float = 0.1,
                       optimization_objective: str = "MINIMIZE_LOG_LOSS",
                       evaluation_metrics: str = "log_loss",
                       deployment_threshold: float = 0.15,
                       train_budget: int = 1000,
                       gcs_destination: str = None):
    """Example Kubeflow pipeline with BigQuery preprocessing and AutoML Tables modelling.

    Args:
        project_id (str): The project hosting BigQuery and AutoML resources.
        region (str): The region for AutoML Table dataset.
        dataset_id (str): The BigQuery dataset storing the preprocessed data.
        dataset_location (str): The BigQuery dataset location. Defaults to 'US'.
        data_missing_threshold (float): The maximum allowed missing data per feature in percentage.
            If the missing rate of any feature is greater or equal to this value,
            the pipeline will be stopped. Defaults to 0.1.
        optimization_objective (str): The metric AutoML tables should optimize for. Defaults to 'MINIMIZE_LOG_LOSS'.
        evaluation_metrics (str): The evaluation metric value to check before deployment.
            Valid values are ['au_prc', 'au_roc', 'log_loss']. Defaults to 'log_loss'.
        deployment_threshold (float): The deployment threshold to meet before deploying the model.
            If the evaluation metrics is 'log_loss', the deployment will happen only if the metric value
            is less than the threshold. Otherwise, the metric value must be greater than the threshold
            in order to deploy.
        train_budget (int): The amount of time (in milli node hours) to spend on training. Defaults to 1000 (1 hour).
        gcs_destination (str): The GCS prefix to store the exported model. Defaults to None.
    """
    current_milliseconds = int(time.time() * 1000.0)
    output_table_id = FEATURE_TABLE_ID.format(current_milliseconds)

    prepare_features = bigquery_query_op(
        query=prepare_query(project_id, dataset_id),
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=output_table_id,
        dataset_location=dataset_location,
        job_config=create_bq_job_config()
    )

    # Create AutoML tables dataset
    create_dataset = automl_create_dataset_op(
        gcp_project_id=project_id,
        gcp_region=region,
        display_name=AUTOML_DATASET_NAME.format(current_milliseconds)
    )
    create_dataset.after(prepare_features)

    # Import the features from BigQuery to AutoML Dataset
    import_data = automl_import_data_from_bq_op(
        dataset_path=create_dataset.outputs['dataset_path'],
        input_uri='bq://{}.{}.{}'.format(project_id, dataset_id, output_table_id)
    )

    # Data Validation
    data_stats = automl_data_statistics_op(project_id, region, create_dataset.outputs['dataset_path'])
    data_stats.after(import_data)

    # Continue the pipeline only if the data missing rate is smaller than a predefined threshold
    with kfp.dsl.Condition(data_stats.outputs['max_data_missing_rate'] < data_missing_threshold,
                           'automl-data-validation'):
        # Set the target column label
        split_column_specs = automl_split_dataset_table_column_names_op(
            dataset_path=import_data.outputs['dataset_path'],
            table_index=0,
            target_column_name='Cover_Type'
        )

        # Create a model
        create_model = automl_create_model_op(
            gcp_project_id=project_id,
            gcp_region=region,
            display_name=AUTOML_MODEL_NAME.format(current_milliseconds),
            dataset_id=create_dataset.outputs['dataset_id'],
            target_column_path=split_column_specs.outputs['target_column_path'],
            input_feature_column_paths=split_column_specs.outputs['feature_column_paths'],
            optimization_objective=optimization_objective,
            train_budget_milli_node_hours=train_budget
        )

        # Export the model
        automl_export_model_op(create_model.outputs['model_path'], gcs_destination)

        # Retrieve the evaluation metric from the model evaluations
        retrieve_metrics = retrieve_classification_metrics_op(
            project_id=project_id,
            region=region,
            model_path=create_model.outputs['model_path'],
            metric_name=evaluation_metrics)

        # Deploy the model if the primary metric is better than threshold
        automl_deploy_model_op(project_id, region, create_model.outputs['model_path'],
                               evaluation_metrics, deployment_threshold,
                               retrieve_metrics.outputs['metric_value'])
