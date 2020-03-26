# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Covertype training pipeline DSL."""

import os
from typing import Dict, List, Text

from kfp import gcp

from tfx.components.base import executor_spec
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform

from tfx.extensions.google_cloud_ai_platform.pusher import \
    executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import \
    executor as ai_platform_trainer_executor

from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from tfx.proto import evaluator_pb2
from tfx.proto import trainer_pb2

from tfx.utils.dsl_utils import external_input


def _create__pipeline(pipeline_name: Text,
                      pipeline_root: Text,
                      data_root: Text,
                      module_file: Text,
                      ai_platform_training_args: Dict[Text, Text],
                      ai_platform_serving_args: Dict[Text, Text],
                      beam_pipeline_args: List[Text]
                      ) -> pipeline.Pipeline:
    """Implements the online news pipeline with TFX."""

    examples = external_input(data_root)

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = CsvExampleGen(input=examples)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs.examples)

    # Generates schema based on statistics files.
    infer_schema = SchemaGen(statistics=statistics_gen.outputs.output)

    # Performs anomaly detection based on statistics and data schema.
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output, schema=infer_schema.outputs.output)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs.examples,
        schema=infer_schema.outputs.output,
        module_file=module_file)

    # Uses user-provided Python function that implements a model using
    # TensorFlow's Estimators API.
    trainer = Trainer(
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            ai_platform_trainer_executor.Executor),
        module_file=module_file,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_graph=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
        custom_config={'ai_platform_training_args': ai_platform_training_args})

    # Uses TFMA to compute a evaluation statistics over features of a model.
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model=trainer.outputs.output,
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(column_for_slicing=['weekday'])
        ]))

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            ai_platform_pusher_executor.Executor),
        model=trainer.outputs.output,
        model_blessing=model_analyzer.outputs.blessing,
        custom_config={'ai_platform_serving_args': ai_platform_serving_args})

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, infer_schema, validate_stats,
            transform,
            trainer, model_analyzer, pusher
        ],
        # enable_cache=True,
        beam_pipeline_args=beam_pipeline_args)


if __name__ == '__main__':
    # Get settings from environment variables
    _pipeline_name = os.environ.get('PIPELINE_NAME')
    _project_id = os.environ.get('PROJECT_ID')
    _gcp_region = os.environ.get('GCP_REGION')
    _pipeline_image = os.environ.get('KUBEFLOW_TFX_IMAGE')
    _gcs_data_root_uri = os.environ.get('DATA_ROOT_URI')
    _artifact_store_uri = os.environ.get('ARTIFACT_STORE_URI')
    _runtime_version = os.environ.get('RUNTIME_VERSION')
    _python_version = os.environ.get('PYTHON_VERSION')

    # AI Platform Training settings
    _ai_platform_training_args = {
        'project': _project_id,
        'region': _gcp_region,
        'masterConfig': {
            'imageUri': _pipeline_image,
        }
    }

    # AI Platform Prediction settings
    _ai_platform_serving_args = {
        'model_name': 'model_' + _pipeline_name,
        'project_id': _project_id,
        'runtimeVersion': _runtime_version,
        'pythonVersion': _python_version
    }

    # Dataflow settings.
    _beam_tmp_folder = '{}/beam/tmp'.format(_artifact_store_uri)
    _beam_pipeline_args = [
        '--runner=DataflowRunner',
        '--experiments=shuffle_mode=auto',
        '--project=' + _project_id,
        '--temp_location=' + _beam_tmp_folder,
        '--region=' + _gcp_region,
    ]

    operator_funcs = [
        gcp.use_gcp_secret('user-gcp-sa'),
    ]

    # Known issue https://github.com/tensorflow/tfx/issues/1287
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    metadata_config.grpc_config.grpc_service_host.value = 'metadata-grpc-service'
    metadata_config.grpc_config.grpc_service_port.value = '8080'

    # Compile the pipeline
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        pipeline_operator_funcs=operator_funcs,
        kubeflow_metadata_config=metadata_config,
        tfx_image=_pipeline_image)

    _module_file = 'modules/transform_train.py'
    _pipeline_root = '{}/{}'.format(_artifact_store_uri, _pipeline_name)
    kubeflow_dag_runner.KubeflowDagRunner(
        output_filename=__file__ + '.yaml',
        config=runner_config
    ).run(
        _create__pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_gcs_data_root_uri,
            module_file=_module_file,
            ai_platform_training_args=_ai_platform_training_args,
            ai_platform_serving_args=_ai_platform_serving_args,
            beam_pipeline_args=_beam_pipeline_args))
