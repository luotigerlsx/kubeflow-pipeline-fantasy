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
from tfx.components.model_validator.component import ModelValidator
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
from tfx.proto import pusher_pb2

from tfx.utils.dsl_utils import external_input

import tensorflow_model_analysis as tfma


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
        custom_config={ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                           ai_platform_training_args})

    # Uses TFMA to compute a evaluation statistics over features of a model.

    eval_config = tfma.EvalConfig(
        model_specs=[
            # This assumes a serving model with signature 'serving_default'. If
            # using estimator based EvalSavedModel, add signature_name='eval' and
            # remove the label_key. Note, if using a TFLite model, then you must set
            # model_type='tf_lite'.
            tfma.ModelSpec(signature_name='eval')
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                # The metrics added here are in addition to those saved with the
                # model (assuming either a keras model or EvalSavedModel is used).
                # Any metrics added into the saved model (for example using
                # model.compile(..., metrics=[...]), etc) will be computed
                # automatically.
                # metrics=[
                #     tfma.MetricConfig(class_name='ExampleCount')
                # ],
                # To add validation thresholds for metrics saved with the model,
                # add them keyed by metric name to the thresholds map.
                thresholds={
                    "accuracy": tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.1}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10}))
                }
            )
        ],
        slicing_specs=[
            # An empty slice spec means the overall slice, i.e. the whole dataset.
            tfma.SlicingSpec(),
            # Data can be sliced along a feature column. In this case, data is
            # sliced along feature column trip_start_hour.
            tfma.SlicingSpec(feature_keys=['weekday'])
        ])

    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model=trainer.outputs.output,
        eval_config=eval_config)

    # Performs quality validation of a candidate model (compared to a baseline).
    # model_validator = ModelValidator(
    #     examples=example_gen.outputs.examples, model=trainer.outputs.output)

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model=trainer.outputs.output,
        model_blessing=model_analyzer.outputs.blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=os.path.join(pipeline_root, 'serving_model'))),
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            ai_platform_pusher_executor.Executor),
        custom_config={ai_platform_pusher_executor.SERVING_ARGS_KEY:
                           ai_platform_serving_args})

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
