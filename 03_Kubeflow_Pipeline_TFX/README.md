# Kubeflow Pipeline TFX
TFX is a platform for building and managing ML workflows in a production environment. TFX provides the following:
- A toolkit for building ML pipelines. TFX pipelines let you orchestrate your ML workflow on several platforms, 
such as: Apache Airflow, Apache Beam, and Kubeflow Pipelines.
- A set of standard components that you can use as a part of a pipeline, or as a part of your ML training script. 
TFX standard components provide proven functionality to help you get started building an ML process easily.
- Libraries which provide the base functionality for many of the standard components. You can use the 
TFX libraries to add this functionality to your own custom components, or use them separately.

A TFX pipeline is a sequence of components that implement an ML pipeline which is specifically designed for scalable, 
high-performance machine learning tasks. That includes modeling, training, serving inference, and managing deployments to online, native mobile, and JavaScript targets.
A TFX pipeline typically includes the following components:
- `ExampleGen` is the initial input component of a pipeline that ingests and optionally splits the input dataset.
- `StatisticsGen` calculates statistics for the dataset.
- `SchemaGen` examines the statistics and creates a data schema.
- `ExampleValidator` looks for anomalies and missing values in the dataset.
- `Transform` performs feature engineering on the dataset.
- `Trainer` trains the model.
- `Tuner` tunes the hyperparameters of the model.
- `Evaluator` performs deep analysis of the training results and helps you validate your exported models, ensuring that they are "good enough" to be pushed to production.
- `InfraValidator` checks the model is actually servable from the infrastructure, and prevents bad model from being pushed.
- `Pusher` deploys the model on a serving infrastructure.
- `BulkInferrer` performs batch processing on a model with unlabelled inference requests.

The two notebooks [TFX Estimatror Walkthrough](TFX_Estimator_Interactive_Walkthrough.ipynb)
and [TFX Keras Walkthrough](TFX_Keras_Interactive_Walkthrough.ipynb) interactively walk through each built-in component of TensorFlow Extended (TFX).
It covers every step in an end-to-end machine learning pipeline, from data ingestion to pushing a model to serving.

Moreover, the following two sub-modules are TFX on Cloud AI Platform Pipeline tutorial to run the TFX example pipeline on Kubeflow. 
TFX components have been containerized to compose the Kubeflow pipeline and the sample illustrates the ability 
to configure the pipeline to read large public dataset and execute training and data processing steps at scale in the cloud.
- [TFX Estimator Pipeline](TFX_Estimator_Kubeflow_Pipeline)
- [TFX Keras Pipeline](TFX_Keras_Kubeflow_Pipeline)

## Kubeflow Pipelines resource considerations
Depending on the requirements of your workload, the default configuration for your 
Kubeflow Pipelines deployment may or may not meet your needs. You can customize your resource configurations 
using `pipeline_operator_funcs` in your call to `KubeflowDagRunnerConfig`. 
`pipeline_operator_funcs` is a list of `OpFunc` items, which transforms all the 
generated `ContainerOp` instances in the KFP pipeline spec which is compiled from `KubeflowDagRunner`.

For example, to configure memory we can use `set_memory_request` to declare the 
amount of memory needed. A typical way to do that is to create a wrapper 
for `set_memory_request` and use it to add to to the list of pipeline `OpFuncs`:
```python
def request_more_memory():
  def _set_memory_spec(container_op):
    container_op.set_memory_request('32G')
  return _set_memory_spec

# Then use this opfunc in KubeflowDagRunner
pipeline_op_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs()
pipeline_op_funcs.append(request_more_memory())
config = KubeflowDagRunnerConfig(
    pipeline_operator_funcs=pipeline_op_funcs,
    ...
)
kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)
```

Similar resource configuration functions include:
- set_memory_limit
- set_cpu_request
- set_cpu_limit
- set_gpu_limit
