# Orchestrating model training and deployment with TFX and Cloud AI Platform

In this lab you will develop, deploy and run a TFX pipeline that uses Kubeflow Pipelines for orchestration and Cloud Dataflow and Cloud AI Platform for data processing, training, and deployment:


## Lab scenario

You will be working with the [Covertype Data Set](https://github.com/jarokaz/mlops-labs/blob/master/datasets/covertype/README.md) dataset. 

The pipeline implements a typical TFX workflow as depicted on the below diagram:

![Lab 14 diagram](/images/lab-14-diagram.png).

The source data in a CSV file format is in the GCS bucket.

The TFX `ExampleGen`, `StatisticsGen`, `ExampleValidator`, `SchemaGen`, `Transform`, and `Evaluator` components use Cloud Dataflow as an execution engine. The `Trainer` and `Pusher` components use AI Platform Training and Prediction services.


## Lab setup

### AI Platform Notebook and KFP environment
Before proceeding with the lab, you must set up an **AI Platform Notebooks** instance and a **KFP** environment.

## Lab Exercises

You will use a JupyterLab terminal terminal as the primary interface during the lab. Before proceeding with the lab exercises configure a set of environment variables that reflect your lab environment. If you used the default settings during the environment setup you don't need to modify the below commands. If you provided custom values for PREFIX, REGION, ZONE, or NAMESPACE update the commands accordingly:
```
export PROJECT_ID=$(gcloud config get-value core/project)
export PREFIX=$PROJECT_ID
export NAMESPACE=kubeflow
export GCP_REGION=us-central1
export ZONE=us-central1-a
export ARTIFACT_STORE_URI=gs://$PREFIX-artifact-store
export GCS_STAGING_PATH=${ARTIFACT_STORE_URI}/staging
export GKE_CLUSTER_NAME=$PREFIX-cluster
export DATA_ROOT_URI=gs://workshop-datasets/covertype/full

gcloud container clusters get-credentials $GKE_CLUSTER_NAME --zone $ZONE
export INVERSE_PROXY_HOSTNAME=$(kubectl describe configmap inverse-proxy-config -n $NAMESPACE | grep "googleusercontent.com")
```

Follow the instructor who will walk you through the lab. The high level summary of the lab flow is as follows:

### Understanding the pipeline's DSL.

The pipeline uses a custom docker image, which is a derivative of the [tensorflow/tfx:0.15.0 image](https://hub.docker.com/r/tensorflow/tfx), as a runtime execution environment for the pipeline's components. The same image is also used as a training image used by **AI Platform Training**

The base `tfx` image includes TFX v0.15 and TensorFlow v2.0. The custom image modifies the base image by downgrading to TensorFlow v1.15 and adding the `modules` folder with the `transform_train.py` file that contains data transformation and training code used by the pipeline's `Transform` and `Train` components.

The pipeline needs to use v1.15 of TensorFlow as the AI Platform Prediction service, which is used as a deployment target, does not yet support v2.0 of TensorFlow.

### Building and deploying the pipeline
#### Creating the custom docker image
The first step is to build the custom docker image and push it to your project's **Container Registry**. You will use **Cloud Build** to build the image.

1. Create the Dockerfile describing the custom image
```
cat > Dockerfile << EOF
FROM tensorflow/tfx:0.15.0
RUN pip install -U tensorflow-serving-api==1.15 tensorflow==1.15
RUN mkdir modules
COPY  transform_train.py modules/
EOF
```

2. Submit the **Cloud Build** job
```
IMAGE_NAME=tfx-image
TAG=latest
export TFX_IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

gcloud builds submit --timeout 15m --tag ${TFX_IMAGE} .
```

#### Compiling and uploading the pipeline to the KFP environment
The pipeline's DSL retrieves the settings controlling how the pipeline is compiled from the environment variables. In addition to the environment settings configured before, you need to set a few additional pipeline specific settings:

```
export PIPELINE_NAME=tfx_covertype_classifier_training
export RUNTIME_VERSION=1.15
export PYTHON_VERSION=3.7

tfx pipeline create --engine kubeflow --pipeline_path pipeline_dsl.py --endpoint $INVERSE_PROXY_HOSTNAME
```


The `tfx pipeline create` command compiles the pipeline's DSL into the KFP package file - `tfx_covertype_classifier_training.tar.gz` and uploads the package to the KFP environment. The package file contains the description of the pipeline in the YAML format. If you want to examine the file, extract from the tarball file and use the JupyterLab editor.

```
tar xvf tfx_covertype_classifier_training.tar.gz
```

The name of the extracted file is `pipeline.yaml`.


### Submitting and monitoring pipeline runs

After the pipeline has been deployed, you can trigger and monitor pipeline runs using **TFX CLI** or **KFP UI**.

To submit the pipeline run using **TFX CLI**:
```
tfx run create --pipeline_name tfx_covertype_classifier_training --endpoint $INVERSE_PROXY_HOSTNAME
```

To list all the active runs of the pipeline:
```
tfx run list --pipeline_name tfx_covertype_classifier_training --endpoint $INVERSE_PROXY_HOSTNAME
```

To retrieve the status of a given run:
```
tfx run status --pipeline_name tfx_covertype_classifier_training --run_id [YOUR_RUN_ID] --endpoint $INVERSE_PROXY_HOSTNAME
```
 To terminate a run:
 ```
 tfx run terminate --run_id [YOUR_RUN_ID] --endpoint $INVERSE_PROXY_HOSTNAME
 ```


