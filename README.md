# Kubeflow Pipeline Tutorial 
[Kubeflow Pipelines](https://github.com/kubeflow/pipelines) is a platform for building and deploying portable, scalable machine learning (ML) pipelines or 
workflows based on Docker containers. The `Kubeflow Pipelines` platform consists of:
- A user interface for managing and tracking experiments, jobs, and runs.
- An engine for scheduling multi-step ML workflows.
- An SDK for defining and manipulating pipelines and components.
- Notebooks for interacting with the system using the SDK.

A pipeline is a description of an ML workflow, including all of the components in the workflow and 
how they combine in the form of a graph. The pipeline includes the definition of the inputs (parameters) required to 
run the pipeline and the inputs and outputs of each component. A pipeline component is a self-contained set of user 
code, packaged as a Docker image, that performs one step in the pipeline. For example, a component can be responsible 
for data preprocessing, data transformation, model training, and so on. 

## Content Overview:
In this repository, we designed a series of notebooks to demonstrate and guide you from basics to advance usage of `Kubeflow Pipelines` through
[Python SDK](https://github.com/kubeflow/pipelines/tree/master/sdk/python/kfp). The contents have been grouped into four folders
- [Kubeflow Pipeline Basics](01_Kubeflow_Pipeline_Basics): One concrete use case, i.e., 
[MNIST classification](https://www.tensorflow.org/tutorials/quickstart/beginner), is used to demonstrate various ways of
authoring a pipeline component aiming for different stages of development;
- [Kubeflow Pipeline Examples](02_Kubeflow_Pipeline_Examples): Complete pipeline examples with combinations of pre-built
and customised components running on various GCP AI and Analytical services;
- [Kubeflow Pipeline TFX](03_Kubeflow_Pipeline_TFX): TFX is a platform for building and managing ML workflows in a production environment, which
can be orchestrated by Kubeflow Pipeline. Notebooks are provided to demonstrate the basic usage of TFX, and complete samples pipelines
are also provided for reference;
- [Kubeflow Pipeline CI/CD Examples](04_Kubeflow_Pipeline_CICD_Examples): MLOps is an ML engineering culture and practice that aims at unifying ML 
development (Dev) and ML operation (Ops). MLOps strongly advocates automation and monitoring at all steps of ML system construction, from integration, testing, 
and releasing to deployment and infrastructure management. CI/CD examples together with Kubeflow Pipeline are provided here.


## Setups Overview:
### Prerequisites
Before you follow the instructions below to deploy your own Kubeflow cluster, you should

- have a [GCP project setup](https://www.kubeflow.org/docs/gke/deploy/project-setup/) for your Kubeflow deployment 
with you having the [owner role](https://cloud.google.com/iam/docs/understanding-roles#primitive_role_definitions) 
for the project and with the following APIs enabled:
    - [Compute Engine API](https://pantheon.corp.google.com/apis/library/compute.googleapis.com)
    - [Kubernetes Engine API](https://pantheon.corp.google.com/apis/library/container.googleapis.com)
    - [Identity and Access Management(IAM) API](https://pantheon.corp.google.com/apis/library/iam.googleapis.com)
    - [Deployment Manager API](https://pantheon.corp.google.com/apis/library/deploymentmanager.googleapis.com)
    - [Cloud Resource Manager API](https://pantheon.corp.google.com/apis/library/cloudresourcemanager.googleapis.com)
    - [Cloud Filestore API](https://pantheon.corp.google.com/apis/library/file.googleapis.com)
    - [AI Platform Training & Prediction API](https://pantheon.corp.google.com/apis/library/ml.googleapis.com)
- have set up [OAuth for Cloud IAP](https://www.kubeflow.org/docs/gke/deploy/oauth-setup/)
- have installed and setup [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- have installed [gcloud-sdk](https://cloud.google.com/sdk/)

### Setup Environment
Deploy Kubeflow Pipeline Service with
- [Kubeflow Full Deployment](00_Kubeflow_Cluster_Setup.ipynb) to help you deploy a full Kubeflow 
deployment through CLI. (The currently version is designed based on kfp 0.7).
- [Kubeflow Pipeline Deployment](00_Kubeflow_Pipeline_Deployment_1.0_ipynb) to help you deploy a standalone Kubeflow Pipeline deployment.

### Running Notebook
Please not that the above configuration is required for notebook service running outside Kubeflow environment. 
And the examples demonstrated are fully tested on notebook service for the following three situations:
- Notebook running on your personal computer
- Notebook on AI Platform, Google Cloud Platform
- Essentially notebook on any environment outside Kubeflow cluster
 
For notebook running inside Kubeflow cluster, for example JupytHub will be deployed together with kubeflow, the 
environment variables, e.g. service account, projects and etc, should have been pre-configured while 
setting up the cluster.

### Regional Artifact Registry
[Artifact Registry](https://cloud.google.com/artifact-registry)
is a single place for your organization to manage container images and language packages (such as Maven and npm). It is fully integrated with Google Cloud’s tooling and runtimes and comes with support for native artifact protocols. More importantly, it supports regional and multi-regional repositories.

#### The steps to create regional Docker repository in Artifact Registry are as follows

- Run the following command to create a new Docker repository named `AF_REGISTRY_NAME` in the location `AF_REGISTRY_LOCATION` with the description "Regional Docker repository". The regional Artifact Registry supports quite a number of regions, e.g., Hong Kong, Taiwan, Singapore, Tokyo in Asia.

```shell
gcloud beta artifacts repositories create $AF_REGISTRY_NAME \
    --repository-format=docker \
    --location=$AF_REGISTRY_LOCATION \
    --project=$PROJECT_ID \
    --description="Regional Docker repository"
```

- Run the following command to verify that your repository was created.

```shell
gcloud beta artifacts repositories list --project=$PROJECT_ID
```

The supported regions can be found [here](https://cloud.google.com/artifact-registry/docs/repo-organize#locations). The repository URI after creation will be
```
{AF_REGISTRY_LOCATION}-docker.pkg.dev/{PROJECT_ID}/{AF_REGISTRY_NAME}/
```
and the image full uri typically will be
```
{AF_REGISTRY_LOCATION}-docker.pkg.dev/{PROJECT_ID}/{AF_REGISTRY_NAME}/{IMAGE_NAME}:{TAG}
```

### Regional Endpoint of AI Platform Prediction
Interacting with AI Platform services, e.g. training and prediction, will require the access of the endpoint. There are two options available, i.e., **global endpoint** and **regional endpoint**:
- When you create a model resource on the global endpoint, you can specify a region for your model. When you create versions within this model and serve predictions, the prediction nodes run in the specified region. 
- When you use a regional endpoint, AI Platform Prediction runs your prediction nodes in the endpoint's region. However, in this case AI Platform Prediction provides additional isolation by running all AI Platform Prediction infrastructure in that region.

For example, if you use the us-east1 region on the global endpoint, your prediction nodes run in us-east1. But the AI Platform Prediction infrastructure managing your resources (routing requests; handling model and version creation, updates, and deletion; etc.) does not necessarily run in us-east1. On the other hand, if you use the europe-west4 regional endpoint, your prediction nodes and all AI Platform Prediction infrastructure run in europe-west4.

Current available regional endpoints are: `us-central1`, `europe-west4` and `asia-east1`. However, **regional endpoints do not currently support AI Platform Training**.

#### Using regional endpoints
```python
from google.api_core.client_options import ClientOptions
from googleapiclient import discovery

endpoint = 'https://REGION-ml.googleapis.com'
client_options = ClientOptions(api_endpoint=endpoint)
ml = discovery.build('ml', 'v1', client_options=client_options)

request_body = { 'name': 'MODEL_NAME' }
request = ml.projects().models().create(parent='projects/PROJECT_ID',
    body=request_body)

response = request.execute()
print(response)
```

### Using regional endpoints of Kubeflow Pipeline GCP components
The pre-built reusable [GCP Kubeflow Pipeline components](https://github.com/kubeflow/pipelines/tree/master/components/gcp/ml_engine) don't provide regional 
endpoints capabilities. We have provided a customized version [here](https://github.com/luotigerlsx/pipelines/tree/master/components/gcp). 
To build and use the customized components, please follow the steps:
- Clone the source code

```shell
git clone git@github.com:luotigerlsx/pipelines.git
```

- Navigate to `pipelines/components` and find the `build_image.sh`. Replace the container registry address accordingly
    - To use global Container Registry, replace `asia.gcr.io/${PROJECT_ID}/` to `gcr.io/${PROJECT_ID}/`
    - To use regional Container Registry, replace `asia.gcr.io/${PROJECT_ID}/` to `Region.gcr.io/${PROJECT_ID}/`. Region can be: `us`, `eu` or `asia`
    - To use regional Artifact Registry, replace `asia.gcr.io/${PROJECT_ID}/` to `${AF_REGISTRY_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AF_REGISTRY_NAME}/`

- Navigate to `pipelines/components/gcp/container`. Run

```shell
bash build_image.sh -p ${PROJECT_ID}
```

- Navigate to `pipelines/components/gcp/ml_engine/deploy` and modify `component.yaml`

```yaml
implementation:
  container:
    image: {newly_built_image}
    args: [
```

- Save the modified `component.yaml` to a location that can be accessed by the ML taks through

```python
mlengine_deploy_op = comp.load_component_from_url(
    '{internal_accessable_address}/component.yaml')

```

## Enabling GPU and TPU
### Configure ContainerOp to consume GPUs

After enabling the GPU, the Kubeflow setup script installs a default GPU pool with type nvidia-tesla-k80 with auto-scaling enabled.
The following code consumes 2 GPUs in a ContainerOp.

```python
import kfp.dsl as dsl
gpu_op = dsl.ContainerOp(name='gpu-op', ...).set_gpu_limit(2)
```

The code above will be compiled into Kubernetes Pod spec:

```yaml
container:
  ...
  resources:
    limits:
      nvidia.com/gpu: "2"
```

If the cluster has multiple node pools with different GPU types, you can specify the GPU type by the following code.

```python
import kfp.dsl as dsl
gpu_op = dsl.ContainerOp(name='gpu-op', ...).set_gpu_limit(2)
gpu_op.add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-p4')
```

The code above will be compiled into Kubernetes Pod spec:


```yaml
container:
  ...
  resources:
    limits:
      nvidia.com/gpu: "2"
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-tesla-p4
```

Check the [GKE GPU guide](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus) to learn more about GPU settings. 

### Configure ContainerOp to consume TPUs

Use the following code to configure ContainerOp to consume TPUs on GKE:

```python
import kfp.dsl as dsl
import kfp.gcp as gcp
tpu_op = dsl.ContainerOp(name='tpu-op', ...).apply(gcp.use_tpu(
  tpu_cores = 8, tpu_resource = 'v2', tf_version = '1.12'))
```

The above code uses 8 v2 TPUs with TF version to be 1.12. The code above will be compiled into Kubernetes Pod spec:

```yaml
container:
  ...
  resources:
    limits:
      cloud-tpus.google.com/v2: "8"
  metadata:
    annotations:
      tf-version.cloud-tpus.google.com: "1.12"
```

## Contributors
- [Shixin Luo](https://github.com/luotigerlsx)
- [Tommy Siu](https://github.com/tommysiu)
- [Kumar Saurabh](https://github.com/saurabh24292)