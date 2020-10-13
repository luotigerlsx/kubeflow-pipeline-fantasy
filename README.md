# Kubeflow Pipeline Tutorial 
[`Kubeflow Pipelines`](https://github.com/kubeflow/pipelines) is a platform for building and deploying portable, scalable machine learning (ML) pipelines or 
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
In this tutorial, we designed a series of notebooks to demonstrate how to interact with `Kubeflow Pipelines` through
[Python SDK](https://github.com/kubeflow/pipelines/tree/master/sdk/python/kfp). In particular
- [Kubeflow Cluster Setup](00_Kubeflow_Cluster_Setup.ipynb): this notebook helps you deploy a Kubeflow 
cluster through CLI. The [UI](https://www.kubeflow.org/docs/gke/deploy/deploy-ui/) method of deploying a Kubeflow 
cluster does not support Kubeflow v0.7 yet.

Then, notebooks 01-04 use one concrete use case, i.e., 
[MINIST classification](https://www.tensorflow.org/tutorials/quickstart/beginner), to demonstrate different ways of
authoring a pipeline component: 
- [Lightweight Python Components](01_Kubeflow_Pipeline_Basics/01_Lightweight_Python_Components.ipynb): this notebook demonstrates how to build a 
component through defining a stand-alone python function and then calling `kfp.components.func_to_container_op(func)` to 
convert, which can be used in a pipeline.

- [Local Development with Docker Image Components](01_Kubeflow_Pipeline_Basics/02_Local_Development_with_Docker_Image_Components.ipynb): this 
notebook guides you on creating a pipeline component with `kfp.components.ContainerOp` from an existing Docker image 
which should contain the program to perform the task required in a particular step of your ML workflow.

- [Reusable Components](01_Kubeflow_Pipeline_Basics/03_Reusable_Components.ipynb): this notebook describes the manual way of writing a full 
component program (in any language) and a component definition for it. Below is a summary of the steps involved in 
creating and using a component.
    - Write the program that contains your component’s logic. The program must use files and command-line arguments 
    to pass data to and from the component.
    - Containerize the program.
    - Write a component specification in YAML format that describes the component for the Kubeflow Pipelines system.
    - Use the Kubeflow Pipelines SDK to load your component, use it in a pipeline and run that pipeline.

- [Reusable and Pre-build Components as Pipeline](01_Kubeflow_Pipeline_Basics/04_Reusable_and_Pre-build_Components_as_Pipeline.ipynb): this 
notebook combines our built components, together with a pre-build GCP AI Platform components 
and a lightweight component to compose a pipeline with three steps.
    - Train a MINIST model and export it to GCS
    - Deploy the exported Tensorflow model on AI Platform prediction service
    - Test the deployment by calling the end point with test data

We have also put together some more examples to demonstrate
- [Pipeline with AI Platform and GCP Service](02_Kubeflow_Pipeline_Examples/Pipeline_with_AI_Platform_and_GCP_Service.ipynb): this notebook demonstrates orchestrating model training and deployment with Kubeflow Pipelines (KFP) and Cloud AI Platform. In particular, you will develop, deploy, and run a KFP pipeline that orchestrates BigQuery and Cloud AI Platform services to train a scikit-learn model. The pipeline uses:
    - Pre-build components. The pipeline uses the following pre-build components that are included with KFP distribution:
        - [BigQuery query component](https://github.com/kubeflow/pipelines/tree/0.1.36/components/gcp/bigquery/query)
        - [AI Platform Training component](https://github.com/kubeflow/pipelines/tree/0.1.36/components/gcp/ml_engine/train)
        - [AI Platform Deploy component](https://github.com/kubeflow/pipelines/tree/0.1.36/components/gcp/ml_engine/deploy)
    - Custom components. The pipeline uses two custom helper components that encapsulate functionality not available in any of the pre-build components. The components are implemented using the KFP SDK's [Lightweight Python Components](https://www.kubeflow.org/docs/pipelines/sdk/lightweight-python-components/) mechanism. The code for the components is in the `helper_components.py` file:
        - **Retrieve Best Run**. This component retrieves the tuning metric and hyperparameter values for the best run of the AI Platform Training hyperparameter tuning job.
        - **Evaluate Model**. This component evaluates the *sklearn* trained model using a provided metric and a testing dataset. 
        the pipeline fails or not.

- TFX Pipeline with AI Platform
    - [TFX Interactive Walkthrough](03_TFX_Kubeflow_Pipeline/TFX_Interactive_Walkthrough.ipynb)
    - [TFX (Estimator) and Kubeflow Pipeline](03_TFX_Kubeflow_Pipeline/TFX_Estimator_Kubeflow_Pipeline)
    - [TFX (Keras) and Kubeflow Pipeline](03_TFX_Kubeflow_Pipeline/TFX_Keras_Kubeflow_Pipeline)


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
* Deploy a kubeflow cluster through [CLI](https://www.kubeflow.org/docs/gke/deploy/deploy-cli/)
    - Download and install kfctl
    - Create user credentials
    - Setup environment variables
    - `NOTE` : The [UI](https://www.kubeflow.org/docs/gke/deploy/deploy-ui/) method of deploying a Kubeflow 
    cluster does not support Kubeflow v0.7 yet

* Create service account
```bash
export SA_NAME = [service account name]
gcloud iam service-accounts create ${SA_NAME}
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --role 'roles/owner'
gcloud iam service-accounts keys create ~/key.json \
    --iam-account ${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com
```

* Install the lastest version of kfp
```bash
pip3 install kfp --upgrade --user
```

* Deploy kubefow
```bash
mkdir -p ${KF_DIR}
cd $kf_dir
kfctl apply -V -f ${CONFIG_URI}
```
### Running Notebook
Please not that the above configuration is required for notebook service running outside Kubeflow environment. 
And the examples demonstrated are fully tested on notebook service for the following three situations:
- Notebook running on your personal computer
- Notebook on AI Platform, Google Cloud Platform
- Essentially notebook on any environment outside Kubeflow cluster
 
For notebook running inside Kubeflow cluster, for example JupytHub will be deployed together with kubeflow, the 
environemt variables, e.g. service account, projects and etc, should have been pre-configured while 
setting up the cluster.

## Regional Artifact Registry
[Artifact Registry](https://cloud.google.com/artifact-registry)
is a single place for your organization to manage container images and language packages (such as Maven and npm). It is fully integrated with Google Cloud’s tooling and runtimes and comes with support for native artifact protocols. More importantly, it supports regional and multi-regional repositories.

### The steps to create regional Docker repository in Artifact Registry are as follows

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

## Regional Container Registry
[Artifact Registry](https://cloud.google.com/artifact-registry)
is a single place for your organization to manage container images and language packages (such as Maven and npm). It is fully integrated with Google Cloud’s tooling and runtimes and comes with support for native artifact protocols. More importantly, it supports regional and multi-regional repositories.

### The steps to create regional Docker repository is as follows

- Run the following command to create a new Docker repository named `AF_REGISTRY_NAME` in the location `AF_REGISTRY_LOCATION` with the description "Regional Docker repository".

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

## Regional Endpoint of AI Platform Prediction
Interacting with AI Platform services, e.g. training and prediction, will require the access of the endpoint. There are two options available, i.e., **global endpoint** and **regional endpoint**:
- When you create a model resource on the global endpoint, you can specify a region for your model. When you create versions within this model and serve predictions, the prediction nodes run in the specified region. 
- When you use a regional endpoint, AI Platform Prediction runs your prediction nodes in the endpoint's region. However, in this case AI Platform Prediction provides additional isolation by running all AI Platform Prediction infrastructure in that region.

For example, if you use the us-east1 region on the global endpoint, your prediction nodes run in us-east1. But the AI Platform Prediction infrastructure managing your resources (routing requests; handling model and version creation, updates, and deletion; etc.) does not necessarily run in us-east1. On the other hand, if you use the europe-west4 regional endpoint, your prediction nodes and all AI Platform Prediction infrastructure run in europe-west4.

Current available regional endpoints are: `us-central1`, `europe-west4` and `asia-east1`. However, **regional endpoints do not currently support AI Platform Training**.

### Using regional endpoints
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
The pre-built reusable [GCP Kubeflow Pipeline components](https://github.com/kubeflow/pipelines/tree/master/components/gcp/ml_engine) don't provide regional endpoints capabilities. We have provided a customized version [here](https://github.com/luotigerlsx/pipelines/tree/master/components/gcp). To build and use the customized components, please follow the steps:
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

## Contributors
- [Shixin Luo](https://github.com/luotigerlsx)
- [Tommy Siu](https://github.com/tommysiu)
- [Kumar Saurabh](https://github.com/saurabh24292)