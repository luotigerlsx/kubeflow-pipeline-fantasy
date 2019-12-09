# Kubeflow Pipeline Tutorial 
A pipeline is a description of an ML workflow, including all of the components in the workflow and how they combine 
in the form of a graph. The pipeline includes the definition of the inputs (parameters) required to run the pipeline 
and the inputs and outputs of each component. A pipeline component is a self-contained set of user code, packaged as 
a Docker image, that performs one step in the pipeline. For example, a component can be responsible for data 
preprocessing, data transformation, model training, and so on. 

Kubeflow Pipelines is a platform for building and deploying portable, scalable machine learning (ML) workflows based on 
Docker containers. The `Kubeflow Pipelines` platform consists of:
- A user interface (UI) for managing and tracking experiments, jobs, and runs.
- An engine for scheduling multi-step ML workflows.
- An SDK for defining and manipulating pipelines and components.
- Notebooks for interacting with the system using the SDK.

## Content Overview:
In this tutorial, we provided a series of notebooks to demonstrate how to interact with Kubeflow Pipelines with
[Python SDK](https://github.com/kubeflow/pipelines/tree/master/sdk/python/kfp). In particular
- [00 Kubeflow Cluster Setup](00_Kubeflow_Cluster_Setup.ipynb): this notebook helps you deploy Kubeflow cluster through CLI.

Then, notebooks 01-04 make use of one concrete use case, i.e., 
[MINIST classification](https://www.tensorflow.org/tutorials/quickstart/beginner), to demonstrate different ways of
authoring a pipeline component: 
- [01 Lightweight Python Components](01_Lightweight_Python_Components.ipynb): this notebook demonstrates how to build a 
component just define a stand-alone python function and then call `kfp.components.func_to_container_op(func)` to 
convert it to a component that can be used in a pipeline.

- [02 Local Development with Docker Image Components](02_Local_Development_with_Docker_Image_Components.ipynb): this 
notebook guides you on creating a pipeline component with `kfp.components.ContainerOp` from an existing Docker image 
which should contain the program to perform the task 
 required in a particular step of your ML workflow.

- [03 Reusable Components](03_Reusable_Components.ipynb): this notebook describes the manual way of writing a full 
component program (in any language) and a component definition for it. Below is a summary of the steps involved in 
creating and using a component.
    - Write the program that contains your component’s logic. The program must use files and command-line arguments 
    to pass data to and from the component.
    - Containerize the program.
    - Write a component specification in YAML format that describes the component for the Kubeflow Pipelines system.
    - Use the Kubeflow Pipelines SDK to load your component, use it in a pipeline and run that pipeline.

- [04 Reusable and Pre-build Components as Pipeline](04_Reusable_and_Pre-build_Components_as_Pipeline.ipynb): this 
notebook combines our built components together with a pre-build components and a lightweight component to compose a 
pipeline with three steps.
    - Train a MINIST model and export to GCS
    - Deploy the exported Tensorflow model on AI Platform
    - Test the deployment by calling the end point

The last two notebooks make use of pre-build reusable components to build two popular pipelines:
- [05 XGBoost on Spark Pipeline](05_XGboost_on_Spark_Pipeline.ipynb): this notebook demonstrates building a machine 
learning pipeling with spark and XGBoost. The pipeline
    - starts by creating an Google DataProc cluster, and then running analysis, transformation, distributed 
    training and prediction in the created cluster.
    - Then a single node confusion-matrix and ROC aggregator is used (for classification case) to provide the confusion 
    matrix data, and ROC data to the front end, respectively.
    - Finally, a delete cluster operation runs to destroy the cluster it creates in the beginning. 
    The delete cluster operation is used as an exit handler, meaning it will run regardless of whether 
    the pipeline fails or not.

- [06 TFX Pipeline from Pre-build Components](06_TFX_Pipeline_from_Pre-build_Components.ipynb): this notebook demonstrates 
the TFX capabilities at scale. The pipeline uses a public BigQuery dataset and uses GCP services to preprocess data 
(Dataflow) and train the model (Cloud ML Engine). The model is then deployed to Kubeflow for prediction service.


## Setups Overview:
### Prerequisites
Before you follow the instructions below to deploy your own kubeflow cluster, you should

- have a [GCP project setup](https://www.kubeflow.org/docs/gke/deploy/project-setup/) for your Kubeflow Deployment 
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