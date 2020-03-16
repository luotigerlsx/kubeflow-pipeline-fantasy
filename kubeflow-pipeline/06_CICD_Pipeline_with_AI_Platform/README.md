# CI/CD for KFP pipelines

In this lab you will walk through authoring of a **Cloud Build** CI/CD workflow that automatically builds and deploys a KFP pipeline. You will also integrate your workflow with **GitHub** by setting up a trigger that starts the  workflow when a new tag is applied to the **GitHub** repo hosting the pipeline's code.

## Lab scenario

This lab uses the KFP DSL and KFP components developed in `lab-12-kfp-pipeline`.

## Lab setup

This lab requires the same setup as `lab-12-kfp-pipeline`. If you completed `lab-12-kfp-pipeline` you are ready to go and you can skip to the **Lab Exercises** section.

### AI Platform Notebook configuration
Before proceeding with the lab, you must set up the lab environment.

### Lab dataset
This lab uses the [Covertype Dataset](../datasets/covertype/README.md). The pipeline developed in the lab sources the dataset from BigQuery. Before proceeding with the lab upload the dataset to BigQuery:

1. Open new terminal in you **JupyterLab**

2. Create the BigQuery dataset and upload the Cover Type csv file.
```
PROJECT_ID=[YOUR_PROJECT_ID]

DATASET_LOCATION=US
DATASET_ID=covertype_dataset
TABLE_ID=covertype
DATA_SOURCE=gs://workshop-datasets/covertype/full/dataset.csv
SCHEMA=Elevation:INTEGER,\
Aspect:INTEGER,\
Slope:INTEGER,\
Horizontal_Distance_To_Hydrology:INTEGER,\
Vertical_Distance_To_Hydrology:INTEGER,\
Horizontal_Distance_To_Roadways:INTEGER,\
Hillshade_9am:INTEGER,\
Hillshade_Noon:INTEGER,\
Hillshade_3pm:INTEGER,\
Horizontal_Distance_To_Fire_Points:INTEGER,\
Wilderness_Area:STRING,\
Soil_Type:STRING,\
Cover_Type:INTEGER

bq --location=$DATASET_LOCATION --project_id=$PROJECT_ID mk --dataset $DATASET_ID

bq --project_id=$PROJECT_ID --dataset_id=$DATASET_ID load \
--source_format=CSV \
--skip_leading_rows=1 \
--replace \
$TABLE_ID \
$DATA_SOURCE \
$SCHEMA
```

## Lab Exercises

During this lab, you will mostly work in a JupyterLab terminal. Before proceeding with the lab exercises configure a set of environment variables that reflect your lab environment. If you used the default settings during the environment setup you don't need to modify the below commands. If you provided custom values for PREFIX, ZONE, or NAMESPACE update the commands accordingly:

```
export PROJECT_ID=$(gcloud config get-value core/project)
export PREFIX=$PROJECT_ID
export ZONE=us-central1-a
export GKE_CLUSTER_NAME=$PREFIX-cluster

gcloud container clusters get-credentials $GKE_CLUSTER_NAME --zone $ZONE
export INVERSE_PROXY_HOSTNAME=$(kubectl describe configmap inverse-proxy-config -n $NAMESPACE | grep "googleusercontent.com")

```

Follow the instructor who will walk you through the lab. The high level summary of the lab exercises is as follows.

###  Authoring the CI/CD workflow that builds and deploys a KFP  pipeline

In this exercise you walk-through authoring a **Cloud Build** CI/CD workflow that automatically builds and deploys a KFP pipeline. 

The CI/CD workflow automates the steps you walked through manually during `lab-12-kfp-pipeline`:
1. Builds the trainer image
1. Builds the base image for custom components
1. Compiles the pipeline
1. Uploads the pipeline to the KFP environment
1. Pushes the trainer and base images to your project's **Container Registry**

The **Cloud Build** workflow configuration uses both standard and custom [Cloud Build builders](https://cloud.google.com/cloud-build/docs/cloud-builders). The custom builder encapsulates **KFP CLI**. 

*The current version of the lab has been developed and tested with v1.36 of KFP. There is a number of issues with post 1.36 versions of KFP that prevent us from upgrading to the newer version of KFP. KFP v1.36 does not have support for pipeline versions. As an interim measure, the **Cloud Build**  workflow appends `$TAG_NAME` default substitution to the name of the pipeline to designate a pipeline version.*




#### Creating KFP CLI builder

To create a **Cloud Build** custom builder that encapsulates KFP CLI.

1. Create the Dockerfile describing the KFP CLI builder
```
cat > Dockerfile << EOF
FROM gcr.io/deeplearning-platform-release/base-cpu
RUN pip install https://storage.googleapis.com/ml-pipeline/release/0.1.36/kfp.tar.gz 

ENTRYPOINT ["/bin/bash"]
EOF
```

2. Build the image and push it to your project's Container Registry. 
```
IMAGE_NAME=kfp-cli
TAG=latest
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

gcloud builds submit --timeout 15m --tag ${IMAGE_URI} .
```

#### Manually triggering CI/CD runs

To manually trigger the CI/CD run :

1. Update the `build_pipeline.sh` script  with your KFP inverting proxy host. 
```
echo $INVERSE_PROXY_HOSTNAME
```
2. Update the Dockerfile files in the  `pipeline/base_image` and `pipeline/trainer_image` folders so they get the base image from your Container Registry.
3. Start the run:
```
./build_pipeline.sh
```
### Setting up GitHub integration
In this exercise you integrate your CI/CD workflow with **GitHub**, using [Cloud Build GitHub App](https://github.com/marketplace/google-cloud-build). 
You will set up a trigger that starts the CI/CD workflow when a new tag is applied to the **GitHub** repo managing the KFP pipeline source code. You will use a fork of this repo as your source GitHub repository.

1. [Follow the GitHub documentation](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) to fork this repo
2. Connect the fork you created in the previous step to your Google Cloud project and create a trigger following the steps in the [Creating GitHub app trigger](https://cloud.google.com/cloud-build/docs/create-github-app-triggers) article. Use the following values on the **Edit trigger** form:

|Field|Value|
|-----|-----|
|Name|[YOUR TRIGGER NAME]|
|Description|[YOUR TRIGGER DESCRIPTION]|
|Trigger type| Tag|
|Tag (regex)|.\*|
|Build Configuration|Cloud Build configuration file (yaml or json)|
|Cloud Build configuration file location|/ labs/lab-13-kfp-cicd/cloudbuild.yaml|


Use the following values for the substitution variables:

|Variable|Value|
|--------|-----|
|_BASE_IMAGE_NAME|base_image|
|_COMPONENT_URL_SEARCH_PREFIX|https://raw.githubusercontent.com/kubeflow/pipelines/0.1.36/components/gcp/|
|_INVERTING_PROXY_HOST|[Your inverting proxy host]|
|_PIPELINE_DSL|covertype_training_pipeline.py|
|_PIPELINE_FOLDER|labs/lab-13-kfp-cicd/pipeline|
|_PIPELINE_NAME|covertype_training_deployment|
|_PIPELINE_PACKAGE|covertype_training_pipeline.yaml|
|_PYTHON_VERSION|3.5|
|_RUNTIME_VERSION|1.14|
|_TRAINER_IMAGE_NAME|trainer_image|

3. In your fork master branch, update the Dockerfile files in the pipeline/base_image and pipeline/trainer_image folders so they get the base image from your Container Registry.

4. To start an automated build [create a new release of the repo in GitHub](https://help.github.com/en/github/administering-a-repository/creating-releases). Alternatively, you can start the build by applying a tag using `git`. 
```
git tag [TAG NAME]
git push origin --tags
```


