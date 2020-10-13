# CI/CD for KFP pipelines

In this lab you will walk through authoring of a **Cloud Build** CI/CD workflow that automatically builds and deploys a KFP pipeline. You will also integrate your workflow with **GitHub** by setting up a trigger that starts the  workflow when a new tag is applied to the **GitHub** repo hosting the pipeline's code.

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
|_PIPELINE_FOLDER|06_CICD_Pipeline_with_AI_Platform/pipeline|
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


