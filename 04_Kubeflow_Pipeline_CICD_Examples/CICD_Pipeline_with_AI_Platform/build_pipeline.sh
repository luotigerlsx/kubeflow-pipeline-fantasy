#!/bin/bash
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
#
# Submits a Cloud Build job that builds and deploys
# the pipelines and pipelines components 
# export GOOGLE_APPLICATION_CREDENTIALS=/Users/luoshixin/LocalDevelop/kubeflow-pipeline/kubeflow-pipeline/kubeflow-pipeline-fantasy.json

PROJECT_ID=$(gcloud config get-value core/project)

# Build the base image with kfp-cli
KFP_IMAGE_NAME=kfp-cli
TAG=latest
KFP_IMAGE_URI="gcr.io/${PROJECT_ID}/${KFP_IMAGE_NAME}:${TAG}"

gcloud builds submit --tag ${KFP_IMAGE_URI} .

# Start the CICD pipeline
# Currently this only works for kubeflow deployment without IAP
# Or kubeflow pipeline deployment with inverse_proxy. Because kfp-cli sdk
# cannot handle iap quietly.
SUBSTITUTIONS=\
_INVERTING_PROXY_HOST=https://69a95965149a4145-dot-asia-east1.pipelines.googleusercontent.com,\
_TRAINER_IMAGE_NAME=trainer_image,\
_BASE_IMAGE_NAME=base_image,\
TAG_NAME=test,\
_PIPELINE_FOLDER=CICD_Pipeline_with_AI_Platform/pipeline,\
_PIPELINE_DSL=covertype_training_pipeline.py,\
_PIPELINE_PACKAGE=covertype_training_pipeline.yaml,\
_PIPELINE_NAME=covertype_training_deployment,\
_RUNTIME_VERSION=1.14,\
_PYTHON_VERSION=3.5,\
_COMPONENT_URL_SEARCH_PREFIX=https://raw.githubusercontent.com/kubeflow/pipelines/0.1.36/components/gcp/

gcloud builds submit .. --config cloudbuild.yaml --substitutions $SUBSTITUTIONS
