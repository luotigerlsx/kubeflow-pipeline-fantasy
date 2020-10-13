#!/bin/bash
# Copyright 2020 Google Inc. All Rights Reserved.
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

SUBSTITUTIONS=\
TAG_NAME=$TAG_NAME,\
_BASE_IMAGE_NAME=base_image,\
_BASE_IMAGE_FOLDER=automl_pipeline/base_image,\
_INVERSE_PROXY_HOSTNAME=$INVERSE_PROXY_HOSTNAME,\
_PIPELINE_FOLDER=automl_pipeline/covertype_pipeline,\
_PIPELINE_DSL=covertype_pipeline.py,\
_PIPELINE_PACKAGE=covertype_pipeline.yaml,\
_PIPELINE_NAME=cover_train_pipeline,\
_EXPERIEMENT_NAME=cover_experiment,\
_COMPONENT_URL_SEARCH_PREFIX=https://raw.githubusercontent.com/kubeflow/pipelines/0.4.0/components/gcp/,\
_PROJECT_ID=$PROJECT_ID,\
_REGION=$REGION,\
_DATASET_ID=$DATASET_ID,\
_MODEL_GCS_DESTINATION=$MODEL_GCS_DESTINATION

gcloud builds submit .. --config cloudbuild.yaml --substitutions ${SUBSTITUTIONS}