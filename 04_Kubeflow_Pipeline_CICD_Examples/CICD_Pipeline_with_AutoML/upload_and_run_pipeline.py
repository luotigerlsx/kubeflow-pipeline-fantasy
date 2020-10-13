# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper function to upload the KFP pipeline and run it optionally."""

import logging
import time

import fire
import kfp

logging.basicConfig(level=logging.INFO)


def upload_and_run_pipeline(endpoint, pipeline_name, pipeline_version, pipeline_package,
                            experiment_name, run=False, params=None):
    client = kfp.Client(endpoint)

    logging.info('Getting existing pipeline by name')
    try:
        pipeline_id = client.get_pipeline_id(pipeline_name)
    except TypeError:
        logging.info('Pipeline does not exist, upload pipeline package')
        response = client.upload_pipeline(pipeline_package_path=pipeline_package, pipeline_name=pipeline_name)
        pipeline_id = response.id

    logging.info('Pipeline ID: {}'.format(pipeline_id))
    logging.info('Upload pipeline version: {}'.format(pipeline_version))
    version = client.upload_pipeline_version(pipeline_package, pipeline_version, pipeline_id=pipeline_id)
    logging.info('Pipeline version with ID {} is created'.format(version.id))

    if run:
        logging.info('Run pipeline in experiment {}'.format(experiment_name))
        experiment = client.get_experiment(experiment_name=experiment_name)

        job_name = '{}_{}'.format(pipeline_name, int(time.time() * 1000.0))
        client.run_pipeline(experiment_id=experiment.id, job_name=job_name, pipeline_id=pipeline_id,
                            version_id=version.id, params=params)


if __name__ == '__main__':
    fire.Fire(upload_and_run_pipeline)
