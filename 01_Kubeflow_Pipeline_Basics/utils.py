#!/usr/bin/env python3

import json
import tarfile
from base64 import b64decode
from io import BytesIO

import kfp


def get_node_id(*, run_id: str, component_name: str, client: kfp.Client):
  run = client.runs.get_run(run_id)
  workflow = json.loads(run.pipeline_runtime.workflow_manifest)
  nodes = workflow["status"]["nodes"]
  for node_id, node_info in nodes.items():
    if node_info["displayName"] == component_name:
      return node_id
  else:
    raise RuntimeError(
      f"Unable to find node_id for Component '{component_name}'")


def get_artifact(*, run_id: str, node_id: str, artifact_name: str,
                 client: kfp.Client):
  artifact = client.runs.read_artifact(run_id, node_id, artifact_name)
  # Artifacts are returned as base64-encoded .tar.gz strings
  data = b64decode(artifact.data)
  io_buffer = BytesIO()
  io_buffer.write(data)
  io_buffer.seek(0)
  data = None
  with tarfile.open(fileobj=io_buffer) as tar:
    member_names = tar.getnames()
    if len(member_names) == 1:
      data = tar.extractfile(member_names[0]).read().decode('utf-8')
    else:
      # Is it possible for KFP artifacts to have multiple members?
      data = {}
      for member_name in member_names:
        data[member_name] = tar.extractfile(member_name).read().decode('utf-8')
  return data


if __name__ == "__main__":
  run_id = "e498b0da-036e-4e81-84e9-6e9c6e64960b"
  component_name = "my-component"
  # For an output variable named "output_data"
  artifact_name = "my-component-output_data"

  client = kfp.Client()
  node_id = get_node_id(run_id=run_id, component_name=component_name,
                        client=client)
  artifact = get_artifact(
      run_id=run_id, node_id=node_id, artifact_name=artifact_name,
      client=client,
  )
  # Do something with artifact ...
