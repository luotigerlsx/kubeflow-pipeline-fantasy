# Kubeflow Pipeline Basics 

 _pipeline_ is a description of a machine learning (ML) workflow, including all
of the components of the workflow and how they work together. The pipeline
includes the definition of the inputs (parameters) required to run the pipeline 
and the inputs and outputs of each component.

A pipeline _component_ is an implementation of a pipeline task. A component
represents a step in the workflow. Each component takes one or more inputs and
may produce one or more outputs. A component consists of an interface
(inputs/outputs), the implementation (a Docker container image and command-line
arguments) and metadata (name, description).

There are the following ways you can use the SDK to build pipelines and components:
- Creating components from existing application code
- Creating components within your application code
- Creating lightweight components
- Using prebuilt, reusuable components in your pipeline

In this section, [MNIST classification](https://www.tensorflow.org/tutorials/quickstart/beginner) use cased 
is used to demonstrate different ways of authoring a pipeline component: 
- [Lightweight Python Components](01_Lightweight_Python_Components.ipynb): this notebook demonstrates how to build a 
component through defining a stand-alone python function and then calling `kfp.components.func_to_container_op(func)` to 
convert, which can be used in a pipeline.

- [Local Development with Docker Image Components](2_Local_Development_with_Docker_Image_Components.ipynb): this 
notebook guides you on creating a pipeline component with `kfp.components.ContainerOp` from an existing Docker image 
which should contain the program to perform the task required in a particular step of your ML workflow.

- [Reusable Components](03_Reusable_Components.ipynb): this notebook describes the manual way of writing a full 
component program (in any language) and a component definition for it. Below is a summary of the steps involved in 
creating and using a component.
    - Write the program that contains your component’s logic. The program must use files and command-line arguments 
    to pass data to and from the component.
    - Containerize the program.
    - Write a component specification in YAML format that describes the component for the Kubeflow Pipelines system.
    - Use the Kubeflow Pipelines SDK to load your component, use it in a pipeline and run that pipeline.

- [Reusable and Pre-build Components as Pipeline](04_Reusable_and_Pre-build_Components_as_Pipeline.ipynb): this 
notebook combines our built components, together with a pre-build GCP AI Platform components 
and a lightweight component to compose a pipeline with three steps.
    - Train a MINIST model and export it to GCS
    - Deploy the exported Tensorflow model on AI Platform prediction service
    - Test the deployment by calling the end point with test data