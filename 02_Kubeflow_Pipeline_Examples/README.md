# Kubeflow Pipeline Examples
Complete pipeline examples with combinations of pre-built
and customised components running on various GCP AI and Analytical services.

- [Pipeline with AI Platform and GCP Service](Pipeline_with_AI_Platform_and_GCP_Service.ipynb): this notebook demonstrates orchestrating 
model training and deployment with Kubeflow Pipelines (KFP) and Cloud AI Platform. In particular, you will develop, deploy, and run a 
KFP pipeline that orchestrates BigQuery and Cloud AI Platform services to train a scikit-learn model. The pipeline uses:
    - Pre-build components. The pipeline uses the following pre-build components that are included with KFP distribution:
        - [BigQuery query component](https://github.com/kubeflow/pipelines/tree/0.1.36/components/gcp/bigquery/query)
        - [AI Platform Training component](https://github.com/kubeflow/pipelines/tree/0.1.36/components/gcp/ml_engine/train)
        - [AI Platform Deploy component](https://github.com/kubeflow/pipelines/tree/0.1.36/components/gcp/ml_engine/deploy)
    - Custom components. The pipeline uses two custom helper components that encapsulate functionality not available in any of the 
    pre-build components. The components are implemented using 
    the KFP SDK's [Lightweight Python Components](https://www.kubeflow.org/docs/pipelines/sdk/lightweight-python-components/) mechanism. 
    The code for the components is in the `helper_components.py` file:
        - **Retrieve Best Run**. This component retrieves the tuning metric and hyperparameter values for the best run of the AI Platform Training hyperparameter tuning job.
        - **Evaluate Model**. This component evaluates the *sklearn* trained model using a provided metric and a testing dataset. 
        the pipeline fails or not.

- [Pipeline with AutoML and BQ Service](Pipeline_with_AutoML_and_BQ_Service.ipynb): In this notebook, you develop a continous training and deployment pipeline using 
Kubeflow Pipelines, BigQuery and AutoML Tables. The scenario used in the lab is  predicting customer lifetime value (CLV).
The goal of CLV modeling is to identify the most valuable customers - customers that are going to generate the highest value in a given future time range. 
The CLV models are built from a variety of data sources - historical sales data being the most important one and in many cases the only one. 
    - The BQ query is run to process sales transactions in the *transactions* table into RFM features in the *features* table. 
    - The data from the *features* table is imported into the AutoML dataset
    - The AutoML model is trained on the imported dataset
    - After the training completes the evaluation metrics are retrieved and compared against the performance threshold
    - If the model performs better than the threshold the model is deployed to AutoML Deployment
    
- [Pipeline with XGBoost and Spark](Pipeline_with_XGboost_on_Spark.ipynb): This tutorial demonstrate building a machine learning pipeling with spark and XGBoost. The pipeline 
    - starts by creating an Google DataProc cluster, and then running analysis, transformation, distributed training and prediction in the created cluster. 
    - Then a single node confusion-matrix and ROC aggregator is used (for classification case) to provide the confusion matrix data, and ROC data to the front end, respectively. 
    - Finally, a delete cluster operation runs to destroy the cluster it creates in the beginning. The delete cluster operation is used as an exit handler, meaning it will run regardless of whether the pipeline fails or not.
