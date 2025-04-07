from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from src.utils.config import AML_PROPERTIES
import tensorflow as tf


SUB_ID = AML_PROPERTIES["subscription_id"]
RSRC_GR = AML_PROPERTIES["resource_group"]
WRKSP_NM = AML_PROPERTIES["workspace_name"]

# Authenticate and connect to Azure ML Workspace
def get_mlflow_uri():
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id=SUB_ID, resource_group_name=RSRC_GR, workspace_name=WRKSP_NM)

    # Retrieve MLflow tracking URI
    mlflow_tracking_uri = ml_client.workspaces.get(WRKSP_NM).mlflow_tracking_uri
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")

    return mlflow_tracking_uri

def get_tensorboard_writer():
    # Define the log directory for TensorBoard
    tbrd_log_dir = f"./outputs/segf/tensorboard_logs/"
    tensorboard_writer = tf.summary.create_file_writer(tbrd_log_dir)
    print(f"TensorBoard log directory: {tbrd_log_dir}")
    return tensorboard_writer