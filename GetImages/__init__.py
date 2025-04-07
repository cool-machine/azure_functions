# GetImages/__init__.py

import logging
import azure.functions as func
import json
import os
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment_variables(variables):
    """Check if environment variables exist"""
    return {var: var in os.environ for var in variables}

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Get container name from request or use default
        container_name = req.params.get('container_name')
        images_path = req.params.get('images_path', '')
        
        if not container_name:
            container_name = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
        
        logger.info(f"Using container: {container_name}, images_path: {images_path}")
        
        # Check for environment variables
        env_vars_to_check = ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT", "AZURE_IMAGES_CONTAINER_NAME"]
        env_var_status = check_environment_variables(env_vars_to_check)
        
        # Log environment variable status
        for var, exists in env_var_status.items():
            logger.info(f"Environment check: {var} exists: {exists}")
        
        # Get all environment variable names (without values for security)
        env_var_names = list(os.environ.keys())
        logger.info(f"Available environment variables: {env_var_names}")
        
        # Try to list blobs from Azure Storage
        image_blobs = []
        try:
            # Get connection string from environment
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            
            if connection_string:
                logger.info("Found AZURE_STORAGE_CONNECTION_STRING, attempting to list blobs")
                
                # Create blob service client and container client
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                container_client = blob_service_client.get_container_client(container_name)
                
                # List blobs with optional prefix
                if images_path:
                    logger.info(f"Listing blobs with prefix: {images_path}")
                    blobs = container_client.list_blobs(name_starts_with=images_path)
                else:
                    logger.info("Listing all blobs in container")
                    blobs = container_client.list_blobs()
                
                # Filter for image files
                for blob in blobs:
                    blob_name = blob.name
                    logger.info(f"Found blob: {blob_name}")
                    if blob_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_blobs.append(blob_name)
                
                logger.info(f"Found {len(image_blobs)} images in Azure container '{container_name}'")
            else:
                logger.error("AZURE_STORAGE_CONNECTION_STRING not found in environment")
        except Exception as e:
            logger.error(f"Error listing blobs: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return response with images array
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": f"Found {len(image_blobs)} images",
                "images": image_blobs,
                "container": container_name,
                "environment_check": {
                    **env_var_status,
                    "all_env_vars": env_var_names
                }
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error in GetImages function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "traceback": tb}),
            mimetype="application/json",
            status_code=500
        )