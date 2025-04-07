# GetImages/__init__.py

import logging
import azure.functions as func
import json
import os
import traceback

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
        
        # Try to list blobs from Azure Storage - isolate each step
        image_blobs = []
        error_details = {}
        
        try:
            # Step 1: Get connection string
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                error_details["step1"] = "Connection string not found"
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment")
            
            # Step 2: Create BlobServiceClient
            try:
                from azure.storage.blob import BlobServiceClient
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("Successfully created BlobServiceClient")
            except Exception as e:
                error_details["step2"] = f"Error creating BlobServiceClient: {str(e)}"
                raise
            
            # Step 3: Create ContainerClient
            try:
                container_client = blob_service_client.get_container_client(container_name)
                logger.info(f"Successfully created ContainerClient for {container_name}")
            except Exception as e:
                error_details["step3"] = f"Error creating ContainerClient: {str(e)}"
                raise
            
            # Step 4: List blobs
            try:
                if images_path:
                    logger.info(f"Listing blobs with prefix: {images_path}")
                    blobs = list(container_client.list_blobs(name_starts_with=images_path))
                else:
                    logger.info("Listing all blobs in container")
                    blobs = list(container_client.list_blobs())
                
                logger.info(f"Successfully listed {len(blobs)} blobs")
            except Exception as e:
                error_details["step4"] = f"Error listing blobs: {str(e)}"
                raise
            
            # Step 5: Filter for image files
            try:
                for blob in blobs:
                    blob_name = blob.name
                    logger.info(f"Found blob: {blob_name}")
                    if blob_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_blobs.append(blob_name)
                
                logger.info(f"Found {len(image_blobs)} images in Azure container '{container_name}'")
            except Exception as e:
                error_details["step5"] = f"Error filtering blobs: {str(e)}"
                raise
            
        except Exception as e:
            logger.error(f"Error in blob operations: {str(e)}")
            tb = traceback.format_exc()
            logger.error(f"Traceback: {tb}")
            error_details["traceback"] = tb
        
        # Return response with images array and any error details
        return func.HttpResponse(
            json.dumps({
                "status": "success" if not error_details else "error",
                "message": f"Found {len(image_blobs)} images" if not error_details else "Error listing blobs",
                "images": image_blobs,
                "container": container_name,
                "environment_check": {
                    **env_var_status,
                    "all_env_vars": env_var_names
                },
                "error_details": error_details
            }),
            mimetype="application/json",
            status_code=200  # Always return 200 to see the response
        )

    except Exception as e:
        logger.error(f"Error in GetImages function: {str(e)}")
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")
        return func.HttpResponse(
            json.dumps({
                "error": str(e), 
                "traceback": tb,
                "status": "error"
            }),
            mimetype="application/json",
            status_code=200  # Return 200 instead of 500 to see the error details
        )