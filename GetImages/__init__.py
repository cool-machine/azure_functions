# GetImages/__init__.py

import logging
import azure.functions as func
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Get container name from request or use default
        container_name = req.params.get('container_name')
        if not container_name:
            container_name = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
        
        # Check for environment variables
        connection_string_exists = "AZURE_STORAGE_CONNECTION_STRING" in os.environ
        account_name_exists = "AZURE_STORAGE_ACCOUNT" in os.environ
        container_name_exists = "AZURE_IMAGES_CONTAINER_NAME" in os.environ
        
        # Log environment variable status
        logger.info(f"Environment check: AZURE_STORAGE_CONNECTION_STRING exists: {connection_string_exists}")
        logger.info(f"Environment check: AZURE_STORAGE_ACCOUNT exists: {account_name_exists}")
        logger.info(f"Environment check: AZURE_IMAGES_CONTAINER_NAME exists: {container_name_exists}")
        
        # Get all environment variable names (without values for security)
        env_var_names = list(os.environ.keys())
        logger.info(f"Available environment variables: {env_var_names}")
        
        # Return response with environment check
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "Environment check completed",
                "images": [],
                "container": container_name,
                "environment_check": {
                    "AZURE_STORAGE_CONNECTION_STRING": connection_string_exists,
                    "AZURE_STORAGE_ACCOUNT": account_name_exists,
                    "AZURE_IMAGES_CONTAINER_NAME": container_name_exists,
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