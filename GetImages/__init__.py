# GetImages/__init__.py

import logging
import azure.functions as func
import json
import os

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
        if not container_name:
            container_name = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
        
        # Check for environment variables
        env_vars_to_check = ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT", "AZURE_IMAGES_CONTAINER_NAME"]
        env_var_status = check_environment_variables(env_vars_to_check)
        
        # Log environment variable status
        for var, exists in env_var_status.items():
            logger.info(f"Environment check: {var} exists: {exists}")
        
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