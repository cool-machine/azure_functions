# GetImages/__init__.py

import logging
import azure.functions as func
import json
import os
import sys
from pathlib import Path
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Log environment variables (excluding sensitive info)
        env_vars = {k: v for k, v in os.environ.items() 
                   if not any(x in k.lower() for x in ['key', 'secret', 'password', 'connection'])}
        logger.info(f"Environment variables: {json.dumps(env_vars)}")
        
        # Get container name from request or use default
        container_name = req.params.get('container_name')
        images_path = req.params.get('images_path')
        masks_path = req.params.get('masks_path')
        
        # Log the parameters for debugging
        logger.info(f"Request parameters: container_name={container_name}, images_path={images_path}, masks_path={masks_path}")
        
        if not container_name:
            # Check if AZURE_IMAGES_CONTAINER_NAME is set
            container_name = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
            logger.info(f"Using default container name: {container_name}")
        
        logger.info(f"Using container: {container_name}")
        
        # Simple test response to verify function is working
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": "GetImages function is running",
                "container": container_name,
                "environment_check": {
                    "AZURE_STORAGE_CONNECTION_STRING": "AZURE_STORAGE_CONNECTION_STRING" in os.environ,
                    "AZURE_STORAGE_ACCOUNT": "AZURE_STORAGE_ACCOUNT" in os.environ,
                    "AZURE_IMAGES_CONTAINER_NAME": "AZURE_IMAGES_CONTAINER_NAME" in os.environ
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