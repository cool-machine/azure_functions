# GetImages/__init__.py

import logging
import azure.functions as func
import json
import os
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Get container name from request or use default
        container_name = req.params.get('container_name')
        images_path = req.params.get('images_path')
        
        # Log the parameters for debugging
        logger.info(f"Request parameters: container_name={container_name}, images_path={images_path}")
        
        if not container_name:
            # Check if AZURE_IMAGES_CONTAINER_NAME is set
            container_name = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
            logger.info(f"Using default container name: {container_name}")
        
        logger.info(f"Using container: {container_name}")
        
        # Try to list blobs from Azure Storage
        image_blobs = []
        try:
            # Get connection string from environment
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if connection_string:
                logger.info("Found AZURE_STORAGE_CONNECTION_STRING in environment")
                
                # Create blob service client and container client
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                container_client = blob_service_client.get_container_client(container_name)
                
                # List blobs with optional prefix
                if images_path:
                    blobs = container_client.list_blobs(name_starts_with=images_path)
                else:
                    blobs = container_client.list_blobs()
                
                # Filter for image files
                for blob in blobs:
                    blob_name = blob.name
                    if blob_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if ('_leftImg8bit' in blob_name or 
                            '/images/' in blob_name.lower() or 
                            blob_name.startswith('images/')):
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
                "container": container_name
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