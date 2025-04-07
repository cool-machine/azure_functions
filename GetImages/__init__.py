# GetImages/__init__.py

import logging
import azure.functions as func
import json
import os
import sys
from pathlib import Path

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Now points to root directory

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import our improved utility functions
from utils.azure_utils import list_blobs

# Log system path for debugging
logging.info(f"System path: {sys.path}")

# # Try different import paths
# try:
#     # Try importing from utils directory in the root
#     from utils.azure_utils import list_blobs
#     logging.info("Successfully imported list_blobs from utils.azure_utils")
# except ImportError as e:
#     logging.error(f"Failed to import list_blobs from utils.azure_utils: {str(e)}")
#     try:
#         # Try from src
#         from src.utils.azure_utils import list_blobs
#         logging.info("Successfully imported list_blobs from src.utils.azure_utils")
#     except ImportError as e:
#         logging.error(f"Failed to import list_blobs from src.utils.azure_utils: {str(e)}")
#         try:
#             # Try direct import
#             import azure_utils
#             list_blobs = azure_utils.list_blobs
#             logging.info("Successfully imported list_blobs directly from azure_utils")
#         except ImportError as e:
#             logging.error(f"All import attempts failed: {str(e)}")
#             raise ImportError("Could not import list_blobs from any location")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Log environment variables (excluding sensitive info)
        env_vars = {k: v for k, v in os.environ.items() if not any(x in k.lower() for x in ['key', 'secret', 'password', 'connection'])}
        logging.info(f"Environment variables: {json.dumps(env_vars)}")
        
        # Get container name from request or use default
        container_name = req.params.get('container_name')
        images_path = req.params.get('images_path')
        masks_path = req.params.get('masks_path')
        
        # Log the parameters for debugging
        logging.info(f"Request parameters: container_name={container_name}, images_path={images_path}, masks_path={masks_path}")
        
        if not container_name:
            # Check if AZURE_IMAGES_CONTAINER_NAME is set
            container_name = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
            logging.info(f"Using default container name: {container_name}")
        
        logging.info(f"Using container: {container_name}")
        logging.info(f"Images path filter: {images_path}")
        logging.info(f"Masks path filter: {masks_path}")

        # List blobs in the container with image extensions
        all_blobs = list_blobs(container_name=container_name)
        
        # Filter for image files
        image_blobs = []
        for blob in all_blobs:
            # Get the blob name as a string
            blob_name = blob.name
            
            # Check for standard image patterns
            if blob_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Apply images_path filter if provided
                if images_path and not (blob_name.startswith(images_path) or f'/{images_path}/' in blob_name):
                    continue
                
                # Look for specific patterns
                if ('_leftImg8bit' in blob_name or 
                    '/images/' in blob_name.lower() or 
                    blob_name.startswith('images/')):
                    image_blobs.append(blob_name)
        
        # logging.info(f"Found {len(image_blobs)} images in Azure container '{container_name}'")
        
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                # "message": "This is a test response from the simplified GetImages function",
                "images": image_blobs,
                "container": container_name,
                "source": "test"
            }),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error in simplified GetImages function: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logging.error(f"Traceback: {tb}")
        return func.HttpResponse(
            json.dumps({"error": str(e), "traceback": tb}),
            mimetype="application/json",
            status_code=500
        )