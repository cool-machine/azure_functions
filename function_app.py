import logging
import json
import os
import sys
import traceback
from pathlib import Path
import numpy as np
import io
import base64
# from PIL import Image
import tensorflow as tf

# GetImages function definition
from azure.storage.blob import BlobServiceClient
import azure.functions as func

from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parent  # Points directly to root directory

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Import utility modules
from utils.azure_utils import get_blob_service_client, get_container_client, download_blob_to_memory
from utils.processor import read_image, normalize
from utils.image_processing import load_image, encode_image_to_png_bytes, load_model_path
from utils.image_processing import is_image_in_range, decode_png_bytes_to_image, prepare_inference_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment_variables(variables):
    """Check if environment variables exist"""
    return {var: var in os.environ for var in variables}

# Create the main function app
app = func.FunctionApp()


@app.function_name("GetImages")
@app.route(route="GetImages", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def get_images(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Python HTTP trigger function processed a request to get images.')
    
    try:
        # Get container name from request or use default
        container_name = req.params.get('container_name')
        images_path = req.params.get('images_path')
        
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


# GetPrediction function definition
@app.function_name("GetPrediction")
@app.route(route="GetPrediction", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def get_prediction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get predictions.')
    
    try:
        # Parse request body
        logging.info("Parsing request body")
        # Get request body
        req_body = req.get_json()
        
        # Get image path from request body
        image_path = req_body.get('image_path', 'images') # Default to 'images1' if not provided
        model_path = req_body.get('model_path', os.environ.get("AZURE_MODEL_PATH", "segformer"))
        image_container = req_body.get('image_container', os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1"))
        model_container = req_body.get('model_container', os.environ.get("AZURE_MODELS_CONTAINER_NAME", "models"))

        # Check if model path is provided
        if not model_path:
            return func.HttpResponse(
                json.dumps({"error": "No model path provided"}),
                mimetype="application/json",
                status_code=400
            )

        # Check if image path is provided
        if not image_path:
            return func.HttpResponse(
                json.dumps({"error": "No image path provided"}),
                mimetype="application/json",
                status_code=400
            )

        # Log parameters
        logging.info(f"Image path: {image_path}")
        logging.info(f"Model path: {model_path}")
        logging.info(f"Using image container: {image_container}")
        logging.info(f"Using model container: {model_container}")
        
        # Get corresponding mask path
        mask_path = None
        mask_tensor = None
        
        # Get mask path from environment variable
        azure_masks_path = os.environ.get("AZURE_MASKS_PATH", "masks")
        logging.info(f"Using mask path from environment: {azure_masks_path}")

        # Extract the image filename and directory parts
        image_name_parts = image_path.split('/')
        image_filename = image_name_parts[-1] if len(image_name_parts) > 1 else image_path
        logging.info(f"Image filename extracted: {image_filename}")

        # Strategy 1: For Cityscapes format (leftImg8bit.png -> gtFine_labelIds.png)
        if '_leftImg8bit.png' in image_filename:
            # Standard Cityscapes format
            mask_path = image_filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            logging.info(f"mask_path before replacement: {mask_path}")
            # Replace "images" with "masks" in the path
            mask_path = f"{azure_masks_path}/{mask_path}"
            logging.info(f"mask_path after replacement: {mask_path}")
        else:
            # If not in Cityscapes format, we don't have a mask
            logging.warning(f"Image {image_filename} is not in Cityscapes format, no mask will be provided")
            mask_path = None
            mask_tensor = None

        if mask_path and image_path:
            logging.info(f"in mask path: {mask_path} and Image path: {image_path}")            
            try:
                # Load image using our improved function with container_name
                image_tensor, mask_tensor = prepare_inference_data(image_path, mask_path, image_container=image_container)
                # mask_tensor = load_image(mask_path, container_name=image_container)
                if mask_tensor is None or mask_tensor is None:
                    logging.warning(f"Mask not found at path: {mask_path}")
            except Exception as e:
                logging.error(f"Error loading mask from {mask_path}: {str(e)}")
                mask_tensor = None

        if not is_image_in_range(image_tensor):
            image_tensor = image_tensor / 255.0
        else:
            image_tensor = tf.cast(image_tensor, tf.uint8)
        
        if is_image_in_range(mask_tensor):
            mask_tensor = tf.cast(mask_tensor * 255, tf.uint8)
        else:
            mask_tensor = tf.cast(mask_tensor, tf.uint8)
        
        logging.info(f"Mask tensor shape before transpose: {mask_tensor.shape}")
        mask_tensor = tf.transpose(mask_tensor, (1, 2, 0))
        logging.info(f"Mask tensor shape after transpose: {mask_tensor.shape}")
        logging.info(f"Image tensor shape after cast: {image_tensor.shape}")
        logging.info(f"Mask tensor shape after cast: {mask_tensor.shape}")
        
        if len(mask_tensor.shape) == 2:
            mask_tensor = tf.expand_dims(mask_tensor, axis=-1)

        model_path_temp = load_model_path(model_path)
        # Reshape the tensor to match the expected input format for SegFormer
        if len(image_tensor.shape) == 3:
            # Log original shape for debugging
            logging.info(f"Original tensor shape before reshaping: {image_tensor.shape}")
            
            # Check if the shape is (channels, height, width)
            if image_tensor.shape[-1] == 3:
                # Convert from (channels, height, width) to (height, width, channels)
                image_tensor = tf.transpose(image_tensor, [2, 0, 1])
                logging.info(f"After transpose: {image_tensor.shape}")
            
            # Now add batch dimension to get (batch, height, width, channels)
            image_tensor = tf.expand_dims(image_tensor, axis=0)
            logging.info(f"Final tensor shape after adding batch dimension: {image_tensor.shape}")
        
        image_tensor = tf.cast(image_tensor, tf.float32)        

        # Create a new configuration
        config = SegformerConfig(
            num_labels=8,  # Set the number of labels/classes
            id2label={0: "flat", 1: "human", 2: "vehicle", 3: "construction", 4: "object", 5: "nature", 6: "sky", 7: "void"},
            label2id={"flat": 0, "human": 1, "vehicle": 2, "construction": 3, "object": 4, "nature": 5, "sky": 6, "void": 7},
            image_size=(512, 1024),  # Specify the input image size
        )   
        logging.info(f"checking model path: {model_path_temp}")
        
        model = TFSegformerForSemanticSegmentation.from_pretrained(model_path_temp, 
                                                                   config=config,
                                                                   ignore_mismatched_sizes=True)
        output_mask = model(image_tensor).logits

        logging.info(f"Output mask shape after logits: {output_mask.shape}")
        
        # logging.info(f"Raw logits min: {tf.reduce_min(output_mask)}, max: {tf.reduce_max(output_mask)}, mean: {tf.reduce_mean(output_mask)}")
        # unique_values, _, counts = tf.unique_with_counts(tf.reshape(output_mask, [-1]))
        # logging.info(f"Unique prediction values: {unique_values.numpy()}, counts: {counts.numpy()}")

        image_tensor = tf.transpose(image_tensor, perm=[0, 2, 3, 1]) 
        logging.info(f"Image tensor shape after transpose: {image_tensor.shape}")
        image_tensor = tf.squeeze(image_tensor, axis=0)
        logging.info(f"Image tensor shape after squeeze: {image_tensor.shape}")        


        if is_image_in_range(image_tensor):
            image_tensor = tf.cast(image_tensor * 255, tf.uint8)
        else:
            image_tensor = tf.cast(image_tensor, tf.uint8)
        
        logging.info(f"Image tensor shape after cast: {image_tensor.shape}")

        # logging.info(f"Output mask shape just after ouptutting from model: {output_mask.shape}")

        output_mask = tf.transpose(output_mask, perm=[0, 2, 3, 1])

        logging.info(f"Output mask shape after transpose: {output_mask.shape}")
        
        output_mask = tf.argmax(output_mask, axis=-1)
 
        unique_values, _, counts = tf.unique_with_counts(tf.reshape(output_mask, [-1]))
        logging.info(f"Unique prediction values just after argmax: {unique_values.numpy()}, counts: {counts.numpy()}")
        
        logging.info(f"Output mask after argmax: min: {tf.reduce_min(output_mask)}, max: {tf.reduce_max(output_mask)}, mean: {tf.reduce_mean(output_mask)}")
        
        # logging.info(f"Output mask shape after argmax: {output_mask.shape}")

        output_mask = tf.transpose(output_mask, perm=[1, 2, 0])
        
        logging.info(f"Output mask shape after transpose: {output_mask.shape}")

        # output_mask = tf.image.convert_image_dtype(output_mask, tf.uint8)

        logging.info(f"Output mask shape after convert_image_dtype: {output_mask.shape}")

        if is_image_in_range(output_mask):
            output_mask = tf.cast(output_mask * 255, tf.uint8)
        else:
            output_mask = tf.cast(output_mask, tf.uint8)

        logging.info(f"Output mask shape after cast: {output_mask.shape}")
        
        unique_values, _, counts = tf.unique_with_counts(tf.reshape(output_mask, [-1]))
        logging.info(f"Unique prediction values after cast: {unique_values.numpy()}, counts: {counts.numpy()}")
        
        image_b64 = None 
        mask_b64 = None
        mask_prediction_b64 = None
        
        if image_tensor is not None:
            image_b64 = encode_image_to_png_bytes(image_tensor)
        if mask_tensor is not None:
            mask_b64 = encode_image_to_png_bytes(mask_tensor)
        if output_mask is not None:
            mask_prediction_b64 = encode_image_to_png_bytes(output_mask)

        logging.info(f"Image and mask converted to image_b64 and mask_b64")

        # Create response with available data 
        response_data = {
            "original": image_b64,
            "ground_truth": mask_b64,
            "prediction": mask_prediction_b64
        }

        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

# Health check function
@app.function_name("HealthCheck")
@app.route(route="HealthCheck", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Simple health check endpoint to verify the function app is running.
    """
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "message": "Azure Functions V2 is running",
            "version": "2.0"
        }),
        mimetype="application/json",
        status_code=200
    )
