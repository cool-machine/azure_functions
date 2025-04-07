# api/GetPrediction
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Manage memory growth
os.environ['TF_DISABLE_JIT'] = '1'  # Disable JIT compilation
os.environ['TF_ENABLE_XLA'] = '0'   # Disable XLA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging noise

import tensorflow as tf
tf.config.optimizer.set_jit(False)
tf.experimental.numpy.experimental_enable_numpy_behavior()
import logging
import azure.functions as func
import json
import sys
import tempfile
import io
import base64
import numpy as np
import tensorflow as tf

from typing import Optional, Any, Union

# Set matplotlib backend to Agg (non-interactive) before importing pyplot
import matplotlib.pyplot as plt
import matplotlib

from transformers import TFSegformerForSemanticSegmentation, SegformerConfig



matplotlib.use('Agg')

from pathlib import Path


# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Now points directly to root directory

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Import our improved utility functions
from utils.azure_utils import download_blob, list_blobs, download_blob_to_memory

def is_image_in_range(img_array):
    return np.all((img_array >= -0.01) & (img_array <= 1.01))



# Assume image is a numpy array of shape (H, W, C)
def encode_image_to_png_bytes(image):
    if image is None:
        logging.warning("encode_image_to_png_bytes received None input")
        return None
    
    # Convert to uint8 if not already
    if not isinstance(image, (np.ndarray, tf.Tensor)):
        logging.error(f"Image is not a numpy array or tensor, it's a {type(image)}")
        return None

    # Make sure we have a proper shape for encoding
    if len(image.shape) == 2:  # Single channel (grayscale)
        image = tf.expand_dims(image, axis=-1)

    try:
        logging.info(f"Image shape: {image.shape}")
        logging.info(f"Image dtype: {image.dtype}")
        png_encoded = tf.io.encode_png(image)
        png_bytes = png_encoded.numpy()  # Convert tensor to raw PNG bytes
        encoded_string = base64.b64encode(png_bytes).decode("utf-8")
        return encoded_string

    except Exception as e:
        logging.error(f"Error encoding image to PNG bytes: {str(e)}")
        return None

def load_image(image_source: Union[str, bytes],
               container_name: Optional[str] = "images1") -> np.ndarray:
    """
    Load an image from various sources (file path, Azure blob, or bytes).
    
    Args:
        image_source: Path to image, Azure blob path, or image bytes
        container_name: Optional Azure container name to override AZURE_STORAGE_CONTAINER_NAME
                       If not provided and AZURE_IMAGES_CONTAINER_NAME exists, it will be used
        
    Returns:
        Image as a normalized numpy array or None if the image couldn't be loaded
    """
    try:
        from utils.azure_utils import download_blob_to_memory
        image_data = download_blob_to_memory(image_source, 
                                            container_name=container_name, 
                                            container_type="images")

        logging.info(f"Downloaded blob size: {len(image_data)} bytes from {image_source}")
        # Convert bytes to numpy array using matplotlib
        try:
            # Create a BytesIO object from the image data
            image_buffer = io.BytesIO(image_data)
            
            # Use matplotlib to load the image
            img = plt.imread(image_buffer)
            
            # Ensure we return a numpy array, not a tensor
            if isinstance(img, tf.Tensor):
                img = img.numpy()
                logging.info(f"Successfully converted image to array with shape: {img.shape}")
            return img

        except Exception as e:
            logging.error(f"Error converting image data to array: {str(e)}")
            # Return the raw bytes as fallback
            return image_data
    except Exception as e:
        logging.error(f"Error downloading blob: {str(e)}")
        return None
 



# Import our improved utility functions
# from src.inference.deployment_helpers import load_model, load_image, create_colored_mask
# Singleton model cache
_MODEL_CACHE = {}

# Function to get the model cache directory
def _get_model_cache_dir():
    """Get the directory where models are cached."""
    # Use a persistent directory for caching models
    cache_dir = os.path.join(tempfile.gettempdir(), "bs_model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
    
# Load the model
def load_model_path(model_path: Optional[str] = None, 
               container_name: Optional[str] = "models") -> Any:
    """
    Load the segmentation model, using a cached version if available.
    
    Args:
        model_path: Optional path to the model, Azure blob path, or HuggingFace model ID.
                   If None, a default HuggingFace model will be used.
        container_name: Optional Azure container name to override AZURE_STORAGE_CONTAINER_NAME.
                       If not provided and AZURE_MODELS_CONTAINER_NAME exists, it will be used.
        
    Returns:
        The loaded model
    """
    global _MODEL_CACHE
    
    # Check if model is already cached
    cache_key = f"model_{model_path}_{container_name}"
    if cache_key in _MODEL_CACHE:
        logging.info("Using cached model")
        return _MODEL_CACHE[cache_key]
    
    # Use the Azure model path from environment variable if none is provided
    if model_path is None:
        if "AZURE_STORAGE_MODEL_PATH" in os.environ:
            model_path = os.environ["AZURE_STORAGE_MODEL_PATH"]
            logging.info(f"Using model path from environment: {model_path}")
        else:
            error_msg = "No model path provided and AZURE_STORAGE_MODEL_PATH environment variable not set"
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    # Determine container name for models
    if container_name is None:
        if "AZURE_MODELS_CONTAINER_NAME" in os.environ:
            container_name = os.environ["AZURE_MODELS_CONTAINER_NAME"]
            logging.debug(f"Using AZURE_MODELS_CONTAINER_NAME: {container_name}")
        else:
            error_msg = "Container name not provided and no container environment variables found"
            logging.error(error_msg)
            raise ValueError(error_msg)    
    
    try:
        logging.info(f"Loading model from Azure: {model_path}")
        # Use a persistent cache directory for models
        cache_dir = _get_model_cache_dir()
        model_dir = os.path.join(cache_dir, os.path.basename(model_path))
        os.makedirs(model_dir, exist_ok=True)
        logging.info(f"Created temporary directory for model: {model_dir}")
        # List all blobs in the model path
        blobs = list_blobs(prefix=model_path, container_name=container_name, container_type="models")

        if not blobs:
            error_msg = f"No model files found at {model_path} in Azure container {container_name or 'default'}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        else:
            logging.info(f"Found {len(blobs)} model files in Azure")
            # Download all model files
            for blob in blobs:
                # Get the relative path of the blob
                blob_path = blob.name
                # Create the local path where the blob will be downloaded
                relative_path = os.path.relpath(blob_path, model_path)
                local_path = os.path.join(model_dir, relative_path)
                # Download the blob using azure_utils
                download_blob(blob_path, local_path, container_name=container_name, container_type="models")
            # Update model_path to point to the downloaded model
            model_path = model_dir
            logging.info(f"Model downloaded to {model_path}")
    
        # Load the model
        # model = load_segformer_model(model_path)

    except Exception as e:
        error_msg = f"Error loading model from Azure: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # Cache the model
    # _MODEL_CACHE[cache_key] = model
    logging.info("Model loaded and cached successfully")    
    return model_path


def prepare_inference_data(image_path, mask_path, image_container):
    """
    Prepare image and mask for inference using the same preprocessing as training
    
    Args:
        image_path (str): Path to the image file
        mask_path (str, optional): Path to the mask file, if available
        image_container (str): Name of the Azure Blob Storage container for images
        
    Returns:
        tuple: (
            model_ready_image: tf.Tensor with batch dimension ready for model input,
            model_ready_mask: tf.Tensor with batch dimension (if mask_path provided) or None,
        )
    """
    from utils.processor import normalize, resize_images, retrieve_mask_mappings, map_labels_tf

    # Load the original image for display
    original_image = None
    try:
        # Use the image_container parameter passed from the request
        original_image = load_image(image_path, container_name=image_container)
        logging.info(f"Loaded original image for display from container {image_container}")
        if original_image is not None:
            logging.info(f"Successfully loaded original  with shape: {np.array(original_image).shape}")    

    except Exception as e:
        logging.error(f"Could not load original image for display: {str(e)}")
        original_image = None
    
    # If original image couldn't be loaded, return early
    if original_image is None:
        logging.error("Original image could not be loaded, cannot proceed with inference")
        return None, None
    
    # Load the original mask for display
    original_mask = None
    try:
        # Use the same container for masks
        logging.info(f"Attempting to load mask from path: {mask_path} in container {image_container}")
        original_mask = load_image(mask_path, container_name=image_container)
        
        if len(original_mask.shape) == 2:
            original_mask = tf.expand_dims(original_mask, axis=-1)
        
        if original_mask is not None:
            logging.info(f"Successfully loaded original mask with shape: {np.array(original_mask).shape}")    
            
    except Exception as e:
        logging.error(f"Could not load original mask for display: {str(e)}")
        original_mask = None

    if original_mask is None:
        logging.error("Original mask could not be loaded, cannot proceed with inference")
        return None, None
    
    logging.info(f"Decoding and converting image and mask with types: {type(original_image)}, {type(original_mask)}")
    # image = tf.image.decode_image(original_image, channels=3)
    logging.info(f"Decoded image shape: {original_image.shape}")
    image = tf.image.convert_image_dtype(original_image, tf.float32)
    logging.info(f"Converted image shape: {image.shape}")
    image.set_shape([1024, 2048, 3])
    
    # label = tf.image.decode_image(original_mask, channels=1)
    label = tf.image.convert_image_dtype(original_mask, tf.uint8)
    label.set_shape([1024, 2048, 1])


    original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    label = map_labels_tf(label, original_classes, class_mapping, new_labels)


    image, label = normalize(image, label)

    image, label = resize_images(image, label)
    logging.info(f"Resized image shape: {image.shape}, resized label shape: {label.shape}")
    return image, label



# Define standard colors for segmentation masks
CATEGORY_COLORS = {
    0: [255, 0, 0],      # Red
    1: [0, 255, 0],      # Green
    2: [0, 0, 255],      # Blue
    3: [255, 255, 0],    # Yellow
    4: [255, 0, 255],    # Magenta
    5: [0, 255, 255],    # Cyan
    6: [128, 128, 128],  # Gray
    7: [255, 165, 0],    # Orange
}


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request to get predictions.')
    
    # Configure logging to be more verbose
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Parse request body
        logging.info("Parsing request body")
        req_body = req.get_json()
        
        # Get parameters
        model_container = req_body.get('model_container', os.environ.get('AZURE_MODELS_CONTAINER_NAME', 'models'))
        model_path = req_body.get('model_path', 'segformer') 

        image_container = req_body.get('image_container', os.environ.get('AZURE_IMAGES_CONTAINER_NAME', 'images1'))
        image_path = req_body.get('image_path')

        if not model_path:
            return func.HttpResponse(
                json.dumps({"error": "No model path provided"}),
                mimetype="application/json",
                status_code=400
            )

        if not image_path:
            return func.HttpResponse(
                json.dumps({"error": "No image path provided"}),
                mimetype="application/json",
                status_code=400
            )
            
        logging.info(f"Processing image: {image_path}")
        logging.info(f"Using image container: {image_container}")
        logging.info(f"Using model container: {model_container}")
        logging.info(f"Using model path: {model_path}")
        
        
        # Get corresponding mask path
        # image_tensor = None
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

        logging.info(f"Input image shape just before inputting image: {image_tensor.shape}")
        
        
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
