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
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import azure.functions as func
from pathlib import Path
import json
import sys
import tempfile
import io
import base64
import numpy as np
import requests
import streamlit as st
from typing import Optional, Any, Union, List, Dict, Tuple
from PIL import Image
from io import BytesIO

# Set matplotlib backend to Agg (non-interactive) before importing pyplot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent  # Points directly to root directory

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import our improved utility functions
from utils.azure_utils import download_blob, list_blobs

# Define Azure Function URLs
AZURE_FUNCTION_URL_IMAGES = "https://test-ocp8.azurewebsites.net/api/GetImages"
AZURE_FUNCTION_URL_PREDICTION = "https://test-ocp8.azurewebsites.net/api/GetPrediction"


# Get environment variables
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_IMAGES_CONTAINER = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
AZURE_MODELS_CONTAINER = os.environ.get("AZURE_MODELS_CONTAINER_NAME", "models")
AZURE_IMAGES_PATH = os.environ.get("AZURE_IMAGES_PATH", "images")
AZURE_MASKS_PATH = os.environ.get("AZURE_MASKS_PATH", "masks")
AZURE_MODELS_PATH = os.environ.get("AZURE_MODELS_PATH", "models")

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def is_image_in_range(img_array):
    logging.info("got into is_image_in_range function")
    logging.info(f"Image array shape: {img_array.shape}")
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
# from inference.deployment_helpers import load_model, load_image, create_colored_mask
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
            original_image: Original image tensor for display,
            original_mask: Original mask tensor for display (if provided) or None
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
        return None, None, None, None
    
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


def decode_png_bytes_to_image(png_bytes, channels=3):
    try:
        decoded_bytes = base64.b64decode(png_bytes)
        image_tensor = tf.io.decode_png(decoded_bytes, channels=channels)  # Adjust channels as needed
        return image_tensor
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        st.error(f"Error decoding image: {str(e)}")
        return None


def is_image_in_range(img_array):
    return np.all((img_array >= -0.01) & (img_array <= 1.01))


def is_image_in_range(img_array):
    return np.all((img_array >= -0.01) & (img_array <= 1.01))


def _get_images():
    """Get list of available images from Azure Function"""
    try:
        logger.info(f"Requesting images from {AZURE_FUNCTION_URL_IMAGES}")
    
        # Ensure the URL doesn't have extra quotes
        url = AZURE_FUNCTION_URL_IMAGES
        if url.startswith('"') and url.endswith('"'):
            url = url[1:-1]
            logger.warning(f"Removed extra quotes from URL: {url}")
        
        response = requests.get(
            url,
            params={
                "container_name": AZURE_IMAGES_CONTAINER, 
                "images_path": AZURE_IMAGES_PATH,
                "masks_path": AZURE_MASKS_PATH,
            },
            timeout=180  # Add timeout to prevent hanging requests
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully retrieved images")
            response_data = response.json()
            
            # Display the full response for debugging
            with st.expander("Debug: GetImages Response"):
                st.json(response_data)
                
            return response_data
        else:
            error_msg = f"Error getting images: {response.status_code} - {response.text}"
            logger.error(error_msg)
            st.error(error_msg)
            return {"images": []}
    except Exception as e:
        import traceback
        error_msg = f"Error connecting to Azure Function: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        st.error(f"Traceback: {traceback.format_exc()}")
        return {"images": []}

def _get_prediction(image_path):
    """Get prediction for an image from Azure Function"""
    try:
        logger.info(f"Requesting prediction for {image_path}")
        # Ensure the URL doesn't have extra quotes
        url = AZURE_FUNCTION_URL_PREDICTION
        if url.startswith('"') and url.endswith('"'):
            url = url[1:-1]
            logger.warning(f"Removed extra quotes from URL: {url}")
        
        response = requests.post(
            url,
            json={
                "image_path": image_path,
                "image_container": AZURE_IMAGES_CONTAINER,
                "model_container": AZURE_MODELS_CONTAINER,
            },
            timeout=180  # Add timeout for longer prediction operations
        )
        if response.status_code == 200:
            logger.info(f"Successfully received prediction")
            return response.json()
        else:
            error_msg = f"Error getting prediction: {response.text}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error connecting to Azure Function: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None

#Plot results - plot colored segmentation
def display_image_from_base64_matplotlib(image, mask=None, is_mask=False, is_prediction=False, caption=""):
    """ 
    Plot an image with a colorized segmentation mask overlay.
    
    Args:
        image (np.ndarray): Original image array with shape (H, W, C).
        mask (np.ndarray): Segmentation mask array with shape (H, W) containing category indices.
        is_mask (bool): Whether to display the mask overlay
        is_prediction (bool): Whether the mask is a prediction (needs argmax)
        caption (str): Caption for the image
    """
    try:
        
        # Target size for display
        new_height, new_width = 512, 1024
        target_size = (new_height, new_width)
         
        # Define colors for segmentation classes (RGB values)
        categories_colors = {
            0: [255, 0, 0],    # Red
            1: [0, 255, 0],    # Green
            2: [0, 0, 255],    # Blue
            3: [255, 255, 0],  # Yellow
            4: [255, 0, 255],  # Magenta
            5: [0, 255, 255],  # Cyan
            6: [128, 128, 128], # Gray
            7: [255, 165, 0],  # Orange
        }
        
        # Process the image
        if isinstance(image, tf.Tensor):
            image_np = image.numpy()
        else:
            image_np = image
            
        # Handle different image shapes
        if len(image_np.shape) == 4:
            image_np = image_np[0]
            
        # Handle channel-first format (e.g., shape is (C, H, W))
        if len(image_np.shape) == 3 and image_np.shape[0] <= 3:
            image_np = np.transpose(image_np, [1, 2, 0])
            
        # Handle RGBA images
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
            
        # Resize image to target size
        if isinstance(image_np, np.ndarray):
            # Use TensorFlow for resizing
            image_tensor = tf.convert_to_tensor(image_np)
            resized_tensor = tf.image.resize(image_tensor, target_size, method=tf.image.ResizeMethod.BILINEAR)
            resized_image = resized_tensor.numpy().astype(np.uint8)
        else:
            # Fallback if not a numpy array
            resized_tensor = tf.image.resize(image_np, target_size, method=tf.image.ResizeMethod.BILINEAR)
            resized_image = resized_tensor.numpy().astype(np.uint8)
        
        # Process mask if provided
        color_true_mask = None
        if is_mask and mask is not None:
            # Convert TensorFlow tensor to numpy if needed
            if isinstance(mask, tf.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = mask
                
            # Handle batch dimension
            if len(mask_np.shape) == 4:
                mask_np = mask_np[0]
                
            # Handle channel-first format
            if len(mask_np.shape) == 3 and mask_np.shape[0] <= 3:
                mask_np = np.transpose(mask_np, [1, 2, 0])
                
            # If mask is a prediction with multiple channels, take argmax
            if is_prediction and len(mask_np.shape) == 3 and mask_np.shape[2] > 1:
                mask_np = np.argmax(mask_np, axis=2)
                
            # Ensure mask is 2D (single channel)
            if len(mask_np.shape) == 3:
                if mask_np.shape[2] == 1:
                    # Single-channel mask with explicit channel dimension
                    mask_np = np.squeeze(mask_np, axis=2)
                else:
                    # Multi-channel mask, take first channel as class indices
                    mask_np = mask_np[:, :, 0]
            
            # Resize mask to target size using TensorFlow (nearest neighbor for masks)
            mask_tensor = tf.convert_to_tensor(mask_np)
            resized_mask_tensor = tf.image.resize(
                tf.expand_dims(mask_tensor, axis=-1),  # Add channel dimension for TF resize
                target_size, 
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            # Remove the added channel dimension and convert back to numpy
            resized_mask = tf.squeeze(resized_mask_tensor, axis=-1).numpy().astype(np.uint8)
            
            # Create a color mask for visualization
            color_true_mask = np.zeros((*target_size, 3), dtype=np.uint8)
            
            # Map each category to its corresponding color
            for category, color in categories_colors.items():
                category_pixels = resized_mask == category
                if np.any(category_pixels):
                    color_true_mask[category_pixels] = color
        
        # Create figure and display
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(resized_image)
        
        # Overlay mask if available
        if is_mask and color_true_mask is not None:
            ax.imshow(color_true_mask, alpha=0.15)

            
        ax.set_title(caption)
        ax.axis("off")
        
        st.pyplot(fig)
        # st.text("Successfully displayed image and mask")
        
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        import traceback
        st.text(traceback.format_exc())