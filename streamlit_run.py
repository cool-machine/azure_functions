import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf

 
tf.config.optimizer.set_jit(False)
tf.experimental.numpy.experimental_enable_numpy_behavior()
import os
import requests
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import base64
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# After load_dotenv()
# Hardcode the correct URLs with /api/ path
AZURE_FUNCTION_URL_IMAGES = 'http://localhost:7071/api/GetImages'
AZURE_FUNCTION_URL_PREDICTION = 'http://localhost:7071/api/GetPrediction'
print(f"Loaded environment variables: AZURE_FUNCTION_URL_IMAGES={AZURE_FUNCTION_URL_IMAGES}")
# Azure Function URLs - Use environment variables if available, otherwise default to localhost:7071
# Force HTTP for local development to avoid SSL errors
# AZURE_FUNCTION_URL_IMAGES = os.environ.get("AZURE_FUNCTION_URL_IMAGES", 'http://localhost:7071/api/GetImages')
# AZURE_FUNCTION_URL_PREDICTION = os.environ.get("AZURE_FUNCTION_URL_PREDICTION", 'http://localhost:7071/api/GetPrediction')

# Force HTTP for localhost connections to prevent SSL errors
if 'localhost' in AZURE_FUNCTION_URL_IMAGES and AZURE_FUNCTION_URL_IMAGES.startswith('https'):
    AZURE_FUNCTION_URL_IMAGES = AZURE_FUNCTION_URL_IMAGES.replace('https://', 'http://')
    logger.warning(f"Forced HTTP for local development: {AZURE_FUNCTION_URL_IMAGES}")

if 'localhost' in AZURE_FUNCTION_URL_PREDICTION and AZURE_FUNCTION_URL_PREDICTION.startswith('https'):
    AZURE_FUNCTION_URL_PREDICTION = AZURE_FUNCTION_URL_PREDICTION.replace('https://', 'http://')
    logger.warning(f"Forced HTTP for local development: {AZURE_FUNCTION_URL_PREDICTION}")

# Log the URLs for debugging
logger.info(f"GetImages URL: {AZURE_FUNCTION_URL_IMAGES}")
logger.info(f"GetPrediction URL: {AZURE_FUNCTION_URL_PREDICTION}")

# Azure Storage container names
AZURE_IMAGES_CONTAINER = os.environ.get("AZURE_IMAGES_CONTAINER_NAME", "images1")
AZURE_MODELS_CONTAINER = os.environ.get("AZURE_MODELS_CONTAINER_NAME", "models")


# Azure images and masks path in container
AZURE_IMAGES_PATH = os.environ.get("AZURE_IMAGES_PATH", "images")
AZURE_MASKS_PATH = os.environ.get("AZURE_MASKS_PATH", "masks")
AZURE_MODELS_PATH = os.environ.get("AZURE_MODELS_PATH", "models")


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


def get_images():
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
            timeout=30  # Add timeout to prevent hanging requests
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully retrieved images")
            return response.json()
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

def get_prediction(image_path):
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
            timeout=60  # Add timeout for longer prediction operations
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
            resized_tensor = tf.image.resize(image_tensor, target_size, method=tf.image.ResizeMethod.BILINEAR)
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



def main():
    st.title("Image Segmentation with Azure Functions")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Display container information
    st.sidebar.subheader("Azure Storage Containers")
    st.sidebar.text(f"Images Container: {AZURE_IMAGES_CONTAINER}")
    st.sidebar.text(f"Models Container: {AZURE_MODELS_CONTAINER}")
    
    # Get list of images
    with st.spinner("Loading images..."):
        image_data = get_images()
        
    if "images" in image_data and image_data["images"]:
        # Create a selectbox for image selection
        selected_image = st.selectbox(
            "Select an image to segment",
            options=image_data["images"],
            format_func=lambda x: x.split("/")[-1]
        )
        
        # Button to run segmentation
        if st.button("Run Segmentation"):
            with st.spinner("Processing image..."):
                st.info(f"Running segmentation on {selected_image}")
                result = get_prediction(selected_image)
                
                if result:
                    # Display each image in full width instead of columns
                    st.subheader("Original Image")
                    if "original" in result and result["original"]:
                        original_array = decode_png_bytes_to_image(result["original"])
                        if original_array is not None:
                            display_image_from_base64_matplotlib(original_array, is_mask=False, caption="Original Image")
                        else:
                            st.warning("Could not decode original image")
                    else:
                        st.warning("Original image not available in response")
                    
                    st.subheader("Ground Truth")
                    if "ground_truth" in result and result["ground_truth"]:
                        original_array = decode_png_bytes_to_image(result["original"])
                        mask_array = decode_png_bytes_to_image(result["ground_truth"], channels=1)
                        if original_array is not None and mask_array is not None:
                            display_image_from_base64_matplotlib(original_array, mask=mask_array, is_prediction=False, is_mask=True, caption="Ground Truth Mask")
                        else:
                            st.warning("Could not decode ground truth mask")
                    else:
                        st.info("Ground Truth Mask not available")
                     
                    st.subheader("Prediction")
                    if "prediction" in result and result["prediction"]:
                        original_array = decode_png_bytes_to_image(result["original"])
                        pred_mask_array = decode_png_bytes_to_image(result["prediction"], channels=1)
                        if original_array is not None and pred_mask_array is not None:
                            display_image_from_base64_matplotlib(original_array, mask=pred_mask_array, is_prediction=True, is_mask=True, caption="Predicted Mask")
                        else:
                            st.warning("Could not decode predicted mask")
                    else:
                        st.warning("Prediction not available in response")
                    

    else:
        st.warning("No images found. Please check your Azure Storage connection and container name.")

if __name__ == "__main__":
    main()
