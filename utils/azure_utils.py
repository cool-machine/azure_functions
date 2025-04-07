"""
Azure utilities for connecting to Azure services and managing resources.
"""

import os
import logging
from typing import Optional, Dict, Any
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# Configure logging
logger = logging.getLogger(__name__)

# Singleton Azure client cache
_AZURE_CLIENTS = {}
_CONTAINER_CLIENTS = {}


def get_blob_service_client() -> BlobServiceClient:
    """
    Get a BlobServiceClient instance, creating it if it doesn't exist.
    Uses a singleton pattern to avoid creating multiple connections.
    
    Returns:
        BlobServiceClient: The Azure Blob Storage client
        
    Raises:
        ValueError: If Azure credentials are not found in environment variables
    """
    global _AZURE_CLIENTS
    
    # Return cached client if it exists
    if "blob_service_client" in _AZURE_CLIENTS:
        logger.debug("Using cached Azure Blob Service client")
        return _AZURE_CLIENTS["blob_service_client"]
    
    # Create a new client
    logger.info("Creating new Azure Blob Service client")
    
    # Get Azure Storage connection info
    if "AZURE_STORAGE_CONNECTION_STRING" in os.environ:
        connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        logger.info("Connected to Azure Blob Storage using connection string")

    elif "AZURE_STORAGE_ACCOUNT" in os.environ:
        account_url = f"https://{os.environ['AZURE_STORAGE_ACCOUNT']}.blob.core.windows.net"
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        logger.info(f"Connected to Azure Blob Storage account: {os.environ['AZURE_STORAGE_ACCOUNT']}")
    
    else:
        error_msg = "Azure Storage credentials not found in environment variables"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Cache the client
    _AZURE_CLIENTS["blob_service_client"] = blob_service_client

    return blob_service_client


def get_container_client(container_name: Optional[str] = None, 
                         container_type: str = "images") -> Any:
    """
    Get a container client for the specified container.
    If container_name is not provided, uses AZURE_STORAGE_CONTAINER_NAME from environment.
    Uses a singleton pattern to avoid creating multiple container clients for the same container.
    
    Args:
        container_name: Name of the container to access ("images1" or "models")
        container_type: Type of container ("images" or "models")
    Returns:
        ContainerClient: The Azure Blob Storage container client
        
    Raises:
        ValueError: If container name is not provided and not found in environment variables
    """
    global _CONTAINER_CLIENTS
    
    # Get container name from environment if not provided
    if container_name is None:
        # First try AZURE_IMAGES_CONTAINER_NAME for image-related operations
        if container_type == "images" and "AZURE_IMAGES_CONTAINER_NAME" in os.environ:
            container_name = os.environ["AZURE_IMAGES_CONTAINER_NAME"]
            logger.debug(f"Using AZURE_IMAGES_CONTAINER_NAME: {container_name}")
        # Then try AZURE_MODELS_CONTAINER_NAME for model-related operations
        elif container_type == "models" and "AZURE_MODELS_CONTAINER_NAME" in os.environ:
            container_name = os.environ["AZURE_MODELS_CONTAINER_NAME"]
            logger.debug(f"Using AZURE_MODELS_CONTAINER_NAME: {container_name}")
        else:
            error_msg = "Container name not provided and no container environment variables found"
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        logger.debug(f"Using provided container name: {container_name}")
    
    # Return cached client if it exists
    if container_name in _CONTAINER_CLIENTS:
        logger.debug(f"Using cached container client for container: {container_name}")
        return _CONTAINER_CLIENTS[container_name]
    
    # Get blob service client
    blob_service_client = get_blob_service_client()
    
    # Get container client
    container_client = blob_service_client.get_container_client(container_name)
    logger.info(f"Created new container client for container: {container_name}")
    
    # Cache the container client
    _CONTAINER_CLIENTS[container_name] = container_client
    
    return container_client


def download_blob(blob_path: str, 
                 local_path: str, 
                 container_name: str = "images1", 
                 container_type: str = "images") -> str:
    
    """
    Download a blob from Azure Storage to a local path.
    
    Args:
        blob_path: Path to the blob in Azure Storage
    Args:
        blob_path: Path to the blob in Azure Storage
        local_path: Local path to save the blob to
        container_name: Name of the container to download from (uses AZURE_STORAGE_CONTAINER_NAME if not provided)
        container_type: Type of container ("images" or "models")
        
    Returns:
        str: The local path where the blob was saved
        
    Raises:
        ValueError: If blob doesn't exist or container name is invalid
    """

    # Get container client
    container_client = get_container_client(container_name, container_type=container_type)
    
    # Get blob client
    blob_client = container_client.get_blob_client(blob_path)
    
    # Check if blob exists
    if not blob_client.exists():
        error_msg = f"Blob {blob_path} does not exist in container {container_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download blob
    with open(local_path, "wb") as file:
        blob_data = blob_client.download_blob()
        file.write(blob_data.readall())
    
    logger.info(f"Downloaded blob {blob_path} to {local_path}")
    
    return local_path


def download_blob_to_memory(blob_path: str, 
                           container_name: Optional[str] = None,
                           container_type: str = "images") -> bytes:
    """
    Download a blob from Azure Storage to memory.
    
    Args:
        blob_path: Path to the blob in Azure Storage
        container_name: Name of the container to download from (uses AZURE_STORAGE_CONTAINER_NAME if not provided)
        
    Returns:
        bytes: The blob data
        
    Raises:
        ValueError: If blob doesn't exist or container name is invalid
    """
    # Get container client
    container_client = get_container_client(container_name, container_type=container_type)
    
    # Get blob client
    blob_client = container_client.get_blob_client(blob_path)
    
    # Download blob
    blob_data = blob_client.download_blob().readall()
    
    logger.info(f"Downloaded blob {blob_path} to memory")
    
    return blob_data




def list_blobs(prefix: Optional[str] = None, 
              container_name: Optional[str] = None,
              container_type: str = "images") -> list:
    """
    List blobs in a container with an optional prefix.
    
    Args:
        prefix: Optional prefix to filter blobs by
        container_name: Name of the container to list blobs from (uses AZURE_STORAGE_CONTAINER_NAME if not provided)
        
    Returns:
        list: List of blob items
    """
    # Get container client
    container_client = get_container_client(container_name, container_type=container_type)
    
    # Log container details
    logger.info(f"Listing blobs in container: {container_client.container_name}")
    logger.info(f"With prefix: {prefix or 'None'}")
    logger.info(f"Container type: {container_type}")
    
    # List blobs
    if prefix:
        logger.info(f"Filtering blobs with prefix: {prefix}")
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
    else:
        logger.info("Listing all blobs (no prefix filter)")
        blobs = list(container_client.list_blobs())
    
    # Get the actual container name that was used (in case it was determined from environment variables)
    actual_container = container_client.container_name
    
    # Log the blob names for debugging
    blob_names = [blob.name for blob in blobs]
    logger.info(f"Found {len(blobs)} blobs in container {actual_container}" + 
                (f" with prefix {prefix}" if prefix else ""))
    logger.info(f"Blob names: {blob_names[:10]}" + ("..." if len(blob_names) > 10 else ""))
    
    return blobs
