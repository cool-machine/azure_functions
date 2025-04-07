"""
process_data.py

This module provides functions for creating TensorFlow Datasets from loaded paths, 
including reading images and labels, normalization, label mapping, augmentation, 
and resizing for model input requirements.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

# Adjust this import to match your project structure. 
# If load_data.py is in the same folder, you can use:
from .loader import get_datasets


def read_image(file_path):
    """
    Read an image file and decode it into a TensorFlow tensor.

    Args:
        file_path (str): The file path of the image.

    Returns:
        tf.Tensor: The decoded and float32 converted image of shape [1024, 2048, 3].
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([1024, 2048, 3])
    return image


def read_label(file_path):
    """
    Read a label file and decode it into a TensorFlow tensor.

    Args:
        file_path (str): The file path of the mask/label.

    Returns:
        tf.Tensor: The decoded label of shape [1024, 2048, 1].
    """
    label = tf.io.read_file(file_path)
    label = tf.image.decode_image(label, channels=1)
    label = tf.image.convert_image_dtype(label, tf.uint8)
    label.set_shape([1024, 2048, 1])
    return label


def normalize(input_image, input_mask):
    """
    Normalize the input image using predefined mean and std values, and clip it to [0,1].
    Labels are left unchanged.

    Args:
        input_image (tf.Tensor): The input image tensor of shape [H, W, 3].
        input_mask (tf.Tensor): The corresponding label tensor.

    Returns:
        tuple: (normalized_image, unchanged_label)
    """
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, K.epsilon())
    input_image = tf.clip_by_value(input_image, 0.0, 1.0)

    return input_image, input_mask


def retrieve_mask_mappings():
    """
    Retrieve original classes, a class mapping dictionary, 
    and new labels for a reduced set of classes.

    Returns:
        tuple: (original_classes, class_mapping, new_labels)
    """
    original_classes = [
        'road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 'car', 'truck',
        'bus', 'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 
        'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 
        'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky', 'ground', 
        'dynamic', 'static'
    ]

    class_mapping = {
        'road': 'flat', 'sidewalk': 'flat',
        'parking': 'flat', 'rail track': 'flat',
        'person': 'human', 'rider': 'human',
        'car': 'vehicle', 'truck': 'vehicle',
        'bus': 'vehicle', 'on rails': 'vehicle',
        'motorcycle': 'vehicle', 'bicycle': 'vehicle',
        'caravan': 'vehicle', 'trailer': 'vehicle',
        'building': 'construction', 'wall': 'construction',
        'fence': 'construction', 'guard rail': 'construction',
        'bridge': 'construction', 'tunnel': 'construction',
        'pole': 'object', 'pole group': 'object',
        'traffic sign': 'object', 'traffic light': 'object',
        'vegetation': 'nature', 'terrain': 'nature',
        'sky': 'sky', 'ground': 'void',
        'dynamic': 'void', 'static': 'void'
    }

    new_labels = {
        'flat': 0, 'human': 1,
        'vehicle': 2, 'construction': 3,
        'object': 4, 'nature': 5,
        'sky': 6, 'void': 7
    }

    return original_classes, class_mapping, new_labels


def map_labels_tf(label_image, original_classes, class_mapping, new_labels):
    """
    Map the original 30-class labels to a reduced set of 8 classes.

    Args:
        label_image (tf.Tensor): The input label image tensor [H, W, 1].
        original_classes (list of str): The full set of original classes.
        class_mapping (dict): Maps each original class to one of the reduced classes.
        new_labels (dict): Maps the reduced classes to numeric labels.

    Returns:
        tf.Tensor: The mapped label image.
    """
    label_image = tf.squeeze(label_image)
    label_shape = tf.shape(label_image)
    mapped_label_image = tf.zeros_like(label_image, dtype=tf.uint8)

    for original_class, new_class in class_mapping.items():
        original_class_index = tf.cast(original_classes.index(original_class), tf.uint8)
        new_class_index = tf.cast(new_labels[new_class], tf.uint8)

        mask = tf.equal(tf.cast(label_image, tf.int32), tf.cast(original_class_index, tf.int32))
        fill_val = tf.fill(label_shape, new_class_index)
        mapped_label_image = tf.where(mask, fill_val, mapped_label_image)

    label = tf.expand_dims(mapped_label_image, axis=-1)
    label = tf.image.convert_image_dtype(label, tf.uint8)
    return label


def augment_image_and_label(image, label, augment_prob=0.3):
    """
    Augment image and label with random flips and brightness/contrast changes.

    Args:
        image (tf.Tensor): The input image tensor [H, W, 3].
        label (tf.Tensor): The input label tensor [H, W, 1].
        augment_prob (float): Probability for performing augmentations.

    Returns:
        tuple: (augmented_image, augmented_label)
    """
    def augment():
        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)

        # Horizontal flip
        if tf.random.uniform([], 0, 1.0) > augment_prob:
            image_in = tf.image.stateless_random_flip_left_right(image, seed=seed)
            label_in = tf.image.stateless_random_flip_left_right(label, seed=seed)
        else:
            image_in = image
            label_in = label

        # Vertical flip
        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
        if tf.random.uniform([], 0, 1.0) > augment_prob:
            image_in = tf.image.stateless_random_flip_up_down(image_in, seed=seed)
            label_in = tf.image.stateless_random_flip_up_down(label_in, seed=seed)

        # Brightness and contrast
        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
        if tf.random.uniform([], 0, 1.0) > augment_prob:
            image_in = tf.image.stateless_random_brightness(image_in, max_delta=0.2, seed=seed)
            image_in = tf.image.stateless_random_contrast(image_in, lower=0.8, upper=1.2, seed=seed)

        return image_in, label_in

    def no_augment():
        return image, label

    random_number = tf.random.uniform([], 0, 1.0)
    image, label = tf.cond(random_number < augment_prob, augment, no_augment)

    image = tf.clip_by_value(image, 0.0, 1.0)
    label = tf.image.convert_image_dtype(label, tf.uint8)

    return image, label



def resize_images(image, label):
    """
    Resize the image and label to the required input shapes for specific models.

    Args:
        image (tf.Tensor): The input image tensor [H, W, 3].
        label (tf.Tensor): The input label tensor [H, W, 1].

    Returns:
        tuple: (resized_image, resized_label)
    """
    image = tf.image.resize(image, size=[512, 1024], method=tf.image.ResizeMethod.BILINEAR)
    label = tf.image.resize(label, size=[128, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.image.convert_image_dtype(label, tf.uint8)

    # Transpose to [C, H, W] format if required
    image = tf.transpose(image, perm=[2, 0, 1])
    output_mask = tf.transpose(label, perm=[2, 0, 1])

    output_image = tf.image.convert_image_dtype(image, tf.float32)
    output_image = tf.clip_by_value(output_image, 0.0, 1.0)

    output_image.set_shape([3, 512, 1024])
    output_mask.set_shape([1, 128, 256])
    
    return output_image, output_mask



def _prepare_dataset(image_paths, mask_paths, batch_size, is_train=True, resize_fn=None):
    """
    Prepares a tf.data.Dataset with common steps including reading, label mapping,
    normalization, optional augmentation, and optional resizing.

    Args:
        image_paths (list): List of image file paths.
        mask_paths (list): List of corresponding mask file paths.
        batch_size (int): The batch size to use.
        is_train (bool): Whether to apply augmentation (only for training).
        resize_fn (function, optional): A function to resize images/labels. If provided, it will be applied.

    Returns:
        tf.data.Dataset: The prepared dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Read and decode images and labels.
    dataset = dataset.map(lambda img, msk: (read_image(img), read_label(msk)), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Map the labels to the reduced set.
    original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    dataset = dataset.map(lambda im, m: (im, map_labels_tf(m, original_classes, class_mapping, new_labels)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # Normalize the images.
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation only to training data.
    if is_train:
        dataset = dataset.map(augment_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optionally apply a resizing function.
    if resize_fn:
        dataset = dataset.map(resize_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch.
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def load_dataset_unet(train_batch_size=1, valid_batch_size=1):
    """
    Create training and validation tf.data.Datasets for U-Net without resizing.
    """
    datasets = get_datasets()
    train_dataset = _prepare_dataset(datasets["train_images"], datasets["train_masks"],
                                     batch_size=train_batch_size, is_train=True, resize_fn=None)
    valid_dataset = _prepare_dataset(datasets["valid_images"], datasets["valid_masks"],
                                     batch_size=valid_batch_size, is_train=False, resize_fn=None)
    return train_dataset, valid_dataset


def load_dataset_segf(train_batch_size=1, valid_batch_size=1):
    """
    Create training and validation tf.data.Datasets for a Segformer-like model with resizing.
    """
    datasets = get_datasets()
    train_dataset = _prepare_dataset(datasets["train_images"], datasets["train_masks"],
                                     batch_size=train_batch_size, is_train=True, resize_fn=resize_images)
    valid_dataset = _prepare_dataset(datasets["valid_images"], datasets["valid_masks"],
                                     batch_size=valid_batch_size, is_train=False, resize_fn=resize_images)
    test_dataset = _prepare_dataset(datasets["test_images"], datasets["test_masks"],
                                     batch_size=valid_batch_size, is_train=False, resize_fn=resize_images)
    return train_dataset, valid_dataset, test_dataset
