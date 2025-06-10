import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Average
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from openai import OpenAI
from google.colab import files
import matplotlib.pyplot as plt
import pickle
import gc
import os

# --- Part 0: OpenAI API Client Initialization ---
client = OpenAI(
    api_key=""
)

# CIFAR-100 class names (fine labels)
CIFAR100_LABELS = []

# Define target image size for models
TARGET_IMG_SIZE = (224, 224)
NUM_CLASSES = 100

# --- Part 1: Data Preprocessing ---

def unpickle_cifar100(file_path):
    """Helper function to load CIFAR-100 batch files."""
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar100_from_pre_extracted_path(extracted_data_path="/root/tensorflow_datasets"):
    """
    Loads the CIFAR-100 dataset from a pre-extracted directory.
    """
    print(f"Loading CIFAR-100 from pre-extracted directory: {extracted_data_path}...")

    # Define paths to expected files
    train_file_path = os.path.join(extracted_data_path, 'train')
    test_file_path = os.path.join(extracted_data_path, 'test')
    meta_file_path = os.path.join(extracted_data_path, 'meta')

    # Verify files exist
    if not os.path.exists(extracted_data_path):
        raise FileNotFoundError(f"Directory not found: {extracted_data_path}")
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"Missing expected training file: {train_file_path}")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Missing expected test file: {test_file_path}")
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Missing expected meta file: {meta_file_path}")

    # Load training data
    train_batch = unpickle_cifar100(train_file_path)
    x_train_raw = train_batch[b'data']
    y_train_raw = train_batch[b'fine_labels']

    # Load test data
    test_batch = unpickle_cifar100(test_file_path)
    x_test_raw = test_batch[b'data']
    y_test_raw = test_batch[b'fine_labels']

    # Load class names
    meta_batch = unpickle_cifar100(meta_file_path)
    global CIFAR100_LABELS
    CIFAR100_LABELS = [label.decode('utf-8') for label in meta_batch[b'fine_label_names']]
    print(f"Loaded {len(CIFAR100_LABELS)} CIFAR-100 class names.")

    # Reshape images to (N, H, W, C) format
    x_train = x_train_raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train_raw).reshape(-1, 1)
    x_test = x_test_raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test_raw).reshape(-1, 1)

    print("CIFAR-100 raw data loaded from specified directory.")
    return (x_train, y_train), (x_test, y_test)

def create_tf_dataset(images, labels, target_size, batch_size, model_type='inception', shuffle=True):
    """
    Creates a TensorFlow Dataset with appropriate preprocessing for each model type.
    """
    def _preprocess_image_and_label(image, label):
        image = tf.image.resize(image, target_size)
        
        # Apply model-specific preprocessing
        if model_type == 'inception':
            image = inception_preprocess(image)
        elif model_type == 'resnet':
            image = resnet_preprocess(image)
        elif model_type == 'efficientnet':
            image = efficientnet_preprocess(image)
            
        label = tf.squeeze(label)
        label = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(_preprocess_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# --- Part 2: Model Training ---

def build_single_model(base_model_class, input_shape, model_name):
    """
    Builds a single transfer learning model with custom top layers.
    """
    print(f"\nBuilding {model_name} base model...")
    
    # Create base model
    base_model = base_model_class(weights='imagenet', 
                                 include_top=False, 
                                 input_shape=input_shape)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"{model_name} model built and compiled.")
    return model

def build_ensemble_model(input_shape):
    """
    Builds an ensemble model combining InceptionV3, ResNet50, and EfficientNetB0.
    """
    # Create individual models
    inception_model = build_single_model(InceptionV3, input_shape, "InceptionV3")
    resnet_model = build_single_model(ResNet50, input_shape, "ResNet50")
    efficientnet_model = build_single_model(EfficientNetB0, input_shape, "EfficientNetB0")
    
    # Create ensemble by averaging predictions
    ensemble_output = Average()([inception_model.output, 
                                resnet_model.output, 
                                efficientnet_model.output])
    
    ensemble_model = Model(inputs=inception_model.input, 
                           outputs=ensemble_output,
                           name='ensemble_model')
    
    # Recompile the ensemble model
    ensemble_model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
    print("Ensemble model created by averaging predictions.")
    return ensemble_model, [inception_model, resnet_model, efficientnet_model]

def train_ensemble_components(models, train_ds, val_ds, epochs=10):
    """
    Trains each component of the ensemble separately.
    """
    histories = []
    
    for i, model in enumerate(models):
        print(f"\nTraining model {i+1}/{len(models)}: {model.name}")
        history = model.fit(train_ds,
                           epochs=epochs,
                           validation_data=val_ds,
                           verbose=1)
        histories.append(history)
        
        # Plot training history for this model
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model.name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model.name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    return histories

# --- Part 3: Prediction and Chatbot Interaction ---

def predict_with_ensemble(ensemble_model, image_path):
    """
    Predicts the class of an uploaded image using the ensemble model.
    """
    print("\n--- 3. Prediction with Ensemble ---")
    
    # Load and preprocess image for InceptionV3 (our ensemble uses Inception input)
    img_batch = load_and_preprocess_single_image(image_path, model_type='inception')
    
    print("Making prediction with ensemble model...")
    predictions = ensemble_model.predict(img_batch)
    predicted_class_idx = np.argmax(predictions[0])
    object_label = CIFAR100_LABELS[predicted_class_idx]
    accuracy_score = predictions[0][predicted_class_idx] * 100
    print(f"Prediction complete. Detected: {object_label} with {accuracy_score:.2f}% confidence.")
    
    # Generate chatbot response
    prompt = f"The image contains: {object_label} with an accuracy of {accuracy_score:.2f}%. Can you describe what is typically happening with this object or in a scene where this object is prominent? Keep it concise and descriptive."
    chatbot_response = generate_chatbot_response(prompt)
    
    return object_label, accuracy_score, chatbot_response

def load_and_preprocess_single_image(image_path, target_size=TARGET_IMG_SIZE, model_type='inception'):
    """
    Loads and preprocesses a single image for the specified model type.
    """
    print(f"Loading and preprocessing single image from: {image_path}")
    img = keras_image.load_img(image_path, target_size=target_size)
    img_arr = keras_image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # Apply model-specific preprocessing
    if model_type == 'inception':
        img_arr = inception_preprocess(img_arr)
    elif model_type == 'resnet':
        img_arr = resnet_preprocess(img_arr)
    elif model_type == 'efficientnet':
        img_arr = efficientnet_preprocess(img_arr)
    
    print(f"Single image preprocessed for {model_type}.")
    return img_arr

def generate_chatbot_response(prompt):
    """
    Generates a descriptive response using the OpenAI GPT-4o-mini model.
    """
    print("Generating chatbot response...")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    chatbot_response = completion.choices[0].message.content
    print("Chatbot response generated.")
    return chatbot_response

# --- Main Execution Flow ---

if __name__ == "__main__":
    # Ensure a fresh Keras session and clean memory at start
    tf.keras.backend.clear_session()
    gc.collect()

    # --- Data Preprocessing ---
    print("--- 1. Data Preprocessing ---")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_cifar100_from_pre_extracted_path(
        extracted_data_path="/content/cifar-100-python"
    )

    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_raw, y_train_raw, test_size=0.2, random_state=42
    )
    print(f"Raw training data shape: {x_train.shape}, Raw validation data shape: {x_val.shape}")

    # Create datasets - we'll use InceptionV3 preprocessing for the ensemble
    BATCH_SIZE = 64
    train_dataset = create_tf_dataset(x_train, y_train, TARGET_IMG_SIZE, BATCH_SIZE, 
                                    model_type='inception', shuffle=True)
    val_dataset = create_tf_dataset(x_val, y_val, TARGET_IMG_SIZE, BATCH_SIZE, 
                                   model_type='inception', shuffle=False)
    test_dataset = create_tf_dataset(x_test_raw, y_test_raw, TARGET_IMG_SIZE, BATCH_SIZE, 
                                    model_type='inception', shuffle=False)

    # --- Model Training ---
    print("\n--- 2. Model Training ---")
    
    # Build the ensemble model and its components
    ensemble_model, component_models = build_ensemble_model((TARGET_IMG_SIZE[0], TARGET_IMG_SIZE[1], 3))
    
    # Train each component model separately
    histories = train_ensemble_components(component_models, train_dataset, val_dataset, epochs=10)
    
    # Evaluate the ensemble model on the test set
    print("\nEvaluating ensemble model on test set...")
    loss, accuracy = ensemble_model.evaluate(test_dataset, verbose=0)
    print(f"\nEnsemble model performance on CIFAR-100 test set:")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # --- Prediction and Chatbot Interaction ---
    print("\n--- Upload an image for ensemble prediction and chatbot description ---")
    uploaded = files.upload()
    image_filename = list(uploaded.keys())[0]
    image_path = f'/content/{image_filename}'

    object_label, accuracy_score, chatbot_description = predict_with_ensemble(ensemble_model, image_path)

    # --- Final Output ---
    print("\n--- Final Results ---")
    print(f"Ensemble model accuracy (on uploaded image): {accuracy_score:.2f}%")
    print(f"Object detection: {object_label}")
    print(f"Chatbot's response: {chatbot_description}")
