import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

CSV_PATH = Path("data_splits.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# -------------------------------
# 2. LOAD DATASET MANIFEST
# -------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError("data_splits.csv not found!")

df = pd.read_csv(CSV_PATH)

required_columns = {"filepath", "label", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError("CSV is missing required columns.")

if not df["filepath"].map(lambda p: Path(p).is_file()).all():
    raise FileNotFoundError("Some image files are missing.")

class_names = sorted(df["label"].unique())
num_classes = len(class_names)

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

df["class_index"] = df["label"].map(class_to_index)

print("\nClasses:", num_classes)
print("Total images:", len(df))

# Save class names for later prediction
with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# -------------------------------
# 3. CLASS WEIGHTS
# -------------------------------

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_df["class_index"].values
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}

print("\nClass weights calculated.")

# -------------------------------
# 4. IMAGE LOADING
# -------------------------------

def load_image(filepath, label):
    image = tf.io.read_file(filepath)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)

    return image, label


def create_dataset(dataframe, shuffle=False):
    paths = dataframe["filepath"].astype(str).values
    labels = dataframe["class_index"].values.astype("int32")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(dataframe), 2000),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


print("\nPreparing image datasets...")

train_ds = create_dataset(train_df, shuffle=True)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)

# -------------------------------
# 5. BUILD MOBILENETV2 MODEL
# -------------------------------

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# 6. TRAIN MODEL
# -------------------------------

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_leaf_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

print("\nStarting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save final training history
history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(MODEL_DIR / "training_history.json", "w") as file:
    json.dump(history_data, file, indent=4)

# -------------------------------
# 7. TEST SET EVALUATION
# -------------------------------

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_DIR / "best_leaf_model.keras"
)

print("\nEvaluating best model on test dataset...")

test_loss, test_accuracy = best_model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

predictions = best_model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

actual_classes = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        actual_classes,
        predicted_classes,
        labels=np.arange(num_classes),
        target_names=class_names,
        zero_division=0
    )
)

matrix = confusion_matrix(
    actual_classes,
    predicted_classes,
    labels=np.arange(num_classes)
)

np.save(MODEL_DIR / "confusion_matrix.npy", matrix)

print("\nTraining and evaluation completed!")
print("Best model saved in:", MODEL_DIR / "best_leaf_model.keras")