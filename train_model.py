
from pathlib import Path
import tensorflow as tf

# 1. Set up the dataset paths
project = Path(__file__).parent
dataset = project / "Dataset"

train_dir = dataset / "train"
validation_dir = dataset / "validation"
test_dir = dataset / "test"

# 2. Set image and training settings
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
SEED = 42

# 3. Load the images
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=SEED
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=SEED,
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# 4. Remember the category names
class_names = train_data.class_names
print("Categories:", class_names)

# 5. Prepare the pretrained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# 6. Build our classifier
inputs = tf.keras.Input(shape=(160, 160, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# 7. Configure training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 8. Train the classifier
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)

# 9. Evaluate on the test images
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test accuracy: {test_accuracy:.2%}")

# 10. Save the trained model
model_path = project / "waste_classifier.keras"
model.save(model_path)

print(f"Model saved to: {model_path}")
