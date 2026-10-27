
from pathlib import Path
import sys
import tensorflow as tf

# Find and load the saved model
project = Path(__file__).parent
model_path = project / "waste_classifier.keras"
model = tf.keras.models.load_model(model_path)

# Get the image path from the command line
if len(sys.argv) < 2:
    print("Usage: python predict.py \"path-to-image.jpg\"")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print("Error: Image file not found.")
    sys.exit(1)

# Load and prepare the image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(160, 160)
)

image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, axis=0)

# Predict the category
predictions = model.predict(image_array, verbose=0)
class_names = ["cardboard", "paper", "plastic"]

predicted_index = int(tf.argmax(predictions[0]).numpy())
predicted_class = class_names[predicted_index]
confidence = float(predictions[0][predicted_index])

print(f"Predicted category: {predicted_class}")
print(f"Model confidence: {confidence:.2%}")
