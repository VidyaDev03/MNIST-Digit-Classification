"""
MNIST Handwritten Digit Classification using a Neural Network
-------------------------------------------------------------
Runs in VS Code (plain Python file) - no Google Colab needed.

Run from the terminal:
    python mnist_digit_classification.py                 (tests the bundled 3-digit.png)
    python mnist_digit_classification.py my_digit.png    (tests your own image)

Outputs: results are printed in the terminal; graphs are saved as PNG files
         (sample_image.png, confusion_matrix.png) in the same folder.
The first run downloads the MNIST dataset (~11 MB), so it needs internet once.
"""

# %% 1. Import the libraries
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TensorFlow's noisy info messages

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save plots to files; works even without a display window
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

tf.random.set_seed(3)
np.random.seed(3)

# Folder where this script lives (graphs are saved here, sample image is read from here)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# %% 2. Load the MNIST data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("Shapes:", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
print("Training images:", X_train.shape[0], "| Test images:", X_test.shape[0])
print("Image size: 28 x 28, grayscale (1 channel)")
print("Unique labels in train:", np.unique(Y_train))
print("Unique labels in test: ", np.unique(Y_test))

# %% 3. Look at one image and its label (saved as sample_image.png)
plt.figure()
plt.imshow(X_train[25], cmap="gray")
plt.title(f"Training image #25 - label: {Y_train[25]}")
plt.savefig(os.path.join(BASE_DIR, "sample_image.png"))
plt.close()
print("Label of training image #25:", Y_train[25], "(saved sample_image.png)")

# %% 4. Scale pixel values from 0-255 down to 0-1
X_train = X_train / 255
X_test = X_test / 255

# %% 5. Build the neural network
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),                         # 28x28 image -> 784 numbers
        keras.layers.Dense(50, activation="relu"),      # hidden layer 1
        keras.layers.Dense(50, activation="relu"),      # hidden layer 2
        keras.layers.Dense(10, activation="softmax"),   # 10 outputs: digits 0-9
    ]
)
model.summary()

# %% 6. Compile the network
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# %% 7. Train the network
model.fit(X_train, Y_train, epochs=10)

# %% 8. Accuracy on the test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"\nTest accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

# %% 9. Predictions on the test data
Y_pred = model.predict(X_test)  # probabilities for each digit
print("Prediction shape:", Y_pred.shape)
print("Probabilities for first test image:", np.round(Y_pred[0], 4))

Y_pred_labels = np.argmax(Y_pred, axis=1)  # most likely digit for every image
print("First test image -> predicted:", Y_pred_labels[0], "| actual:", Y_test[0])

# %% 10. Confusion matrix (saved as confusion_matrix.png)
conf_mat = confusion_matrix(Y_test, Y_pred_labels)
print("\nConfusion matrix:")
print(conf_mat)

plt.figure(figsize=(12, 7))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.ylabel("True Labels")
plt.xlabel("Predicted Labels")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.close()
print("Saved confusion_matrix.png")


# %% 11. Predictive system: recognise a digit from an image file
def predict_digit(image_path):
    """Read an image file and return the digit the network thinks it is."""
    # Read directly as grayscale (1 channel)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")

    # MNIST digits are WHITE on a BLACK background.
    # If the picture is dark-on-light (e.g. pen on paper), flip it.
    if image.mean() > 127:
        image = 255 - image

    image = cv2.resize(image, (28, 28))   # same size as the training images
    image = image / 255                   # same scaling as the training images
    image = image.reshape(1, 28, 28)      # one image in a batch

    probabilities = model.predict(image)
    return int(np.argmax(probabilities))


# Use the image given on the command line, otherwise the bundled sample
if len(sys.argv) > 1:
    input_image_path = sys.argv[1]
else:
    input_image_path = os.path.join(BASE_DIR, "3-digit.png")

print(f"\nPredicting the digit in: {input_image_path}")
predicted_digit = predict_digit(input_image_path)
print("The Handwritten Digit is recognised as", predicted_digit)


# %% 12. Drawing window: draw a digit with the mouse and the network predicts it
# (Everything above is unchanged. This window opens after the code above has finished.)
print("\nOpening the drawing window ... (close the window to end the program)")

try:
    import tkinter as tk
except ImportError:
    print("tkinter is not available in this Python installation.")
    print("Re-run the Python installer, choose 'Modify' and tick 'tcl/tk and IDLE'.")
    sys.exit(1)

from PIL import Image, ImageDraw

CANVAS_SIZE = 280   # drawing area: 280 x 280 pixels (10x bigger than 28 x 28)
BRUSH_SIZE = 20     # thickness of the pen


def prepare_for_model(pil_image):
    """Turn the 280x280 drawing into a 28x28 image that looks like MNIST digits."""
    pixels = np.array(pil_image)
    ys, xs = np.where(pixels > 20)
    if len(ys) == 0:
        return None  # nothing drawn yet

    # 1) Cut out just the digit
    digit = pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2) Shrink so the longest side is 20 pixels (MNIST digits are 20x20 inside 28x28)
    height, width = digit.shape
    scale = 20.0 / max(height, width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3) Put it in the middle of a 28x28 black image
    image28 = np.zeros((28, 28), dtype=np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    image28[top:top + new_h, left:left + new_w] = digit

    # 4) Move the digit so its centre of mass is in the middle (like MNIST)
    total = image28.sum()
    if total > 0:
        grid_y, grid_x = np.indices(image28.shape)
        centre_y = (grid_y * image28).sum() / total
        centre_x = (grid_x * image28).sum() / total
        shift = np.float32([[1, 0, 14 - centre_x], [0, 1, 14 - centre_y]])
        image28 = cv2.warpAffine(image28, shift, (28, 28))

    return image28 / 255.0  # same 0-1 scaling as the training images


class DigitApp:
    def __init__(self, window):
        self.window = window
        window.title("MNIST Digit Recognizer - draw a digit")
        window.resizable(False, False)

        tk.Label(window, text="Draw a digit (0-9) with the mouse",
                 font=("Arial", 14, "bold")).pack(pady=(10, 5))

        main = tk.Frame(window)
        main.pack(padx=15, pady=5)

        # Black drawing area
        self.canvas = tk.Canvas(main, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="cross",
                                highlightthickness=2, highlightbackground="gray")
        self.canvas.grid(row=0, column=0, padx=(0, 15))

        # Right side: prediction + probability bars
        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky="n")
        tk.Label(right, text="Prediction", font=("Arial", 12)).pack()
        self.digit_label = tk.Label(right, text="-", font=("Arial", 72, "bold"),
                                    fg="#0a7d28", width=2)
        self.digit_label.pack()
        self.confidence_label = tk.Label(right, text="Confidence: -", font=("Arial", 12))
        self.confidence_label.pack(pady=(0, 8))

        self.bars = tk.Canvas(right, width=200, height=170, highlightthickness=0)
        self.bars.pack()

        # Buttons
        buttons = tk.Frame(window)
        buttons.pack(pady=10)
        tk.Button(buttons, text="Clear", width=12, font=("Arial", 11),
                  command=self.clear).grid(row=0, column=0, padx=5)
        tk.Button(buttons, text="Predict", width=12, font=("Arial", 11),
                  command=self.predict).grid(row=0, column=1, padx=5)

        # Hidden copy of the drawing that we give to the model
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.last_x = self.last_y = None

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        self.draw_bars(None)

    def draw(self, event):
        r = BRUSH_SIZE / 2
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill="white", width=BRUSH_SIZE,
                                    capstyle=tk.ROUND, smooth=True)
            self.pen.line([self.last_x, self.last_y, event.x, event.y],
                          fill=255, width=BRUSH_SIZE)
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                                fill="white", outline="white")
        self.pen.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill=255)
        self.last_x, self.last_y = event.x, event.y

    def stop_drawing(self, event):
        self.last_x = self.last_y = None
        self.predict()  # predict automatically when you lift the mouse

    def clear(self):
        self.canvas.delete("all")
        self.pen.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.digit_label.config(text="-")
        self.confidence_label.config(text="Confidence: -")
        self.draw_bars(None)

    def predict(self):
        image28 = prepare_for_model(self.image)
        if image28 is None:
            return
        probabilities = model.predict(image28.reshape(1, 28, 28), verbose=0)[0]
        digit = int(np.argmax(probabilities))
        self.digit_label.config(text=str(digit))
        self.confidence_label.config(text=f"Confidence: {probabilities[digit] * 100:.1f}%")
        self.draw_bars(probabilities)

    def draw_bars(self, probabilities):
        self.bars.delete("all")
        for d in range(10):
            y = d * 17 + 2
            value = 0 if probabilities is None else float(probabilities[d])
            best = probabilities is not None and d == int(np.argmax(probabilities))
            self.bars.create_text(8, y + 7, text=str(d), font=("Arial", 9, "bold"))
            self.bars.create_rectangle(20, y, 20 + 150 * value, y + 14,
                                       fill="#0a7d28" if best else "#9aa5b1", outline="")
            self.bars.create_text(175, y + 7, text=f"{value * 100:.0f}%",
                                  font=("Arial", 9), anchor="w")


root = tk.Tk()
DigitApp(root)
root.lift()
root.attributes("-topmost", True)
root.after(500, lambda: root.attributes("-topmost", False))
root.mainloop()
