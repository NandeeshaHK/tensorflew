dl1 = '''
import numpy as np

# Input and target
X = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 0]
])
T = np.array([1, 1, 0, 0])

# Parameters
epochs, lr = 10, 0.1

# Activation function (step)
activation = lambda x: np.where(x >= 0, 1, 0)

# Hebbian Learning
W_hebb = np.sum([x * t for x, t in zip(X, T)], axis=0)

# Perceptron Learning
W_perceptron = np.zeros(X.shape[1])
for _ in range(epochs):
    for x, t in zip(X, T):
        y = activation(np.dot(x, W_perceptron))
        W_perceptron += lr * (t - y) * x

# Delta Learning (Widrow-Hoff Rule)
W_delta = np.zeros(X.shape[1])
for _ in range(epochs):
    for x, t in zip(X, T):
        y = np.dot(x, W_delta)
        W_delta += lr * (t - y) * x

# Correlation Learning (dot product for single output)
W_corr = np.dot(T, X)

# OutStar Learning (based on convergence to T)
W_outstar = np.random.rand(len(T))
for _ in range(epochs):
    W_outstar += lr * (T - W_outstar)

# Output results
print("Hebbian Weights:", W_hebb)
print("Perceptron Weights:", W_perceptron)
print("Delta Rule Weights:", W_delta)
print("Correlation Weights:", W_corr)
print("OutStar Weights:", W_outstar)'''

dl2 = '''import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Plotting
def plot_activation(x, y, title):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    plt.show()

# Generate input range
x = np.linspace(-5, 5, 100)

# Plot each activation function
plot_activation(x, sigmoid(x), 'Sigmoid')
plot_activation(x, tanh(x), 'Tanh')
plot_activation(x, relu(x), 'ReLU')
plot_activation(x, leaky_relu(x), 'Leaky ReLU')

# Softmax needs special handling (works on vectors)
x_soft = np.array([1.0, 2.0, 3.0])
print("Softmax output:", softmax(x_soft))'''

dl3 = '''
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Input: [Favorite hero, Exam, Climate]
X = np.array([
    [1, 0, 1],  # Scenario 1
    [0, 1, 1],  # Scenario 2
    [1, 1, 0],  # Scenario 3
    [0, 0, 1],  # Scenario 4
    [1, 1, 1]   # Scenario 5
])

# Expected output (1 = Go for movie, 0 = Don't go)
y = np.array([1, 1, 1, 0, 0])

# Create and configure the Perceptron model
model = Perceptron(max_iter=1000, eta0=1.0, fit_intercept=True)
model.fit(X, y)

# Predict using the same inputs
y_pred = model.predict(X)

# Print detailed results
print("Input\t\t", "Prediction\t\t", "Expected\t\t", "Result")
print("="*80)
for i in range(len(X)):
    input_str = str(X[i])
    pred = y_pred[i]
    expected = y[i]
    result = "CORRECT" if pred == expected else "WRONG"
    print(f"{input_str} \t\t{pred}\t\t {expected}\t\t {result}")

# Calculate accuracy
acc = accuracy_score(y, y_pred)
print("\nAccuracy: {:.2f}%".format(acc * 100))'''
dl4 = '''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread(r'C:\Users\ASUS\Desktop\6th sem\DL\DL Lab\sample.png', cv2.IMREAD_GRAYSCALE)

# 1. Histogram Equalization
equalized = cv2.equalizeHist(img)

# 2. Thresholding (Binary)
_, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 3. Edge Detection (Canny)
edges = cv2.Canny(img, 100, 200)

# 4. Data Augmentation (Flip)
flipped = cv2.flip(img, 1)  # Horizontal flip

# 5. Morphological Operation (Closing)
kernel = np.ones((5,5), np.uint8)
morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Display results
titles = ['Original', 'Equalized', 'Thresholded', 'Edges', 'Flipped', 'Morphological']
images = [img, equalized, thresholded, edges, flipped, morph]

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()'''

dl5 = '''
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Load and preprocess images
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img[tf.newaxis, :]

content_img = load_img('content.jpg')
style_img = load_img('style.jpg')
style_img = tf.image.resize(style_img, content_img.shape[1:3])

# Style transfer
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_img = model(content_img, style_img)[0]

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1).imshow(content_img[0])
plt.subplot(1, 3, 2).imshow(style_img[0])
plt.subplot(1, 3, 3).imshow(stylized_img.numpy().squeeze(0))
plt.axis('off')
plt.show()'''

dl6 = '''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Normalize and one-hot encode
X_train, X_test = x_train/255.0, x_test/255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Improved model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with learning rate schedule
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train with early stopping
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=64, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# Prediction example (unchanged from original)
new_img = x_test[10]
plt.imshow(new_img)
plt.axis("off")
plt.show()

img = np.expand_dims(new_img/255.0, axis=0)
pred = model.predict(img)
prediction = np.argmax(pred)
print(f"Predicted class: {classes[prediction]}")
print(f"Confidence: {np.max(pred)*100:.2f}%")
'''
dl7 = '''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train[..., np.newaxis]/255.0, X_test[..., np.newaxis]/255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# 2. Build CNN model
model = Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# 4. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 5. Predict and visualize
sample_images = X_test[:5]
predictions = np.argmax(model.predict(sample_images), axis=1)

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')
    plt.title(f"Pred: {predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()'''