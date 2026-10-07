from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Input
from tensorflow.keras.optimizers import SGD
from sklearn.datasets import make_moons
import numpy as np

X, y = make_moons(n_samples=400, noise=0.20, random_state=1)
y = y.reshape(-1, 1).astype(float)
X = (X - X.mean(0)) / X.std(0)

model = Sequential([Input(shape = (2,)),
                    Dense(16, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy',
              optimizer=SGD(learning_rate=0.5),
              metrics=['accuracy'])

Batch = model.fit(X, y, epochs=200, batch_size=len(X), verbose=0)

SDG = model.fit(X, y, epochs=200, batch_size=16, verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)

print(f"Final Loss: {loss:.4f}")
print(f"Final Accuracy: {accuracy:.4f}")

batch_accuracy = Batch.history['accuracy'][-1]
print(f"Final Batch Training Accuracy: {batch_accuracy:.4f}")

sgd_accuracy = SDG.history['accuracy'][-1]
print(f"Final SGD Training Accuracy: {sgd_accuracy:.4f}")
