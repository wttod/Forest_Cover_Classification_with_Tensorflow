import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

# read data
data = pd.read_csv("cover_data.csv")

# split features and labels
features = data.iloc[:, 0:54]
labels = data.iloc[:, -1]

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

ct = ColumnTransformer([("standardize", StandardScaler(), features.columns)], remainder="passthrough")

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

model = Sequential()
model.add(keras.Input(shape=(features.shape[1],)))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(1, activation="relu"))
print(model.summary())

learning_rate = 1e-3
num_epochs = 250
batch_size = 440

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss="mse", metrics=["mae"], optimizer=opt)
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=25)
history = model.fit(features_train_scaled, labels_train, epochs=num_epochs, verbose=1,
                    batch_size=batch_size, validation_split=0.3, callbacks=[es])

val_mse, val_mae = model.evaluate(features_test_scaled, labels_test)
print(f"MSE: {val_mse}\nMAE: {val_mae}")

label_predicted = model.predict(features_test_scaled)
print(f"r2 score: {r2_score(labels_test, label_predicted)}")

loss, acc = model.evaluate(features_test, labels_test)
print(f"Loss: {loss}\nAccuracy: {acc}")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

ax1.plot(history.history["mae"], label="Training")
ax1.plot(history.history["val_mae"], label="Validation")
ax1.set_title("MAE Model")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MAE")
ax1.legend(loc="best")

ax2.plot(history.history["loss"], label="Training")
ax2.plot(history.history["val_loss"], label="Validation")
ax2.set_title("Loss Model")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(loc="best")
plt.show()
