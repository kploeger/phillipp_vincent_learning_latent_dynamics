import matplotlib.pyplot as plt
import numpy as np

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
print(pd.__version__)


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)


# location of the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

# download dataset csv into pandas dataframe
raw_dataset = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)

dataset = raw_dataset.copy()
print(dataset.tail())

# clean the data
# isna() detects missing values
print(dataset.isna().sum())

# drop the rows with missing values
dataset = dataset.dropna()

# convert categorical column to one-hot encoded columns
dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
print(dataset.tail())
dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
print(dataset.tail())

# split the data into train and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# inspect the data
# sns.pairplot(
# train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
# )
# plt.show()

# overall statistics
print(train_dataset.describe().transpose())

# split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")

# normalize the data
print(train_dataset.describe().transpose()[["mean", "std"]])

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print("First example:", first)
    print()
    print("Normalized:", normalizer(first).numpy())

# deep neural network model
dnn_model = keras.Sequential(
    [
        normalizer,
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)

dnn_model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
print(dnn_model.summary())

history = dnn_model.fit(
    train_features, train_labels, validation_split=0.2, verbose=0, epochs=100
)

# save weights
dnn_model.save_weights("dnn_model.weights.h5")

# create image of model and save it
dot_img_file = "model_1.png"
keras.utils.plot_model(dnn_model, to_file=dot_img_file, show_shapes=True)

plt.close()
plot_loss(history)
plt.show()

test_results = {}
test_results["dnn_model"] = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(test_results)

print(pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T)

# make predictions
test_predictions = dnn_model.predict(test_features).flatten()

plt.close()
a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

plt.close()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

# dnn_model.save("dnn_model.keras")
#
# reloaded = tf.keras.models.load_model("dnn_model.keras")
#
# test_results["reloaded"] = reloaded.evaluate(test_features, test_labels, verbose=0)
# print(pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T)
