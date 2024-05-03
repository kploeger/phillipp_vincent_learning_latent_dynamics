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
    plt.ylabel("Error [joint_state_effort]")
    plt.legend()
    plt.grid(True)


# location of the dataset
# column_names = [
#     "joint_state_pos",
#     "joint_state_vel",
#     "joint_state_acc",
#     "joint_state_effort",
# ]

# download dataset csv into pandas dataframe
raw_dataset = pd.read_csv(
    "testset.csv",
    # names=column_names,
    # na_values="?",
    # comment="\t",
    # sep=",",
    # skipinitialspace=True,
)

dataset = raw_dataset.copy()
print(dataset.tail())

# select last 4 columns
# dataset = dataset.iloc[:, -4:]
# print(dataset.tail())

# clean the data
# isna() detects missing values
print(dataset.isna().sum())

# drop the rows with missing values
dataset = dataset.dropna()

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

train_labels = pd.concat(
    [
        train_features.pop("joint_state_effort_joint1"),
        train_features.pop("joint_state_effort_joint2"),
        train_features.pop("joint_state_effort_joint3"),
        train_features.pop("joint_state_effort_joint4"),
    ],
    axis=1,
)
print(train_labels.tail())
test_labels = pd.concat(
    [
        test_features.pop("joint_state_effort_joint1"),
        test_features.pop("joint_state_effort_joint2"),
        test_features.pop("joint_state_effort_joint3"),
        test_features.pop("joint_state_effort_joint4"),
    ],
    axis=1,
)

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
        layers.Dense(4),
    ]
)

dnn_model.compile(
    loss="mean_absolute_error", optimizer=tf.keras.optimizers.legacy.Adam(0.001)
)
print(dnn_model.summary())

history = dnn_model.fit(
    train_features, train_labels, validation_split=0.2, verbose=0, epochs=100
)

plt.close()
plot_loss(history)
plt.show()

test_results = {}
test_results["dnn_model"] = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(test_results)

print(pd.DataFrame(test_results, index=["Mean absolute error [joint_state_effort]"]).T)

# make predictions
test_predictions = dnn_model.predict(test_features).flatten()

plt.close()
a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [joint_state_effort_joint1]")
plt.ylabel("Predictions [joint_state_effort_joint1]]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

print(dnn_model.predict(test_features[:10]))

# plt.close()
# a = plt.axes(aspect="equal")
# plt.scatter(test_labels[:, 1], test_predictions[:, 1])
# plt.xlabel("True Values [joint_state_effort_joint2]")
# plt.ylabel("Predictions [joint_state_effort_joint2]]")
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)
# plt.show()
#
# plt.close()
# a = plt.axes(aspect="equal")
# plt.scatter(test_labels[:, 2], test_predictions[:, 2])
# plt.xlabel("True Values [joint_state_effort_joint3]")
# plt.ylabel("Predictions [joint_state_effort_joint3]]")
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)
# plt.show()
#
# plt.close()
# a = plt.axes(aspect="equal")
# plt.scatter(test_labels[:, 3], test_predictions[:, 3])
# plt.xlabel("True Values [joint_state_effort_joint4]")
# plt.ylabel("Predictions [joint_state_effort_joint4]]")
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)
# plt.show()

# plt.close()
# error = test_predictions - test_labels
# plt.hist(error, bins=25)
# plt.xlabel("Prediction Error [joint_state_effort_joint4]")
# _ = plt.ylabel("Count")
# plt.show()