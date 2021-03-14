import os
import tensorflow as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import tensorflow.keras as keras
import datetime
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

DATADIR = 'MGR_DATASET/SPECTO/'
CATEGORIES = ['nonAFIB', 'AFIB']
IMG_SIZE = 50

training_data = []

dense_layers = [2]#[0, 1, 2]
layer_sizes = [64] #[32, 64, 128]
conv_layers = [3]#[1, 2, 3]


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
print(len(training_data))
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

# TODO BALANCED
classes = np.array([training_data[i][1] for i in range(len(training_data))])
pos = len(np.where(classes==1)[0])
neg = len(np.where(classes==0)[0])
print('nr of positives : ', pos)
print('nr of negatives : ', neg)
total = pos+neg

counter = 0
for features, label in training_data:
    if label==0:
        if counter<pos:
            X.append(features)
            y.append(label)
            counter+=1
    else:
        X.append(features)
        y.append(label)
# for features, label in training_data:
#     X.append(features)
#     y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0
y = np.array(y)

print('nr of positives in balanced : ', len(np.where(y==1)[0]))
print('nr of negatives in balanced : ', len(np.where(y==0)[0]))

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

# CALLBACK
class PredictionCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(self.validation_data[0])
    print('prediction: {} at epoch: {}'.format(y_pred, epoch))

from sklearn.model_selection import train_test_split
frac_test_split = 0.1

inputs = X
targets = y
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=frac_test_split)

# Save and load temporarily
np.save('./data.npy', (X_train, X_test, y_train, y_test))
X_train, X_test, y_train, y_test = np.load('./data.npy', allow_pickle=True)
print()
print('positive in test : ', len(np.where(y_test==1)[0]))

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = 'SPECTROGRAM_AFDB_BALANCED_conv-{}-nodes-{}-dense-{}-data-{}'.format(conv_layer, layer_size, dense_layer, str(time.time()))
            print(NAME)
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=X_train.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for i in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            for j in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer="adam",
                          metrics=METRICS)
            model.summary()
            # checkpoint_path = "training_1/cp.ckpt"
            # checkpoint_dir = os.path.dirname(checkpoint_path)
            # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
            #                                                  save_weights_only=True,
            #                                                  verbose=1)

            model.fit(X_train, y_train, batch_size=1000, validation_split=0.1, epochs=5, callbacks=[tensorboard])


def plot_diagnostic_curves(proba, y_true):
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, axarr = plt.subplots(1, 2)
    axarr[0].plot([0, 1], [0, 1], 'k--')
    axarr[0].plot(fpr, tpr)
    axarr[0].set_title('ROC curve')
    axarr[0].set_xlabel('FPR')
    axarr[0].set_ylabel('TPR')
    axarr[0].set_aspect('equal')
    axarr[0].grid(True)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, proba)
    axarr[1].plot([0, 1], [0, 1], 'k--')
    axarr[1].plot(recall, precision)
    axarr[1].set_title('Precision-Recall curve')
    axarr[1].set_xlabel('Recall')
    axarr[1].set_ylabel('Precision')
    axarr[1].set_aspect('equal')
    axarr[1].grid(True)


predictions = model.predict(X_test)
plot_diagnostic_curves(predictions, y_test)
plt.savefig(f'figs/SPECTROGRAM/diagnostics_curves.png')


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p, normalize='true')
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".2f")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


baseline_results = model.evaluate(X_test, y_test, batch_size=1000, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(y_test, predictions)
plt.savefig(f'figs/SPECTROGRAM/conf_matrix_on_test_set.png')