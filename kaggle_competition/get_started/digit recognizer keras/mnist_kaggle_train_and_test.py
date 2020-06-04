"""
@software: PyCharm
@file: SegNet.py
@author: Haiyang Peng
@time: 2020/06/04 14:58
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(2)  # set for a fixed random sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

# Keras lib
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU

"""
1.Load the data
"""
train = pd.read_csv("./mnist_data/train.csv")
test = pd.read_csv("./mnist_data/test.csv")

"""
2.Data pre-processing
"""
Y_train = train["label"]  # series
# Y_train = train.iloc[:, 0:1]  # if you need column name (dataframe)
print(Y_train)
print("\n")

# Only acquire the image pixels
X_train = train.drop(labels=["label"], axis=1)

# Plot the counting statistics for digit classes
plt.figure(figsize=(12, 8))
sns.set(style='white', context='notebook', palette='deep')
# sns.countplot(x="label", data=Y_train)  # if you need column name (dataframe)
sns.countplot(Y_train)
plt.show()

# Another counting method
print(Y_train.value_counts())  # only for series
print("\n")

# Check the data
# count(non-nan) unique(only one) top(freq_highest) freq(highest_freq)
print(X_train.isnull().any().describe())
print("\n")

print(test.isnull().any().describe())
print("\n")

# Normalize the data to the range of [0, 1]
# If you don't know the prior data range, use (x-x_min)/(x_max-x_min)

X_train = X_train / 255.0
test = test / 255.0

# Reshape data to a tensor shape
# If you consider a MLP model, the original shape is okay
# Notice the reshape order

X_train = X_train.values.reshape(-1, 28, 28, 1)  # values
test = test.values.reshape(-1, 28, 28, 1)

# Generate a one-hot label
Y_train = to_categorical(Y_train, num_classes=10)

# Set the random seed
random_seed = 2

# Split the (0.9) train and the (0.1) validation set for training
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# Show some examples after normalization
# plt.figure(figsize=(60, 40))  # useless for a large figure
fig, axes = plt.subplots(5, 5, figsize=(4, 4))
fig.suptitle("Digit examples after normalization", fontsize=8)  # setting the main title
for i in range(5):
    for j in range(5):
        axes[i, j].imshow(X_train[i][:, :, 0])
        axes[i, j].set_title("Label: {}".format(Y_train[i].argmax()), fontsize=7)  # setting the title of subplots
plt.show()

# Apply the technique of data augmentation to prevent over-fitting
# referred to https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
print("shape of X_train", X_train.shape)

# Some examples after augmentation
# plt.figure(figsize=(60, 40))  # useless for a large figure
fig, axes = plt.subplots(4, 1, figsize=(4, 4))
fig.suptitle("Digit examples after augmentation", fontsize=8)  # setting the main title
batch = []
i = 0
for batch_array in datagen.flow(X_train[0][:, :, 0].reshape(1, 28, 28, 1), batch_size=1, seed=random_seed):
    batch_array = batch_array.reshape(28, 28)  # for gray image
    if i == 4:
        break
    i += 1
    batch.append(batch_array)

for i in range(4):
    axes[i].imshow(batch[i])
    axes[i].set_title("Label: {}".format(Y_train[0].argmax()), fontsize=7)  # setting the title of subplots
plt.show()

"""
3.Build some CNN models
"""
# A sequential model referred to https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# However, I just achieved 0.99200 testing accuracy(top30%) after 150 epochs
# The submitted csv: Sorry, I lost this file
def cnn_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                     activation='relu', input_shape=(28, 28, 1)))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # shape(14, 14, 32)

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # shape(7, 7, 64)

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation="softmax"))  # shape(,10)

    return model

# A functional model that adds BN layers
# I achieved 0.99471 testing accuracy(top17%) after 50 epochs
# The submitted csv: ./submission_csv/cnn_mnist_keras2.csv
def cnn_model2(input_shape, classes, include_top):
    inputs = Input(shape=input_shape)
    # Block 1
    x = Conv2D(32, (5, 5),
                      padding='same',
                      name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(32, (5, 5),
                      padding='same',
                      name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(name='block1_pool')(x)  # shape(14, 14, 32)

    x = Dropout(rate=0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3),
               padding='same',
               name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3),
               padding='same',
               name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(name='block2_pool')(x)  # shape(7, 7, 64)

    x = Dropout(rate=0.25)(x)

    # Block 3
    x = Conv2D(64, (3, 3),
               padding='same',
               name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dropout(rate=0.25)(x)  # shape(7, 7, 64)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)

        x = Dense(256, name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Dropout(rate=0.15)(x)

        x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs, x, name='cnn_model2')

# A functional model that contains two fc layers and removes the third block in cnn_model2
# I achieved 0.99671 testing accuracy(top9%) after 50 epochs
# The submitted csv: ./submission_csv/cnn_mnist_keras3.csv
def cnn_model3(input_shape, classes, include_top):
    inputs = Input(shape=input_shape)
    # Block 1
    x = Conv2D(32, (5, 5),
                      padding='same',
                      name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(32, (5, 5),
                      padding='same',
                      name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(name='block1_pool')(x)  # shape(14, 14, 32)

    x = Dropout(rate=0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3),
               padding='same',
               name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3),
               padding='same',
               name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(name='block2_pool')(x)  # shape(7, 7, 64)

    x = Dropout(rate=0.25)(x)

    # Block 3
    # x = Conv2D(64, (3, 3),
    #            padding='same',
    #            name='block3_conv1')(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    #
    # x = Dropout(rate=0.25)(x)  # shape(7, 7, 64)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)

        x = Dense(512, name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Dropout(rate=0.15)(x)

        x = Dense(256, name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs, x, name='cnn_model3')


"""
4.Select one model and hyper-parameters for training
"""
model = cnn_model3(input_shape=(28, 28, 1), classes=10, include_top=True)

# Select the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optimizer = Adadelta()
# optimizer = Adam(lr=0.001)

# Training hyper-parameters
epochs = 50
batch_size = 64

# Compile the selected model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Set checkpoint and Tensorboard to record weights and model
checkpoint = ModelCheckpoint(
    filepath='./weight_and_model/weights5.h5',
    monitor='acc',
    mode='auto',
    save_best_only='True',
    save_weights_only='False'  # save weight and model graph
)

tensorboard = TensorBoard(log_dir='./weight_and_model/tensorboard_record5')

# Set a learning rate annealer for jumping out of local minimum
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Set earlystopping for avoiding running excessive epochs
# Notice that I did not use this trick, since # epochs is not big
earlystopping = EarlyStopping(monitor='val_acc', min_delta=0,
                             patience=10, verbose=1, mode='auto')


"""
5.Model training
"""
# Fit the model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_val, Y_val),
                              verbose=1, steps_per_epoch=X_train.shape[0] // batch_size
                              , validation_steps=X_val.shape[0] // batch_size, callbacks=[checkpoint, tensorboard, learning_rate_reduction])


"""
6.Results analysis
"""
# Plot the acc, loss curves of training and validation stages
print(history.history.keys())

# Summarize the history for acc
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy curves')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# Summarize the history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss curves')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# Look at confusion matrix
# referred to https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# Actually, you can use a confusion lib :heatmap to draw
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()  # automatically adjust subplots to the best location
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)

# Convert one-hot like vector to the specified class
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Convert one-hot like vector to the specified class
Y_true = np.argmax(Y_val, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
print("shape of confusion_mtx:", confusion_mtx.shape)

# Plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(10))

# Display some error results
# Errors are differences between the predicted and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]


# Plot the misclassified validation samples
# referred to https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1
    plt.show()


# Probabilities of the incorrectly predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted and true labels
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors
most_important_errors = sorted_dela_errors[-6:]  # the last six numbers

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)

# Predict results for the unseen testing dataset
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")
print("results: ", results)

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv("./submission_csv/cnn_mnist_keras5.csv", index=False)  # do not save index (i.e. row name)
