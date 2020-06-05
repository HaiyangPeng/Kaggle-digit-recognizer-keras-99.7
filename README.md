# Kaggle-digit-recognizer-keras-99.7
Three optional CNN models that achieves testing acc of 99.2%, 99.5%, 99.7%, respectively on MNIST dataset.

The code has detailed interpretation.

Training and validation curves can be seen in weight_and_model/

All the model weights can be seen in weight_and_model/

All the submitted csv can be seen in submission_csv/
## Three models
1. cnn_model
2. cnn_model2
3. cnn_model3 

Detailed model implementation can be seen in mnist_kaggle_train_and_test.py
## Training hyper-parameters
1. Training epochs: 50 or 150
2. Training batch_size : 64
3. Optimizer: Adam(best for this task), RMSprop
## Training Tricks
1. Data augmentation
2. Batch normalization
3. Different activation functions
4. EarlyStopping
5. Learning rate descent while there is no improvement of performance
6. Avoid using image resize
## Dependencies
* TensorFlow 1.12.0
* Keras 2.2.0
* Pandas 1.0.3
* Sklearn 0.22.0
* Numpy
* Matplotlib
* Seaborn
