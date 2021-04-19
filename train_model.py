# commands:
# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python train_model.py <training_data_name> <load previous weights> ..
# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python train_model.py <training_data_name> True  <prev_model_name> <new_model_name>
# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python train_model.py <training_data_name> False <new_model_name>

# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python train_model.py simulated12k_17-17-34_3-1-2021 False new_model
# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python train_model.py simulated14k_21-22-44_2-1-2021 False new_model
# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python train_model.py <training_data_name> False <new_model_name>

from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import numpy as np
import pandas as pd
import sys

print(len(sys.argv), sys.argv[0], sys.argv[1])

training_data_name = sys.argv[1]
load_prev_weights = (sys.argv[2] == 'True')
if (load_prev_weights):
    prev_model_name = sys.argv[3]
    new_model_name = sys.argv[4]
else:
    new_model_name = sys.argv[3]

# reading data and one-hot encoding
x_train = pd.read_csv('./data/' + training_data_name + '.csv')
y_train = x_train['weight']
x_train.drop(['weight'], axis=1, inplace=True)

dummy = pd.read_csv("./data/dummy.csv")
dummy_x = dummy.drop(['weight'], axis=1)
dummy_y = dummy['weight']
x_train = pd.concat([x_train, dummy_x], ignore_index=True)
y_train = pd.concat([y_train, dummy_y], ignore_index=True)

x_train = pd.get_dummies(x_train, columns=x_train.columns)

# building
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(250,
                           activation='relu', input_shape=[x_train.shape[1]]),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(250,
                           activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(250,
                           activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(250,
                           activation='relu'),
        keras.layers.Softmax(),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(0.0002)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()
if (load_prev_weights):
    model.load_weights('./trained_models/' + prev_model_name + '.h5')

# training
checkpoint_filepath = './trained_models/checkpoint/'
# def lr_scheduler(epoch, lr):
#     if epoch > 30:
#         lr = 0.0003
#     if epoch > 40:
#         lr = 0.0004
#     if epoch > 50:
#         lr = 0.0005
#     if epoch > 60:
#         lr = 0.0006
#     if epoch > 70:
#         lr = 0.0007
#     if epoch > 100:
#         lr = 0.001
#     return lr
def lr_scheduler(epoch, lr):
    if epoch > 10:
        lr = 0.001
    if epoch > 40:
        lr = 0.0004
    if epoch > 50:
        lr = 0.0003
    if epoch > 60:
        lr = 0.0002
    if epoch > 70:
        lr = 0.0001
    if epoch > 100:
        lr = 0.00005
    return lr

nnr_callbacks = [
    EarlyStopping(patience=10),
    LearningRateScheduler(lr_scheduler, verbose=0),
    #ModelCheckpoint(filepath=checkpoint_filepath,
    #                save_weights_only=True,
    #                monitor='val_loss',
    #                save_best_only=True)
]
EPOCHS = 200
history = model.fit(
    np.array(x_train), np.array(y_train),
    epochs=EPOCHS, validation_split= 0.2, verbose=2, batch_size=512,
    callbacks=nnr_callbacks)

model.save('./trained_models/' + new_model_name + '.h5')
model_json = model.to_json()
with open('./trained_models/' + new_model_name + '.json', 'w') as json_file:
    json_file.write(model_json)
