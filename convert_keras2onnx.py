# commands:
# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python convert_keras2onnx.py <old_model_name> <new_model_name>
# cd C:\Users\cagat\projects\auto-weapons-ml && activate tf_gpu && python convert_keras2onnx.py old_model new_model
	
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd
import sys
import keras2onnx

print(len(sys.argv), sys.argv[0], sys.argv[1])

prev_model_name = sys.argv[1]
new_model_name = sys.argv[2]

# reading data and one-hot encoding
x_train = pd.read_csv('./data/default.csv')
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
model.load_weights('./trained_models/' + prev_model_name + '.h5')

onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, new_model_name)

# model.save('./trained_models/' + new_model_name + '.h5')
# model_json = model.to_json()
# with open('./trained_models/' + new_model_name + '.json', 'w') as json_file:
#    json_file.write(model_json)
