import math
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow

from os.path import exists
from tensorflow.keras.backend import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, LSTM, Reshape
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.metrics import MAE, MSE, RootMeanSquaredError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import TimeDistributed as TD

# Returns the fit model and history to be saved (and plotted) as desired
def run_model(model, X_train, y_train, X_test, y_test, batch_size, epochs, 
              early_stop_patience=5, verbose=1):
    checkpoint_filepath = '../models/checkpoint'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, verbose=verbose,
                                                save_best_only=True, save_weights_only=False)
    callbacks = [model_checkpoint_callback]
    if early_stop_patience != None: # early_stop_patience can be 0
        stop_early = EarlyStopping(patience=early_stop_patience)
        callbacks.append(stop_early)
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size, 
        epochs=epochs, 
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        verbose=verbose,
        workers=4,
        use_multiprocessing=True,
    )
    best_model = load_model(checkpoint_filepath)
    best_results = best_model.evaluate(X_test, y_test, batch_size=batch_size, return_dict=True)
    shutil.rmtree(checkpoint_filepath)
    return best_model, best_results, history


# Records model result metrics and supplied hyperparameters
# Loads, adds to, then saves the shared model history CSV
# Returns the path to where the model is saved.
# Called in a modeling loop
def save_model(model_directory, model, results, batch_size, dual_outputs, telemetry_columns,
               dataset_directory, scaler_file=None, dropbox_model_directory=None):
    
    model_history_file = f'{model_directory}/model_history.csv'
    model_index = 0
    history_columns = ['model_file', 'batch_size', 'telemetry_columns', 'dataset_directory'
                       'scaler_file', 'mse_score', 'rmse_score', 'mae_score']
    ## Make sure model history exists
    if not exists(model_history_file):
        model_history = pd.DataFrame(columns=history_columns)
    else:
        model_history = pd.read_csv(model_history_file, index_col=0)
        model_index = max(0, model_history.index.max() + 1)
        
    model_file = f'model_{model_index}.h5'
    model_path = f'{model_directory}/{model_file}'

    history_dictionary = {
        'model_file': model_file,
        'batch_size': batch_size,
        'telemetry_columns': '[' + ', '.join(f"'{t}'" for t in telemetry_columns) + ']',
        'dataset_directory': dataset_directory,
        'scaler_file': scaler_file,
    }
    if dual_outputs:
        history_dictionary['mse_score'] = (
            results['val_steering_outputs_loss'],
            results['val_throttle_outputs_loss'],
        )
        history_dictionary['mae_score'] = (
            results['val_steering_outputs_mae'],
            results['val_throttle_outputs_mae'],
        )
        history_dictionary['rmse_score'] = (
            results['val_steering_outputs_root_mean_squared_error'],
            results['val_throttle_outputs_root_mean_squared_error'] 
        )
    else:
        history_dictionary['mse_score'] = (results['loss'])
        history_dictionary['mae_score'] = (results['mae'])
        history_dictionary['rmse_score'] = (results['root_mean_squared_error'])
        
    model_history = model_history.append(history_dictionary, ignore_index=True)
    ## Saving as h5 for backwards compatibility
    model.save(model_path, include_optimizer=False, save_format='h5')
    model_history.to_csv(model_history_file, index_label='model_index')
    
    if dropbox_model_directory:
        dropbox_model_directory = f'/home/grant/Dropbox/Projects/DonkeyCar/models'
        dropbox_model_history_file = f'{dropbox_model_directory}/model_history.csv'
        dropbox_model_path = f'{dropbox_model_directory}/{model_file}'
        model.save(dropbox_model_path, include_optimizer=False, save_format='h5')
        model_history.to_csv(dropbox_model_history_file, index_label='model_index')
        
    return model_path
    
    
# def save_model(model_directory, model, results, batch_size, dual_outputs, scaler_file, telemetry_columns):
#     model_history_file = '../models/model_history.csv'
#     model_index = 0
#     ## Make sure model history exists
#     if not exists(model_history_file):
#         model_history = pd.DataFrame(columns=['model_file', 'scaler_file', 'batch_size', 'telemetry_columns'
#                                               'mse_score', 'rmse_score'])
#     else:
#         model_history = pd.read_csv(model_history_file, index_col=0)
#         model_index = max(0, model_history.index.max() + 1)
        
#     model_file = f'model_{model_index}.h5'
#     model_path = f'{model_directory}/{model_file}'
    
#     history_dictionary = {
#         'model_file': model_file,
#         'scaler_file': scaler_file,
#         'batch_size': batch_size,
#         'telemetry_columns': '[' + ', '.join(f"'{t}'" for t in telemetry_columns) + ']'
#         # 'history': results.history, # uncomment for lots of data
#     }
#     if dual_outputs:
#         history_dictionary['mae_score'] = (
#             results.history['val_steering_outputs_mae'][-1],
#             results.history['val_throttle_outputs_mae'][-1],
#         )
#         history_dictionary['mse_score'] = (
#             results.history['val_steering_outputs_loss'][-1],
#             results.history['val_throttle_outputs_loss'][-1]
#         )
#         history_dictionary['rmse_score'] = (
#             results.history['val_steering_outputs_root_mean_squared_error'][-1],
#             results.history['val_throttle_outputs_root_mean_squared_error'][-1] 
#         )
#     else:
#         history_dictionary['mae_score'] = (
#             results.history['val_mae'][-1],
#         )
#         history_dictionary['mse_score'] = (
#             results.history['val_loss'][-1],
#         )
#         history_dictionary['rmse_score'] = (
#             results.history['val_root_mean_squared_error'][-1],
#         )
#     model_history = model_history.append(history_dictionary, ignore_index=True)
#     ## Saving as h5 for backwards compatibility
#     model.save(model_path, save_format='h5')
#     model_history.to_csv(model_history_file, index_label='model_index')
#     return model_path

# This is a helper function called to plot each model in a traiing loop
# Plotted values:
# - Mean Squared Error (MSE) <- This is the loss function used for fitting
# - Mean Absolute Error (MAE)
# - Root Mean Squared Error (RMSE)
# Each model within the loop will have a different batch size
def plot_metrics(history, batch_size, dual_outputs):
    fig, ax = plt.subplots(1, 3, figsize=(20,5))
    fig.suptitle(f'Metrics for Batch Size {batch_size}', size=15)
    ax[0].set_title('Loss (MSE)', size=13)
    ax[0].plot(history['loss'], label = 'MSE')
    ax[0].plot(history['val_loss'], label = 'MSE')
    if dual_outputs:
        ax[0].plot(history['val_steering_outputs_loss'], label = 'Val Str MSE')
        ax[0].plot(history['val_throttle_outputs_loss'], label = 'Val Thr MSE')
    ax[0].legend()
    ax[1].set_title('RMSE', size=13)
    if dual_outputs:
        ax[1].plot(history['val_steering_outputs_root_mean_squared_error'], label = 'Val Str RMSE')
        ax[1].plot(history['val_throttle_outputs_root_mean_squared_error'], label = 'Val Thr RMSE')
    else:
        ax[1].plot(history['root_mean_squared_error'], label = 'RMSE')
        ax[1].plot(history['val_root_mean_squared_error'], label = 'Val RMSE')
    ax[1].legend()
    ax[2].set_title('MAE', size=13)
    if dual_outputs:
        ax[2].plot(history['val_steering_outputs_mae'], label = 'Val Str MAE')
        ax[2].plot(history['val_throttle_outputs_mae'], label = 'Val Thr MAE')
    else:
        ax[2].plot(history['mae'], label = 'MAE')
        ax[2].plot(history['val_mae'], label = 'Val MAE')
    ax[2].legend()
    plt.tight_layout();
    

# -------------------------------------------------------------------
# The following code was taken from the Donkey Car Github repository: 
# https://github.com/autorope/donkeycar/blob/dev/donkeycar/parts/keras.py
#
# See also:
# https://docs.donkeycar.com/dev_guide/model/
#
# Minor changes have been made to fit my specific input and outputs
# -------------------------------------------------------------------

# This function approximates KerasIMU create_model()
# It consolidates the default_imu() method with helper core_cnn_layers()
# It has been modfied to allow specification of output layer configuration,
# as well as to adjust the telemetry input layer to suit a wider range of
# input sizes.
def create_donkey_vimu(img_input_shape, tel_input_shape, dual_outputs, *args, **kwargs):
    
    # power of 2 by convention
    tel_nodes = 2**(tel_input_shape[0] - 1).bit_length()

    drop = 0.2
    # telem_drop = 0.1
    
    img_in = Input(shape=img_input_shape, name='img_in') 
    x = img_in
    x = Rescaling(1./255, name='img_rescale')(x)
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    tel_in = Input(tel_input_shape, name='tel_in')
    y = tel_in
    y = Dense(tel_nodes, activation='relu', name='dense_3')(y)
    y = Dense(tel_nodes, activation='relu', name='dense_4')(y)
    y = Dense(tel_nodes, activation='relu', name='dense_5')(y)

    z = concatenate([x, y])
    z = Dense(50, activation='relu', name='dense_6')(z)
    z = Dropout(drop)(z)
    z = Dense(50, activation='relu', name='dense_7')(z)
    z = Dropout(drop)(z)
    
    if dual_outputs:
    # two outputs for angle and throttle
        outputs = [
            Dense(1, activation='linear', name='steering_outputs')(z),
            Dense(1, activation='linear', name='throttle_outputs')(z)
        ]
    else:
        # combined output
        outputs = Dense(2, activation='linear', name='combined_outputs')(z)

    model = Model(inputs=[img_in, tel_in], outputs=outputs)
    return model

def create_normed_donkey_vimu(img_input_shape, tel_input_shape, dual_outputs, x_train, batch_size, *args):
    
    # power of 2 by convention
    tel_nodes = 2**(tel_input_shape[0] - 1).bit_length()

    drop = 0.2
    # telem_drop = 0.1
    
    img_in = Input(shape=img_input_shape, name='img_in') 
    x = img_in
    x = Rescaling(1./255, name='img_rescale')(x)
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)
    
    tel_norm = tensorflow.keras.layers.experimental.preprocessing.Normalization()
    tel_norm.adapt(x_train, batch_size)

    tel_in = Input(tel_input_shape, name='tel_in')
    y = tel_in
    y = tel_norm(tel_in)
    y = Dense(tel_nodes, activation='relu', name='dense_3')(y)
    y = Dense(tel_nodes, activation='relu', name='dense_4')(y)
    y = Dense(tel_nodes, activation='relu', name='dense_5')(y)

    z = concatenate([x, y])
    z = Dense(50, activation='relu', name='dense_6')(z)
    z = Dropout(drop)(z)
    z = Dense(50, activation='relu', name='dense_7')(z)
    z = Dropout(drop)(z)
    
    if dual_outputs:
    # two outputs for angle and throttle
        outputs = [
            Dense(1, activation='linear', name='steering_outputs')(z),
            Dense(1, activation='linear', name='throttle_outputs')(z)
        ]
    else:
        # combined output
        outputs = Dense(2, activation='linear', name='combined_outputs')(z)

    model = Model(inputs=[img_in, tel_in], outputs=outputs)
    return model



## See above for source. WIP adapting it.
def create_donkey_lstm(img_input_shape, tel_input_shape, dual_outputs, seq_length=3):
    # add sequence length dimensions as keras time-distributed expects shape
    # of (num_samples, seq_length, input_shape)
    if len(img_input_shape) < 4:
        img_seq_shape = (seq_length,) + img_input_shape
    else:
        img_seq_shape = img_input_shape
    if len(tel_input_shape) < 3:
        tel_seq_shape = (seq_length,) + tel_input_shape
    else:
        tel_seq_shape = tel_input_shape
    # tel_seq_shape = tel_input_shape
        
    img_in = Input(shape=img_seq_shape, name='img_in')
    tel_in = Input(shape=tel_seq_shape, name='tel_in')
    
    # power of 2 by convention
    tel_nodes = 2**(tel_input_shape[1] - 1).bit_length()
    # tel_nodes = 2**(tel_input_shape[0] - 1).bit_length()
    
    drop_out = 0.3

    x = img_in
    x = TD(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TD(Flatten(name='flattened_img'))(x)
    x = TD(Dense(100, activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    # x = Reshape((-1, 100, 3))(x)
    
#     y = tel_in
#     y = TD(Dense(tel_nodes, activation='relu'))(y)
#     y = TD(Dense(tel_nodes, activation='relu'))(y)
#     y = TD(Dense(tel_nodes, activation='relu'))(y)
    
    # y = tel_in
    # y = Dense(tel_nodes, activation='relu', name='dense_3')(y)
    # # y = Dropout(drop)(y)
    # y = Dense(tel_nodes, activation='relu', name='dense_4')(y)
    # # y = Dropout(drop)(y)
    # y = Dense(tel_nodes, activation='relu', name='dense_5')(y)
    # # y = Dropout(drop)(y)
    
    z = x
    # z = concatenate([x, y])
    z = LSTM(128, return_sequences=True, name="LSTM_seq")(z)
    z = Dropout(.1)(z)
    z = LSTM(128, return_sequences=False, name="LSTM_fin")(z)
    z = Dropout(.1)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(10, activation='relu')(z)
    
    if dual_outputs:
    # two outputs for angle and throttle
        outputs = [
            Dense(1, activation='linear', name='steering_outputs')(z),
            Dense(1, activation='linear', name='throttle_outputs')(z)
        ]
    else:
        # combined output
        outputs = Dense(2, activation='linear', name='combined_outputs')(z)

    model = Model(inputs=[img_in, tel_in], outputs=outputs, name='lstm')
    return model


# https://github.com/autorope/donkeycar/blob/dev/donkeycar/parts/keras.py
def conv2d(filters, kernel, strides, layer_num, activation='relu'):
    """
    Helper function to create a standard valid-padded convolutional layer
    with square kernel and strides and unified naming convention
    :param filters:     channel dimension of the layer
    :param kernel:      creates (kernel, kernel) kernel matrix dimension
    :param strides:     creates (strides, strides) stride
    :param layer_num:   used in labelling the layer
    :param activation:  activation, defaults to relu
    :return:            tf.keras Convolution2D layer
    """
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))
