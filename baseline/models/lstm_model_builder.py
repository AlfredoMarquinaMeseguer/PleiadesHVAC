import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, ConvLSTM2D, Flatten, Dense, Dropout, BatchNormalization, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from .model_builder import ModelBuilder

class LSTMModelBuilder(ModelBuilder):
    def __init__(self, input_shape, lstm_filters_1=64, lstm_filters_2=32,
                 dense_units_1=64, dense_units_2=32, dropout_rate=0.1, learning_rate=0.0003):
        self.input_shape = input_shape
        self.lstm_filters_1 = lstm_filters_1
        self.lstm_filters_2 = lstm_filters_2
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self) -> tf.keras.Model:
        model = Sequential([            
            Input(shape=self.input_shape),          # 3D: no spatial dims

            LSTM(self.lstm_filters_1, return_sequences=True),             # learns temporal patterns, passes full sequence
            LayerNormalization(),                        # normalizes per-timestep, safe for recurrence
            Dropout(self.dropout_rate),

            LSTM(self.lstm_filters_2, return_sequences=False),            # compresses to final hidden state only
            LayerNormalization(),
            Dropout(self.dropout_rate),
                                                        # NO Flatten needed
            Dense(self.dense_units_1, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.dense_units_2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model

class ConvLSTMModelBuilder(ModelBuilder):
    def __init__(self, input_shape, convlstm_filters_1=64, convlstm_filters_2=32,
                 dense_units_1=64, dense_units_2=32, dropout_rate=0.1, learning_rate=0.0003):
        self.input_shape = input_shape
        self.convlstm_filters_1 = convlstm_filters_1
        self.convlstm_filters_2 = convlstm_filters_2
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self) -> tf.keras.Model:
        model = Sequential([
            # ← USAR Input() COMO PRIMERA CAPA
            Input(shape=self.input_shape),
            ConvLSTM2D(self.convlstm_filters_1, (1, 3), activation='relu', 
                    padding='same', return_sequences=True),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            ConvLSTM2D(self.convlstm_filters_2, (1, 3), activation='relu', 
                    padding='same', return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Flatten(),
            Dense(self.dense_units_1, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.dense_units_2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model