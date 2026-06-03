import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Flatten, Dense, Dropout, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from .model_builder import ModelBuilder

class GRUModelBuilder(ModelBuilder):
    def __init__(self, input_shape, gru_filters_1=64, gru_filters_2=32,
                 dense_units_1=64, dense_units_2=32, dropout_rate=0.1, 
                 learning_rate=0.0003):
        self.input_shape = input_shape
        self.gru_filters_1 = gru_filters_1
        self.gru_filters_2 = gru_filters_2
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self) -> tf.keras.Model:
        model = Sequential([
            # The dedicated Input layer handles the shape
            Input(shape=self.input_shape), 
            
            GRU(units=self.gru_filters_1, return_sequences=True), 
            LayerNormalization(),
            Dropout(self.dropout_rate),

            GRU(units=self.gru_filters_2),
            LayerNormalization(), # normalizes per-timestep, safe for recurrence
            Dropout(self.dropout_rate),
            
            Dense(self.dense_units_1, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(self.dense_units_2, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(units=1 , activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), 
            loss='mean_squared_error', 
            metrics=['mae'] 
        )
        return model


class GRUSimpleModelBuilder(ModelBuilder):
    def __init__(self, input_shape, gru_filters_1=64, gru_filters_2=32,
                 dense_units_1=64, dense_units_2=32, 
                 learning_rate=0.0003):
        self.input_shape = input_shape
        self.gru_filters_1 = gru_filters_1
        self.gru_filters_2 = gru_filters_2
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.learning_rate = learning_rate

    def build(self) -> tf.keras.Model:
        model = Sequential([
            Input(shape=self.input_shape), 
            
            GRU(units=self.gru_filters_1, return_sequences=True), 

            GRU(units=self.gru_filters_2),
            
            Dense(units=1 , activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), 
            loss='mean_squared_error', 
            metrics=['mae'] 
        )
        return model