import tensorflow as tf
from tensorflow.keras import layers
from .model_builder import ModelBuilder

class TransformerModelBuilder(ModelBuilder):
    """
    Constructor para un modelo basado en Transformer para regresión de series temporales.
    """
    #def __init__(self, input_shape, head_size=128, num_heads=4, ff_dim=128, num_transformer_blocks=4, dense_units=64, dropout_rate=0.1, learning_rate=1e-4):
    def __init__(self, input_shape, head_size=64, num_heads=4, ff_dim=64, num_transformer_blocks=2, dense_units=32, dropout_rate=0.2, learning_rate=1e-4):
        self.input_shape = input_shape
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def _transformer_encoder(self, inputs):
        """Crea un bloque codificador del Transformer."""
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout_rate
        )(x, x)
        x = layers.Dropout(self.dropout_rate)(x)
        res = x + inputs

        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build(self) -> tf.keras.Model:
        """Construye y compila el modelo Transformer."""
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        
        # Crear varios bloques de Transformer apilados
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)

        # Pooling para reducir la dimensionalidad de la secuencia
        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        
        # Capas densas finales para la regresión
        x = layers.Dense(self.dense_units, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model