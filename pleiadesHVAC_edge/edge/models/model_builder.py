from abc import ABC, abstractmethod
import tensorflow as tf

class ModelBuilder(ABC):
    """
    Patron Factoria (Abierto/Cerrado de SOLID)
    Clase base abstracta para construcción de modelos.
    """
    
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Construye y retorna un modelo."""
        pass


        