from ..base_attack import BaseAttack
import numpy as np

class LabelFlippingAttack(BaseAttack):
    """
    Implementa el ataque de inversión de etiquetas.
    
    Este ataque modifica las etiquetas del conjunto de datos de entrenamiento
    para degradar el rendimiento del modelo global.
    """
    
    def __init__(self, flip_ratio: float = 1.0):
        self.flip_ratio = flip_ratio
        
    def attack(self, data: tuple) -> tuple:
        """
        Ejecuta el ataque de inversión de etiquetas.
        
        Args:
            data (tuple): Tupla (x_train, y_train) con datos de entrenamiento
            
        Returns:
            tuple: Datos modificados (x_train, y_train_poisoned)
        """
        x_train, y_train = data
        num_classes = len(np.unique(y_train))
        
        # Randomly select samples to flip
        num_samples = len(y_train)
        num_flip = int(num_samples * self.flip_ratio)
        flip_indices = np.random.choice(num_samples, num_flip, replace=False)
        
        # Flip labels
        y_train_poisoned = y_train.copy()
        y_train_poisoned[flip_indices] = np.random.randint(0, num_classes, num_flip)
        
        return x_train, y_train_poisoned