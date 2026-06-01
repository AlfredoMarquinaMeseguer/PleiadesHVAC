from ..base_attack import BaseAttack
import numpy as np

class MinMaxAttack(BaseAttack):
    """
    Implementa el ataque Min-Max sobre los parámetros del modelo.
    
    Este ataque busca maximizar la distancia entre los parámetros benignos
    y maliciosos mientras mantiene las actualizaciones maliciosas dentro
    de un rango aceptable para evitar la detección.
    """
    
    def __init__(self, dev_type: str = 'unit_vec', lambda_param: float = 50):
        self.dev_type = dev_type
        self.lambda_param = lambda_param
        
    def attack(self, data: tuple) -> np.ndarray:
        """
        Ejecuta el ataque Min-Max.
        
        Args:
            data (tuple): Tupla conteniendo (all_updates, model_re)
            
        Returns:
            np.ndarray: Actualización maliciosa generada
        """
        all_updates, model_re = data
        
        if self.dev_type == 'unit_vec':
            if np.linalg.norm(model_re) == 0:
                deviation = np.random.normal(0, 1, model_re.shape)
                deviation /= np.linalg.norm(deviation)
            else:
                deviation = model_re / np.linalg.norm(model_re)
        elif self.dev_type == 'sign':
            deviation = np.sign(model_re)
        elif self.dev_type == 'std':
            deviation = np.std(all_updates, 0)
            
        # Rest of min-max attack implementation...
        return mal_update