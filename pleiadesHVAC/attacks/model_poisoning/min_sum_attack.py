from ..base_attack import BaseAttack
import numpy as np
from typing import List, Tuple, Union, Optional

class MinSumAttack(BaseAttack):
    """
    Implementa el ataque Min-Sum sobre los parámetros del modelo.
    
    Este ataque busca minimizar la suma de los parámetros del modelo global
    para degradar su rendimiento mientras evita la detección.
    """
    
    def __init__(self, scale_factor: float = 1.0, clip_threshold: float = 100.0):
        """
        Inicializa el ataque Min-Sum.
        
        Args:
            scale_factor (float): Factor de escala para las actualizaciones maliciosas
            clip_threshold (float): Umbral para recortar valores extremos
        """
        self.scale_factor = scale_factor
        self.clip_threshold = clip_threshold
    
    def attack(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Ejecuta el ataque Min-Sum sobre los parámetros del modelo.
        
        Args:
            data: Tupla (actualizaciones_benignas, modelo_referencia)
                - actualizaciones_benignas: Actualizaciones de clientes benignos
                - modelo_referencia: Parámetros del modelo de referencia
        
        Returns:
            np.ndarray: Actualización maliciosa generada
        """
        benign_updates, reference_model = data
        
        # Calcular la dirección del ataque (opuesta a la media de actualizaciones benignas)
        attack_direction = -np.mean(benign_updates, axis=0)
        
        # Normalizar la dirección del ataque
        attack_norm = np.linalg.norm(attack_direction)
        if attack_norm > 0:
            attack_direction /= attack_norm
            
        # Escalar el ataque
        malicious_update = attack_direction * self.scale_factor
        
        # Recortar valores extremos
        malicious_update = np.clip(
            malicious_update, 
            -self.clip_threshold, 
            self.clip_threshold
        )
        
        # Asegurar que la actualización maliciosa tiene la misma forma que el modelo
        assert malicious_update.shape == reference_model.shape, \
            "La actualización maliciosa debe tener la misma forma que el modelo"
            
        return malicious_update
    
    def _check_update_magnitude(self, update: np.ndarray) -> bool:
        """
        Verifica si la magnitud de la actualización está dentro de límites aceptables.
        
        Args:
            update: Actualización a verificar
            
        Returns:
            bool: True si la actualización es válida, False en caso contrario
        """
        update_norm = np.linalg.norm(update)
        return update_norm <= self.clip_threshold