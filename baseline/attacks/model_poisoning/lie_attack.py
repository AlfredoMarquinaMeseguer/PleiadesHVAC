from ..base_attack import BaseAttack
import numpy as np
from typing import Tuple, Optional

class LieAttack(BaseAttack):
    """
    Implementa el ataque de mentira (Lie Attack)(Little Is Enough) sobre los parámetros del modelo.
    
    Este ataque envía actualizaciones falsas pero plausibles al servidor,
    escalando las actualizaciones benignas y añadiendo ruido controlado.
    """
    
    def __init__(
        self, 
        scale_factor: float = 1.5,
        noise_range: float = 0.1,
        clip_threshold: float = 100.0
    ):
        """
        Inicializa el ataque de mentira.
        
        Args:
            scale_factor: Factor de escala para las actualizaciones benignas
            noise_range: Rango del ruido añadido [-noise_range, noise_range]
            clip_threshold: Umbral para recortar valores extremos
        """
        self.scale_factor = scale_factor
        self.noise_range = noise_range
        self.clip_threshold = clip_threshold
    
    def attack(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Ejecuta el ataque de mentira.
        
        Args:
            data: Tupla (actualizaciones_benignas, modelo_referencia)
        
        Returns:
            np.ndarray: Actualización maliciosa generada
        """
        benign_updates, reference_model = data
        
        # Calcular la media de actualizaciones benignas
        mean_update = np.mean(benign_updates, axis=0)
        
        # Escalar la actualización
        scaled_update = mean_update * self.scale_factor
        
        while True:
            # Generar ruido aleatorio
            noise = np.random.uniform(
                -self.noise_range,
                self.noise_range,
                size=scaled_update.shape
            )
            
            # Combinar actualización escalada con ruido
            malicious_update = scaled_update + noise
            
            # Recortar valores extremos
            malicious_update = np.clip(
                malicious_update,
                -self.clip_threshold,
                self.clip_threshold
            )
            
            # Validar forma
            assert malicious_update.shape == reference_model.shape, \
                "La actualización maliciosa debe tener la misma forma que el modelo"
            
            if self._validate_update(malicious_update):
                break
        
        return malicious_update
    
    def _validate_update(self, update: np.ndarray) -> bool:
        """
        Valida si la actualización generada es plausible.
        
        Args:
            update: Actualización a validar
            
        Returns:
            bool: True si la actualización es plausible
        """
        update_norm = np.linalg.norm(update)
        return update_norm <= self.clip_threshold