from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List, Optional

class BaseAttack(ABC):
    """
    Interfaz base para estrategias de ataque.
    
    Define la interfaz común que deben implementar todos los ataques,
    ya sean de envenenamiento de datos o de modelo.
    """
    
    @abstractmethod
    def attack(self, data: Any) -> Any:
        """
        Ejecuta el ataque sobre los datos o modelo proporcionado.
        
        Args:
            data: Datos o modelo a atacar
            
        Returns:
            Datos o modelo modificado tras el ataque
        """
        pass