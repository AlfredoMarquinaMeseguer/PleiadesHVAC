from typing import Dict, Any
from .base_attack import BaseAttack
from .model_poisoning import (
    MinMaxAttack,
    MinSumAttack,
    LieAttack,
    StatOptAttack
)
from .data_poisoning import (
    LabelFlippingAttack
    )

class AttackFactory:
    """Factory para crear instancias de ataques según la configuración."""
    
    @staticmethod
    def create_attack(config: Dict[str, Any]) -> BaseAttack:
        """
        Crea una instancia del ataque especificado.
        
        Args:
            config: Configuración del ataque del archivo YAML
            
        Returns:
            BaseAttack: Instancia del ataque configurado
            
        Raises:
            ValueError: Si el tipo de ataque o estrategia no es válido
        """
        attack_type = config["type"]
        attack_strategy = config["strategy"]
        
        # Get correct parameter group based on attack type
        if attack_type == "model_poisoning":
            params = config.get("model_poisoning_params", {}).get(attack_strategy, {})
        else:
            params = config.get("data_poisoning_params", {}).get(attack_strategy, {})
        
        if attack_type == "model_poisoning":
            if attack_strategy == "min_max":
                return MinMaxAttack(**params)
            elif attack_strategy == "min_sum":
                return MinSumAttack(**params)
            elif attack_strategy == "lie":
                return LieAttack(**params)
            elif attack_strategy == "statopt":
                return StatOptAttack(**params)
            
        elif attack_type == "data_poisoning":
            if attack_strategy == "label_flipping":
                return LabelFlippingAttack(**params)
            
        valid_model_attacks = ["min_max", "min_sum", "lie", "statopt"]
        valid_data_attacks = ["label_flipping"]
        
        raise ValueError(
            f"Invalid attack configuration. Type must be 'model_poisoning' or 'data_poisoning'. "
            f"Strategy must be one of {valid_model_attacks} for model poisoning "
            f"or one of {valid_data_attacks} for data poisoning."
        )