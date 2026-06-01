"""
Model Poisoning Attacks Module

Este módulo proporciona implementaciones de diferentes ataques de envenenamiento
de modelo para sistemas de aprendizaje federado.

Available Attacks:
    - MinMaxAttack: Ataque que maximiza la distancia entre actualizaciones
    - MinSumAttack: Ataque que minimiza la suma de parámetros
    - LieAttack: Ataque basado en mentiras plausibles
    - StatOptAttack: Ataque basado en optimización estadística
"""

from .min_max_attack import MinMaxAttack
from .min_sum_attack import MinSumAttack
from .lie_attack import LieAttack
from .statopt_attack import StatOptAttack

__version__ = '0.1.0'
__author__ = 'FLAegis Team'

__all__ = [
    'MinMaxAttack',
    'MinSumAttack',
    'LieAttack',
    'StatOptAttack',
]