"""
Data Poisoning Attacks Module

Implementaciones de ataques de envenenamiento de datos para sistemas 
de aprendizaje federado.

Available Attacks:
    - LabelFlippingAttack: Invierte etiquetas para degradar el modelo
"""

from .label_flipping import LabelFlippingAttack

__version__ = '0.1.0'
__author__ = 'FLAegis Team'

__all__ = [
    'LabelFlippingAttack'
    ]