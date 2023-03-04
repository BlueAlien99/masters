from enum import Enum
from typing import Literal


class Datasets(str, Enum):
    AS = 'answers-students'
    H = 'headlines'
    I = 'images'


DataType = Literal['test', 'train']
