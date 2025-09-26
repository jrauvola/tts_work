from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SynthesisResult:
    audio_44k: np.ndarray
    sample_rate: int
    chars: List[str]
    alignments_ms: List[Tuple[float, float]]


