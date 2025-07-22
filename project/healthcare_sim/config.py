NUM_PATIENTS = 5
NUM_PATHWAYS = 5
NUM_ACTIONS = 4
NUM_STEPS = 15
BASE_CAPACITY = 10

import numpy as np
import random

# --- Ideal clinical values ---
IDEAL_CLINICAL_VALUES = {
    'bp': 120,
    'glucose': 90,
    'bmi': 22,
    'oxygen': 98,
    'mental_health': 80,
}

INPUT_ACTIONS = 'a0'  # Two standard input actions- random actions assigned at the start of a pathway
OUTPUT_ACTIONS = 'a2'       # Standard output action 

np.random.seed(42)
random.seed(42)
