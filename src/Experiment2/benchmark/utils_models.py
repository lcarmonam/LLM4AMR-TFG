from tensorflow.keras import backend as K
import numpy as np
import random, os, json
import tensorflow as tf

### RESET KERAS ###
def reset_keras(seed=42):
    """Function to ensure that results from Keras models
    are consistent and reproducible across different runs"""
    
    K.clear_session()
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
   