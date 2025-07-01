from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from utils import PATCH_SIZE

# plot 0 metrics
# OPTIMIZER = Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.001,
#                                                      decay_steps=2000 * 20,
#                                                      end_learning_rate=0.00005,
#                                                      power=2.0))
# LOSS_WEIGHTS = {'sigmoid': PATCH_SIZE * PATCH_SIZE}
# LOSS = {'sigmoid': SigmoidFocalCrossEntropy(from_logits=False,
#                                             alpha=0.5,
#                                             gamma=1.2,
#                                             reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) # average total loss over the batch size
#         }
# METRICS = {'sigmoid': [BinaryAccuracy(name='binary_accuracy', threshold=0.5),
#                              AUC(curve="ROC", name='ROC_AUC'),
#                              AUC(curve="PR", name='PR_AUC')
#                              ]
#                  }

# plot 1 metrics
OPTIMIZER = Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.001,
                                                     decay_steps=2000 * 20,
                                                     end_learning_rate=0.00005,
                                                     power=2.0))
LOSS_WEIGHTS = {'sigmoid': PATCH_SIZE * PATCH_SIZE}
LOSS = {'sigmoid': SigmoidFocalCrossEntropy(from_logits=False,
                                            alpha=0.98,
                                            gamma=2.0,
                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) # average total loss over the batch size
        }
METRICS = {'sigmoid': [BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                             AUC(curve="ROC", name='ROC_AUC'),
                             AUC(curve="PR", name='PR_AUC')
                             ]
                 }