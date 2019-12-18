""" Configuration """
from absl import flags

flags.DEFINE_string('tbdir', './tmp/tensorboard/',
                    'Directory to write Tensorboard summaries to.')

flags.DEFINE_string('ckptdir', './tmp/ckpts/',
                    'Directory to write checkpoints to.')

flags.DEFINE_string('logdir', './tmp/logs/',
                    'Directory to write logfiles to.')

flags.DEFINE_string('noise_type', 'identity',
                    'Noise layer to apply.')

flags.DEFINE_multi_enum(
    'noise_layers', 'identity',
    ['identity', 'gaussian', 'crop', 'cropout', 'jpeg_mask', 'dropout'],
    'A list of noise layers to apply.')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', None, "Numer of epochs to train the model for.")
flags.DEFINE_integer('msg_length', 8, 'Message length in bits.')


flags.DEFINE_float(
    'loss_weight_recover', 1.0,
    "Loss weight for message recovery loss.")
flags.DEFINE_float(
    'loss_weight_distortion', 0.7,
    "Loss weight for distortion between encoded and cover image.")
flags.DEFINE_float(
    'loss_weight_adversarial', 1e-3,
    "Loss weight for adversary.")
