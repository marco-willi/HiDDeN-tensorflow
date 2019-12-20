""" Configuration """
from absl import flags

# Save / Log Directories
flags.DEFINE_string('tbdir', './tmp/tensorboard/',
                    'Directory to write Tensorboard summaries to.')

flags.DEFINE_string('plotdir', './tmp/plots/',
                    'Directory to write plots to.')

flags.DEFINE_string('ckptdir', './tmp/ckpts/',
                    'Directory to write checkpoints to.')

flags.DEFINE_string('logdir', './tmp/logs/',
                    'Directory to write logfiles to.')

# Dataset
flags.DEFINE_enum(
    'dataset', None, ['dir', 'mnist', 'cifar10'],
    "Define training dataset source.")
    
flags.DEFINE_string('train_dir', None, "Directory with training images.")
flags.DEFINE_string('val_dir', None, "Directory with validation images.")
flags.DEFINE_string('test_dir', None, "Directory with test images.")

flags.DEFINE_multi_integer("train_crop", [128, 128, 3],
 'Crop to take from images for model training (H, W, C)')

# Noise Layers
flags.DEFINE_bool('to_yuv', False,
 'Wheter to convert input images to YUV -- strongly \
  recommended for RGB images and JPEG noise')

flags.DEFINE_string('noise_type', 'identity',
                    'Noise layer to apply.')

flags.DEFINE_multi_enum(
    'noise_layers', 'identity',
    ['identity', 'gaussian', 'crop', 'cropout', 'jpeg_mask', 'dropout'],
    'A list of noise layers to apply.')

# Logging Options
flags.DEFINE_integer('summary_freq', 100, 'Write summaries every Nth step.')
flags.DEFINE_integer('keep_ckpts', 3, 'Max Number of model checkpoints to keep.')

# Training options
flags.DEFINE_integer('seed', 123, 'Random Global Seed')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', None, "Numer of epochs to train the model for.")
flags.DEFINE_integer('msg_length', 8, 'Message length in bits.')
flags.DEFINE_bool(
    'load_from_ckpt', True,
    'Whether to load checkpoint before model training if one is available.')

# Loss Weights
flags.DEFINE_float(
    'loss_weight_recovery', 1.0,
    "Loss weight for message recovery loss.")
flags.DEFINE_float(
    'loss_weight_distortion', 0.7,
    "Loss weight for distortion between encoded and cover image.")
flags.DEFINE_float(
    'loss_weight_adversarial', 1e-3,
    "Loss weight for adversary.")


# Architecture
flags.DEFINE_integer(
    'n_convbnrelu_encoder', 4,
    'Number of ConvBNRelu blocks of the encoder.')

flags.DEFINE_integer(
    'n_convbnrelu_decoder', 7,
    'Number of ConvBNRelu blocks of the decoder.')

flags.DEFINE_integer(
    'n_convbnrelu_discriminator', 3,
    'Number of ConvBNRelu blocks of the discriminator.')

flags.DEFINE_string(
    'cbr_initializer', 'he_normal', "Initializer of ConvBnRelu blocks")
