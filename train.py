"""
    HiDDeN - "HiDDeN: Hiding data with deep networks (Zhu et. al, 2018)"
"""
import os
import time

import tensorflow as tf

from absl import app
from absl import logging
from cfg import flags

import utils
import nets
import steps
import datasets

FLAGS = flags.FLAGS

exp_id = 'exp1_cd_crop'
FLAGS.noise_type = 'crop'


FLAGS.tbdir = os.path.join('./tmp/tensorboard/', exp_id)
FLAGS.plotdir = os.path.join('./tmp/plots/', exp_id)
FLAGS.ckptdir = os.path.join('./tmp/ckpts/', exp_id)
FLAGS.logdir = os.path.join('./tmp/logs/', exp_id)

FLAGS.dataset = 'dir'
FLAGS.train_dir = 'D:\Kaggle\cats_and_dogs\hidden\\train\\'
FLAGS.val_dir = 'D:\Kaggle\cats_and_dogs\hidden\\val\\'
FLAGS.test_dir = 'D:\Kaggle\cats_and_dogs\hidden\\test\\'
FLAGS.train_crop = [64, 64, 3]
FLAGS.batch_size = 12
FLAGS.epochs = 50
FLAGS.to_yuv = False
FLAGS.loss_weight_distortion = 10


#FLAGS.dataset = 'cifar10'
# FLAGS.batch_size = 12
# FLAGS.epochs = 2
# FLAGS.to_yuv = False


def main(argv):

    tf.random.set_seed(FLAGS.seed)

    if FLAGS.tbdir is not None:
        summary_writers = utils.create_summary_writers(
            utils.fix_path(FLAGS.tbdir))

    # prepare dataset
    dataset = datasets.get_dataset()()
    input_shape = dataset.get_input_shape()

    # Create Nets and Optimizers
    encoder_decoder = nets.encoder_decoder(
        input_shape=input_shape,
        msg_length=FLAGS.msg_length,
        noise_type=FLAGS.noise_type,
        n_convbnrelu_encoder=FLAGS.n_convbnrelu_encoder,
        n_convbnrelu_decoder=FLAGS.n_convbnrelu_decoder)

    discriminator = nets.discriminator(
        input_shape=input_shape,
        n_convbnrelu=FLAGS.n_convbnrelu_discriminator)

    optimizer_encoder_decoder = tf.keras.optimizers.Adam(1e-3)
    optimizer_discriminator = tf.keras.optimizers.Adam(1e-3)

    # global step / epoch variables
    step = tf.Variable(0, dtype=tf.int64)
    epoch = tf.Variable(0, dtype=tf.int64)

    # prepare checkpointer
    ckpt = tf.train.Checkpoint(
        step=step,
        epoch=epoch,
        optimizer_encoder_decoder=optimizer_encoder_decoder,
        optimizer_discriminator=optimizer_discriminator,
        encoder_decoder=encoder_decoder,
        discriminator=discriminator)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, utils.fix_path(FLAGS.ckptdir), max_to_keep=FLAGS.keep_ckpts)

    if ckpt_manager.latest_checkpoint is not None:
        if FLAGS.load_from_ckpt:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            logging.info("Loading model from checkpoint: {}".format(
                ckpt_manager.latest_checkpoint))

    while epoch < FLAGS.epochs:

        dataset_train = dataset.create_train_dataset()

        for epoch_step, cover_images in enumerate(dataset_train):

            messages = tf.random.uniform(
                [FLAGS.batch_size, FLAGS.msg_length],
                minval=0, maxval=2, dtype=tf.int32)
            messages = tf.cast(messages, dtype=tf.float32)

            time_start = time.time()
            outputs = steps.train(
                cover_images=cover_images,
                messages=messages,
                encoder_decoder=encoder_decoder,
                discriminator=discriminator,
                training=True,
                optimizer_encoder_decoder=optimizer_encoder_decoder,
                optimizer_discriminator=optimizer_discriminator)

            ms_per_step = (time.time() - time_start) * 1000.0
            ms_per_sample = ms_per_step / FLAGS.batch_size

            # Write step summaries
            is_summary_step = (step.numpy() % FLAGS.summary_freq) == 0
            if is_summary_step:
                
                step_losses = steps.calculate_step_losses(
                    cover_images,
                    messages,
                    encoder_decoder_output=outputs['encoder_decoder'],
                    discriminator_on_cover=outputs['discriminator_on_cover'],
                    discriminator_on_encoded=outputs['discriminator_on_encoded'])
                
                step_metrics = steps.calculate_step_metrics(
                    messages,
                    encoder_decoder_output=outputs['encoder_decoder'],
                    discriminator_on_cover=outputs['discriminator_on_cover'],
                    discriminator_on_encoded=outputs['discriminator_on_encoded'])
                
                step_metrics_losses = {**step_losses, **step_metrics}

                with summary_writers['train'].as_default():
                    for metric_name, metric_value in step_metrics_losses.items():
                        tf.summary.scalar(
                            metric_name,
                            metric_value,
                            step=step)

                    tf.summary.scalar(
                        'ms_per_step', ms_per_step, step=step)

                    tf.summary.scalar(
                        'ms_per_sample', ms_per_sample, step=step)

            step.assign_add(1)

        ckpt_save_path = ckpt_manager.save()
        logging.info("Saved model after epoch {} to {}".format(
            epoch.numpy(), ckpt_save_path))

        # Training Loss
        logging.info("Epoch {} Stats".format(epoch.numpy()))
        logging.info("Training Stats ===========================")
        for loss_name, loss_value in step_metrics_losses.items():
            logging.info("{}: {:.4f}".format(loss_name, loss_value))
        
        # Evaluate
        dataset_val = dataset.create_val_dataset()

        eval_metrics_tracker = tf.metrics.MeanTensor()

        for cover_images in dataset_val:
            messages = tf.random.uniform(
                [FLAGS.batch_size, FLAGS.msg_length],
                minval=0, maxval=2, dtype=tf.int32)
            messages = tf.cast(messages, dtype=tf.float32)

            outputs = steps.train(
                cover_images=cover_images,
                messages=messages,
                encoder_decoder=encoder_decoder,
                discriminator=discriminator,
                training=False)

            losses_val_step = steps.calculate_step_losses(
                cover_images,
                messages,
                encoder_decoder_output=outputs['encoder_decoder'],
                discriminator_on_cover=outputs['discriminator_on_cover'],
                discriminator_on_encoded=outputs['discriminator_on_encoded'])

            metrics_val_step = steps.calculate_step_metrics(
                messages,
                encoder_decoder_output=outputs['encoder_decoder'],
                discriminator_on_cover=outputs['discriminator_on_cover'],
                discriminator_on_encoded=outputs['discriminator_on_encoded'])

            step_metrics_losses = {**losses_val_step, **metrics_val_step}

            metrics_names = sorted(step_metrics_losses.keys())
            metric_values = list()
            for metric in metrics_names:
                metric_values.append(step_metrics_losses[metric])

            eval_metrics_tracker.update_state(metric_values)

        metrics_results = eval_metrics_tracker.result().numpy()

        messages = utils.create_messages(
            batch_size=cover_images.shape[0],
            msg_length=FLAGS.msg_length)

        encoder_decoder_output = encoder_decoder(
            inputs={'cover_image': cover_images, 'message': messages},
            training=False)

        # write example images to Summaries
        with summary_writers['val'].as_default():

            difference_images = tf.math.abs(tf.subtract(
                cover_images, encoder_decoder_output['encoded_image'])) * 10.0

            images_to_plot = [
                cover_images,
                encoder_decoder_output['encoded_image'],
                difference_images,
                encoder_decoder_output['transmitted_encoded_image'],
                encoder_decoder_output['transmitted_cover_image']]

            names_to_plot = [
                'cover_images',
                'encoded_images',
                'difference_images',
                'transmitted_encoded_images',
                'transmitted_cover_images']

            descriptions_to_plot = [
                'Cover Images',
                'Encoded Images',
                'Abslute Diff. Coded/Encoded Images (magnified)',
                'Transmitted Encoded Images',
                'Transmitted Cover Images']

            if FLAGS.to_yuv:
                images_to_plot = [
                    tf.image.yuv_to_rgb(x) for x in images_to_plot]

            for i, name in enumerate(names_to_plot):
                tf.summary.image(
                    name=name,
                    data=images_to_plot[i],
                    step=step,
                    max_outputs=6,
                    description=descriptions_to_plot[i]
                )

        logging.info("Validation Stats ===========================")
        with summary_writers['val'].as_default():
            for m_name, m_value in zip(metrics_names, metrics_results):
                tf.summary.scalar(m_name, m_value, step=step)
                logging.info("{}: {:.4f}".format(m_name, m_value))

        eval_metrics_tracker.reset_states()

        epoch.assign_add(1)


if __name__ == '__main__':
    app.run(main)
