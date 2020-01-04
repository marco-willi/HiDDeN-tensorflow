"""
    HiDDeN - "HiDDeN: Hiding data with deep networks (Zhu et. al, 2018)"
"""
import time

import tensorflow as tf

from absl import app
from absl import logging
from cfg import flags

import utils
import nets
import steps
import datasets
import metrics
import losses

FLAGS = flags.FLAGS


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
        noise_layers=FLAGS.noise_layers,
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
    
    # Metrics Tracker
    metrics_train = metrics.MetricsTracker()
    metrics_val = metrics.MetricsTracker()

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
                
                step_losses = losses.step_loss(
                    cover_images,
                    messages,
                    encoder_decoder_output=outputs['encoder_decoder'],
                    discriminator_on_cover=outputs['discriminator_on_cover'],
                    discriminator_on_encoded=outputs['discriminator_on_encoded'])
                
                metrics_train.update(
                    step_losses,
                    messages,
                    encoder_decoder_output=outputs['encoder_decoder'],
                    discriminator_on_cover=outputs['discriminator_on_cover'],
                    discriminator_on_encoded=outputs['discriminator_on_encoded'])
                    
                metrics_train_results = metrics_train.results()
                metrics_train.reset()

                with summary_writers['train'].as_default():
                    for _name, _value in metrics_train_results.items():
                        tf.summary.scalar(
                            _name,
                            _value,
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
        for _name, _value in metrics_train_results.items():
            logging.info("{}: {:.4f}".format(_name, _value))
        
        # Evaluate
        dataset_val = dataset.create_val_dataset()

        for cover_images in dataset_val:

            messages = utils.create_messages(
                batch_size=cover_images.shape[0],
                msg_length=FLAGS.msg_length)

            # messages = tf.random.uniform(
            #     [FLAGS.batch_size, FLAGS.msg_length],
            #     minval=0, maxval=2, dtype=tf.int32)
            # messages = tf.cast(messages, dtype=tf.float32)

            outputs = steps.train(
                cover_images=cover_images,
                messages=messages,
                encoder_decoder=encoder_decoder,
                discriminator=discriminator,
                training=False)

            losses_val_step = losses.step_loss(
                cover_images,
                messages,
                encoder_decoder_output=outputs['encoder_decoder'],
                discriminator_on_cover=outputs['discriminator_on_cover'],
                discriminator_on_encoded=outputs['discriminator_on_encoded'])

            metrics_val.update(
                losses_val_step,
                messages,
                encoder_decoder_output=outputs['encoder_decoder'],
                discriminator_on_cover=outputs['discriminator_on_cover'],
                discriminator_on_encoded=outputs['discriminator_on_encoded'])
            
        metrics_val_results = metrics_val.results()
        metrics_val.reset()

        logging.info("Validation Stats ===========================")
        with summary_writers['val'].as_default():
            for _name, _value in metrics_val_results.items():
                tf.summary.scalar(_name, _value, step=step)
                logging.info("{}: {:.4f}".format(_name, _value))

        messages = utils.create_messages(
            batch_size=cover_images.shape[0],
            msg_length=FLAGS.msg_length)

        encoder_decoder_output = encoder_decoder(
            inputs={'cover_image': cover_images, 'message': messages},
            training=False)

        # write example images to Summaries
        with summary_writers['val'].as_default():

            transform_fn = None

            if FLAGS.to_yuv:
                transform_fn = tf.image.yuv_to_rgb
 
            utils.summary_images(
                cover=cover_images,
                encoded=encoder_decoder_output[
                    'encoded_image'],
                transmitted_encoded=encoder_decoder_output[
                    'transmitted_encoded_image'],
                transmitted_cover=encoder_decoder_output[
                    'transmitted_cover_image'],
                step=step,
                transform_fn=transform_fn)

        epoch.assign_add(1)


if __name__ == '__main__':
    app.run(main)
