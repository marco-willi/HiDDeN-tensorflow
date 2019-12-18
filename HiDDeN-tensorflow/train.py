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
import losses
import steps
import datasets

FLAGS = flags.FLAGS

def main(argv):

    if FLAGS.tbdir is not None:
        summary_writers = utils.create_summary_writers(FLAGS.tbdir)

    # prepare dataset
    dataset = datasets.MNIST()
    input_shape = dataset.get_input_shape()

    # Create Nets and Optimizers
    encoder_decoder = nets.encoder_decoder(
        input_shape, FLAGS.msg_length, FLAGS.noise_type)
    discriminator = nets.discriminator(input_shape)

    optimizer_encoder_decoder = tf.keras.optimizers.Adam(1e-3)
    optimizer_discriminator = tf.keras.optimizers.Adam(1e-3)

    # global step
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
        ckpt, FLAGS.ckptdir, max_to_keep=3)

    if ckpt_manager.latest_checkpoint is not None:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Loading model from checkpoint: {}".format(
            ckpt_manager.latest_checkpoint))

    while epoch < FLAGS.epochs:

        dataset_train = dataset.create_train_dataset()

        for epoch_step, cover_images in enumerate(dataset_train):

            # messages = create_messages(batch_size, msg_length)
            # TODO: verify tf.function speed-up
            # suspicion that messages = create_me... is way slower

            messages = tf.random.uniform(
                [FLAGS.batch_size, FLAGS.msg_length],
                minval=0, maxval=2, dtype=tf.int32)
            messages = tf.cast(messages, dtype=tf.float32)

            time_start = time.time()
            losses_train_step, metrics_train_step = steps.train(
                cover_images=cover_images,
                messages=messages,
                encoder_decoder=encoder_decoder,
                discriminator=discriminator,
                losses=losses,
                training=True,
                optimizer_encoder_decoder=optimizer_encoder_decoder,
                optimizer_discriminator=optimizer_discriminator)
            ms_per_step = (time.time() - time_start) / 1000.0

            # Write step summaries
            with summary_writers['train'].as_default():
                for metric_name, metric_value in losses_train_step.items():
                    tf.summary.scalar(
                        'loss_{}'.format(metric_name),
                        metric_value, step=step)
                tf.summary.scalar(
                    'ms_per_step', ms_per_step, step=step)

            step.assign_add(1)

        ckpt_save_path = ckpt_manager.save()
        print("Saved model after epoch {} to {}".format(
            epoch.numpy(), ckpt_save_path))

        # Training Loss
        print("Epoch {} Stats".format(epoch.numpy()))
        print("Training Stats ===========================")
        for loss_name, loss_value in losses_train_step.items():
            print("{}: {:.4f}".format(loss_name, loss_value))
        
        # Evaluate
        dataset_val = dataset.create_val_dataset()

        eval_metrics_tracker = tf.metrics.MeanTensor()

        for cover_images in dataset_val:
            messages = tf.random.uniform(
                [FLAGS.batch_size, FLAGS.msg_length],
                minval=0, maxval=2, dtype=tf.int32)
            messages = tf.cast(messages, dtype=tf.float32)

            losses_val_step, metrics_val_step = steps.train(
                cover_images=cover_images,
                messages=messages,
                encoder_decoder=encoder_decoder,
                discriminator=discriminator,
                losses=losses,
                training=False)

            all_metrics_val_step = {**losses_val_step, **metrics_val_step}
            metrics_names = sorted(all_metrics_val_step.keys())
            metric_values = list()
            for metric in metrics_names:
                metric_values.append(all_metrics_val_step[metric])

            eval_metrics_tracker.update_state(metric_values)

        metrics_results = eval_metrics_tracker.result().numpy()

        print("Evaluation Stats ===========================")
        with summary_writers['eval'].as_default():
            for m_name, m_value in zip(metrics_names, metrics_results):
                tf.summary.scalar(m_name, m_value, step=step)
                print("{}: {:.4f}".format(m_name, m_value))

        eval_metrics_tracker.reset_states()

        epoch.assign_add(1)


if __name__ == '__main__':
    app.run(main)
