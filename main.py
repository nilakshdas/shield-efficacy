import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from absl import app, flags, logging
from cleverhans.attacks import ProjectedGradientDescent
import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from constants import *
from data import load_tfrecords_dataset
from jobby import JobbyJob
from metrics import AccuracyMeter
from model import (AttackSHIELDModel, 
                   EvalSHIELDModel, 
                   load_jpeg_trained_ensemble)


FLAGS = flags.FLAGS

flags.DEFINE_list(
    'attack_models', None,
    'Models to be attacked using PGD')
flags.DEFINE_list(
    'eval_models', None,
    'Models to be used in SHIELD ensemble')
flags.DEFINE_integer(
    'seed', 1234,
    'Seed for the PRNG for this experiment')
flags.DEFINE_integer(
    'batch_size', 16,
    'Batch size for evaluating the results')
flags.DEFINE_integer(
    'epsilon', 16,
    'Epsilon parameter to be used with PGD attack')
flags.DEFINE_float(
    'eps_iter', 1.0,
    'eps_iter parameter to be used with PGD attack')
flags.DEFINE_integer(
    'nb_iter', 20,
    'nb_iter parameter to be used with PGD attack')
flags.DEFINE_boolean(
    'attack_differentiable_slq', True,
    'Attack differentiable SLQ instead of differentiable JPEG')

flags.mark_flag_as_required('attack_models')
flags.mark_flag_as_required('eval_models')


def main(argv):
    del argv # unused

    args_keys = {
        'attack_models', 'eval_models', 'seed', 'batch_size',
        'epsilon', 'eps_iter', 'nb_iter',
        'attack_differentiable_slq'
    }

    args_dict = {
        k: v for k, v in FLAGS.flag_values_dict().items() 
        if k in args_keys}

    with JobbyJob('127.0.0.1:27017', args_dict) as job:
        logging.info('Attacking models: %s' % FLAGS.attack_models)
        logging.info('Evaluating with models: %s' % FLAGS.eval_models)
        logging.info('Epsilon for PGD attack = %d' % FLAGS.epsilon)

        tf.set_random_seed(FLAGS.seed)

        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=1.,
                    allow_growth=True)))
        
        keras.backend.set_session(sess)
        keras.backend.set_learning_phase(0)

        with tf.name_scope('TFRecordsLoader'):
            dataset = load_tfrecords_dataset(TFRECORDS_FILENAMES)
            dataset = dataset.batch(FLAGS.batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            
            X, y_true = next_element

        with sess.as_default():
            attack_model_paths = [MODEL_NAME_TO_CKPT_PATH_MAP[m] 
                                for m in FLAGS.attack_models]
            eval_model_paths = [MODEL_NAME_TO_CKPT_PATH_MAP[m] 
                                for m in FLAGS.eval_models]

            attack_model = AttackSHIELDModel(
                load_jpeg_trained_ensemble(
                    FLAGS.attack_models, attack_model_paths),
                attack_differentiable_slq=FLAGS.attack_differentiable_slq)
            
            eval_model = EvalSHIELDModel(
                load_jpeg_trained_ensemble(
                    FLAGS.eval_models, eval_model_paths))

            y_target = attack_model.get_least_likely_prediction(X)
            y_target_one_hot = tf.one_hot(y_target, 1000, axis=-1)
            
            attack = ProjectedGradientDescent(attack_model, sess=sess)
            attack_kwargs = {
                'y_target': y_target_one_hot,
                'eps': FLAGS.epsilon, 
                'eps_iter': FLAGS.eps_iter, 
                'nb_iter': FLAGS.nb_iter}

            X_adv = attack.generate(X, **attack_kwargs)
            y_pred_shield = eval_model.get_predicted_class(X_adv)
        
            writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
            writer.close()
            
            model_accuracy = AccuracyMeter()
            attack_success = AccuracyMeter()
            with tqdm(total=1171, unit='imgs') as pbar:
                while True:
                    try:
                        y_true_np, y_target_np, y_pred_shield_np = \
                            sess.run([y_true, y_target, y_pred_shield])
                        
                        model_accuracy.offer(y_pred_shield_np, y_true_np)
                        attack_success.offer(y_pred_shield_np, y_target_np)
                        
                        pbar.set_postfix(
                            model_accuracy=model_accuracy.evaluate(),
                            attack_success=attack_success.evaluate())
                        pbar.update(y_true_np.shape[0])

                    except tf.errors.OutOfRangeError:
                        break

        job.update_output(model_accuracy=model_accuracy.evaluate(),
                          attack_success=attack_success.evaluate())

        logging.info('model_accuracy = %.04f' % model_accuracy.evaluate())
        logging.info('attack_success = %.04f' % attack_success.evaluate())


if __name__ == "__main__":
    app.run(main)
