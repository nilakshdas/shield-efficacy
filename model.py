import os

from cleverhans.model import Model as CleverhansModel
from cleverhans.utils_keras import KerasModelWrapper
import keras
import tensorflow as tf

from processing import (differentiable_jpeg, 
                        differentiable_slq, 
                        resnet50_preprocessing_fn)


def load_resnet50_keras_model(checkpoint_path):
    with tf.name_scope(
            'JPEGTrainedResNet50Model/%s' 
            % os.path.basename(checkpoint_path)):
        model = keras.applications.resnet50.ResNet50(
            weights=checkpoint_path)
    
    return model


def load_jpeg_trained_ensemble(path_format, qualities):
    with tf.name_scope(
            'JPEGTrainedEnsemble/%s' 
            % '_'.join(map(str, qualities))):
        ensemble = [
            KerasModelWrapper(
                load_resnet50_keras_model(
                    path_format.format(q=q))) 
            for q in qualities]

    return ensemble


class AttackSHIELDModel(CleverhansModel):
    def __init__(
        self, ensemble,
        attack_jpeg_qualities=(
            None, 90, 80, 70, 60, 
            50, 40, 30, 20, 10),
        attack_differentiable_slq=False):
        
        super(AttackSHIELDModel, self).__init__()

        self.ensemble = ensemble
        self.layer_names = ['logits', 'probs']
        self.attack_jpeg_qualities = list(attack_jpeg_qualities)
        self.attack_differentiable_slq = attack_differentiable_slq

    def _get_preprocessed_inputs(self, x):
        if self.attack_differentiable_slq:
            with tf.name_scope('DifferentiableSLQPreprocessing'):
                preprocessed_inputs = [resnet50_preprocessing_fn(
                    tf.map_fn(differentiable_slq, x))]
        else:
            with tf.name_scope('DifferentiableJPEGPreprocessing'):
                preprocessed_inputs = list(map(
                    resnet50_preprocessing_fn,
                    [differentiable_jpeg(x, q)
                    if q is not None else x
                    for q in self.attack_jpeg_qualities]))

        return preprocessed_inputs

    def fprop(self, x):
        preprocessed_inputs = self._get_preprocessed_inputs(x)
        num_preprocessed_inputs = len(preprocessed_inputs)
        num_models = len(self.ensemble)
        outer_logits = list()
        
        for model in self.ensemble:
            inner_logits = [model.get_logits(x_) 
                            for x_ in preprocessed_inputs]

            outer_logits.append(
                tf.math.divide(
                    tf.math.add_n(inner_logits), 
                    num_preprocessed_inputs))

        out = dict()
        
        out['logits'] = tf.math.divide(
            tf.math.add_n(outer_logits),
            num_preprocessed_inputs)
        
        out['probs'] = tf.nn.softmax(out['logits'])

        return out


class EvalSHIELDModel(object):
    def __init__(self, ensemble):
        self.ensemble = ensemble

    @staticmethod
    def _get_majority_vote(preds):
        y, _, count = tf.unique_with_counts(preds)
        return y[tf.argmax(count)]
    
    def get_predicted_class(self, x):
        with tf.name_scope('DifferentiableSLQPreprocessing'):
            x = resnet50_preprocessing_fn(
                tf.map_fn(differentiable_slq, x))
        
        preds = tf.transpose(tf.stack(
            [model.get_predicted_class(x)
             for model in self.ensemble]))
        preds = tf.map_fn(self._get_majority_vote, preds)
        
        return preds
