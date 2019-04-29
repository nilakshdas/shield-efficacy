import glob
import os

from cleverhans.model import Model as CleverhansModel
from cleverhans.utils_keras import KerasModelWrapper
import keras
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import \
    resnet_arg_scope, resnet_v2_50

from processing import (differentiable_jpeg, 
                        differentiable_slq, slq,
                        resnet50_keras_preprocessing_fn)

slim = tf.contrib.slim


def load_jpeg_trained_ensemble(model_names, model_paths, sess=None):
    assert len(model_names) == len(model_paths)
    
    with tf.name_scope(
            'JPEGTrainedEnsemble/%s' 
            % '_'.join(map(str, model_names))):
        
        ensemble = list()
        for model_name, model_path in zip(model_names, model_paths):
            model = None
            if 'old' in model_name:
                model = ResNet50v2TFSlimModel(model_path, sess=sess)
            elif 'new' in model_name:
                model = ResNet50v2KerasModel(model_path, sess=sess)
            
            assert model is not None
            ensemble.append(model)

    return ensemble


class CleverhansEvalModel(object):
    def get_predicted_class(self, x):
        probs = self.get_probs(x)
        preds = tf.argmax(probs, axis=1)
        return preds

    def get_least_likely_prediction(self, x):
        probs = self.get_probs(x)
        preds = tf.argmin(probs, axis=1)
        return preds


class ResNet50v2TFSlimModel(CleverhansModel, CleverhansEvalModel):
    
    _scopes_loaded = set()

    def __init__(self, ckpt_dir_path, sess=None):
        super(ResNet50v2TFSlimModel, self).__init__()

        if sess is None:
            sess = tf.get_default_session()
        assert sess is not None

        ckpt_dir_name = os.path.basename(ckpt_dir_path)
        
        self._sess = sess
        self._ckpt_dir_path = ckpt_dir_path
        self._var_scope = '-'.join(['slim', ckpt_dir_name])

    def _get_latest_checkpoint_path(self):
        checkpoint_paths = glob.glob(
            os.path.join(self._ckpt_dir_path, '*.data-*'))
        latest_checkpoint_path = max(checkpoint_paths, 
                                     key=lambda p: os.path.basename(p))
        latest_checkpoint_path = latest_checkpoint_path.split('.data')[0]
        return latest_checkpoint_path

    def _get_updated_endpoints(self, original_end_points):
        logits_var_name = self._var_scope + '/resnet_v2_50/logits'

        end_points = dict(original_end_points)

        original_logits = tf.squeeze(end_points[logits_var_name], [1, 2])
        new_logits = tf.slice(original_logits, [0, 1], [-1, -1])

        end_points['logits'] = new_logits
        end_points['probs'] = tf.nn.softmax(end_points['logits'])

        return end_points
    
    def _preprocessing_fn(self, x):
        # From https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py#L244
        x = tf.divide(x, 255.0)
        x = tf.subtract(x, 0.5)
        x = tf.multiply(x, 2.0)
        return x
    
    def fprop(self, x):
        num_original_classes = 1001

        var_to_ckpt_name = lambda v: \
            v.name.replace(self._var_scope+'/', '')\
                  .replace(':0', '')
        
        with slim.arg_scope(resnet_arg_scope()), \
                tf.variable_scope(self._var_scope):
            
            x = self._preprocessing_fn(x)
            
            net, end_points = resnet_v2_50(
                x, num_classes=num_original_classes,
                is_training=False, reuse=tf.AUTO_REUSE)
            end_points = self._get_updated_endpoints(end_points)
        
        # Load weights for a particular scope only once
        if self._var_scope not in self._scopes_loaded:
            variables_to_restore = list(filter(
                lambda v: v.name.split('/')[0] == self._var_scope,
                slim.get_variables_to_restore(exclude=[])))
            
            variable_name_map = {
                var_to_ckpt_name(v): v 
                for v in variables_to_restore}
            
            saver = tf.train.Saver(var_list=variable_name_map)
            saver.restore(self._sess, self._get_latest_checkpoint_path())

            self._scopes_loaded.add(self._var_scope)

        return end_points


class ResNet50v2KerasModel(KerasModelWrapper, CleverhansEvalModel):
    
    def __init__(self, ckpt_path, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        assert sess is not None

        ckpt_file_name = os.path.basename(ckpt_path)
        
        self._var_scope = '-'.join(['keras', ckpt_file_name])

        with tf.name_scope(self._var_scope):
            kmodel = keras.applications.resnet50.ResNet50(weights=ckpt_path)
            super(ResNet50v2KerasModel, self).__init__(kmodel)

    def _preprocessing_fn(self, x):
        return resnet50_keras_preprocessing_fn(x)
    
    def fprop(self, x):
        with tf.name_scope(self._var_scope):
            x = self._preprocessing_fn(x)
            return super(ResNet50v2KerasModel, self).fprop(x)


class AttackSHIELDModel(CleverhansModel, CleverhansEvalModel):
    def __init__(
        self, ensemble,
        attack_jpeg_qualities=None,
        attack_differentiable_slq=True):
        
        super(AttackSHIELDModel, self).__init__()

        if not attack_differentiable_slq and attack_jpeg_qualities is None:
            attack_jpeg_qualities = [
                None, 90, 80, 70, 60, 
                50, 40, 30, 20, 10]

        self.ensemble = ensemble
        self.layer_names = ['logits', 'probs']
        self.attack_jpeg_qualities = attack_jpeg_qualities
        self.attack_differentiable_slq = attack_differentiable_slq

    def _preprocessing_fn(self, x):
        if self.attack_differentiable_slq:
            with tf.name_scope('DifferentiableSLQPreprocessing'):
                preprocessed_inputs = [tf.map_fn(differentiable_slq, x)]
        else:
            with tf.name_scope('DifferentiableJPEGPreprocessing'):
                preprocessed_inputs = [
                    differentiable_jpeg(x, q) if q is not None else x
                    for q in self.attack_jpeg_qualities]

        return preprocessed_inputs

    def fprop(self, x):
        preprocessed_inputs = self._preprocessing_fn(x)
        num_preprocessed_inputs = len(preprocessed_inputs)
        num_models = len(self.ensemble)
        outer_logits = list()
        
        for model in self.ensemble:
            inner_logits = [
                model.get_logits(x_) 
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
        with tf.name_scope('SLQPreprocessing'):
            x = tf.map_fn(slq, x)
        
        preds = tf.transpose(tf.stack(
            [model.get_predicted_class(x)
             for model in self.ensemble]))
        preds = tf.map_fn(self._get_majority_vote, preds)
        
        return preds
