from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import configparser
from tensorflow.python.lib.io import file_io

from constants.constants import *
from trainer import utils

from tensorflow.contrib.rnn import LSTMStateTuple

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

## hyper parameters ##
tf.app.flags.DEFINE_string("kernel", "(1,3,3)", "kernel of convolutions")
tf.app.flags.DEFINE_string("classkernel", "(3,3)", "kernelsize of final classification convolution")
tf.app.flags.DEFINE_string("cnn_activation", 'leaky_relu',
                           "activation function for convolutional layers ('relu' or 'leaky_relu')")

tf.app.flags.DEFINE_boolean("bidirectional", True, "Bidirectional Convolutional RNN")
tf.app.flags.DEFINE_integer("convrnn_compression_filters", -1,
                            "number of convrnn compression filters or (default) -1 for no compression")
tf.app.flags.DEFINE_string("convcell", "gru", "Convolutional RNN cell architecture ('gru' (default) or 'lstm')")
tf.app.flags.DEFINE_string("convrnn_kernel", "(3,3)", "kernelsize of recurrent convolution. default (3,3)")
tf.app.flags.DEFINE_integer("convrnn_filters", None, "number of convolutional filters in ConvLSTM/ConvGRU layer")
tf.app.flags.DEFINE_float("recurrent_dropout_i", 1.,
                          "input keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_float("recurrent_dropout_c", 1.,
                          "state keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_integer("convrnn_layers", 1, "number of convolutional recurrent layers")
tf.app.flags.DEFINE_boolean("peephole", False,
                            "use peephole connections at convrnn layer. only for lstm (default False)")
tf.app.flags.DEFINE_boolean("convrnn_normalize", True, "normalize with batchnorm at convrnn layer (default True)")
tf.app.flags.DEFINE_string("aggr_strat", "state",
                           "aggregation strategie to reduce temporal dimension (either default 'state' or 'sum_output' or 'avg_output')")

tf.app.flags.DEFINE_float("learning_rate", None, "Adam learning rate")
tf.app.flags.DEFINE_float("beta1", 0.9, "Adam beta1")
tf.app.flags.DEFINE_float("beta2", 0.999, "Adam beta2")
tf.app.flags.DEFINE_float("epsilon", 1e-08, "Adam epsilon")

## expected data format ##
tf.app.flags.DEFINE_string("expected_datatypes",
                           "(tf.float32, tf.float32, tf.float32, tf.float32, tf.int64)", "expected datatypes")
tf.app.flags.DEFINE_integer("pix250m", None, "number of 250m pixels")
tf.app.flags.DEFINE_integer("num_bands_250m", None, "number of bands in 250 meter resolution (4)")
tf.app.flags.DEFINE_integer("num_bands_500m", None, "number of bands in 500 meter resolution (6)")
tf.app.flags.DEFINE_integer("num_classes", 8,
                            "number of classes not counting unknown class -> e.g. 0:uk,1:a,2:b,3:c,4:d -> num_classes 4")

## performance ##
tf.app.flags.DEFINE_boolean("swap_memory", True, "Swap memory between GPU and CPU for recurrent layers")

# #datadir
tf.app.flags.DEFINE_string('datadir', None, 'datadir')
tf.app.flags.DEFINE_string('modeldir', None, 'modeldir')
tf.app.flags.DEFINE_string('batchsize', None, 'batchsize')
tf.app.flags.DEFINE_string('train_on', None, 'train_on')
tf.app.flags.DEFINE_string('epochs', None, 'epochs')
tf.app.flags.DEFINE_string('limit_batches', None, 'limit_batches')
tf.app.flags.DEFINE_string('reference', None, 'reference')
tf.app.flags.DEFINE_string('storedir', '', 'storedir')
tf.app.flags.DEFINE_string('dataset', '', 'dataset')
tf.app.flags.DEFINE_string('writetiles', '', 'writetiles')
tf.app.flags.DEFINE_string('writeconfidences', '', 'writeconfidences')
tf.app.flags.DEFINE_string('step', '', 'step')
tf.app.flags.DEFINE_string('experiment', None, 'experiment')
tf.app.flags.DEFINE_string('optimizertype', None, 'experiment')

FLAGS = tf.app.flags.FLAGS

MODEL_CFG_FILENAME = 'params.ini'

ADVANCED_SUMMARY_COLLECTION_NAME="advanced_summaries"


def inference(input, is_train=True, num_classes=None):
    x, sequence_lengths = input

    rnn_output_list = list()
    rnn_state_list = list()

    x_rnn = x
    for j in range(1, FLAGS.convrnn_layers + 1):
        convrnn_kernel = eval(FLAGS.convrnn_kernel)
        x_rnn, state = convrnn_layer(input=x_rnn, is_train=is_train, filter=FLAGS.convrnn_filters,
                                     kernel=convrnn_kernel,
                                     bidirectional=FLAGS.bidirectional, convcell=FLAGS.convcell,
                                     sequence_lengths=sequence_lengths, scope="convrnn" + str(j))
        rnn_output_list.append(x_rnn)
        rnn_state_list.append(state)

    # # concat outputs from cnns and rnns in a dense scheme
    x = tf.concat(rnn_output_list, axis=-1)

    # # take concatenated states of last rnn block (might contain multiple conrnn layers)
    state = tf.concat(rnn_state_list, axis=-1)

    # use the cell state as featuremap for the classification step
    # cell state has dimensions (b,h,w,d) -> classification strategy
    if FLAGS.aggr_strat == 'state':
        class_input = state  # shape (b,h,w,d)
        classkernel = eval(FLAGS.classkernel)
        logits = conv_bn_relu(input=class_input, is_train=is_train, filter=num_classes,
                              kernel=classkernel, dilation_rate=(1, 1), conv_fun=tf.keras.layers.Conv2D,
                              var_scope="class")

    elif (FLAGS.aggr_strat == 'avg_output') or (FLAGS.aggr_strat == 'sum_output'):
        # last rnn output at each time t
        # class_input = x_rnn  # shape (b,t,h,w,d)
        class_input = x  # shape (b,t,h,w,d)

        # kernel = (1,FLAGS.classkernel[0],FLAGS.classkernel[1])
        kernel = (1, eval(FLAGS.classkernel)[0], eval(FLAGS.classkernel)[1])

        # logits for each single timeframe
        logits = conv_bn_relu(input=class_input, is_train=is_train, filter=num_classes, kernel=kernel,
                              dilation_rate=(1, 1, 1), conv_fun=tf.keras.layers.Conv3D, var_scope="class")

        if FLAGS.aggr_strat == 'avg_output':
            # average logit scores at each observation
            # (b,t,h,w,d) -> (b,h,w,d)
            logits = tf.reduce_mean(logits, axis=1)

        elif FLAGS.aggr_strat == 'sum_output':
            # summarize logit scores at each observation
            # the softmax normalization later will normalize logits again
            # (b,t,h,w,d) -> (b,h,w,d)
            logits = tf.reduce_sum(logits, axis=1)

    else:
        raise ValueError("please provide valid aggr_strat flag ('state' or 'avg_output' or 'sum_output')")

    tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, logits)

    return logits


def loss(logits, labels, mask, name):

    loss_per_px = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    _ = tf.identity(loss_per_px, name="loss_per_px")
    _ = tf.identity(mask, name="mask_per_px")

    lpp = tf.boolean_mask(loss_per_px, mask)

    return tf.reduce_mean(lpp, name=name)


def softmax_focal_loss(logits, labels, mask, gamma, alpha, name):
    #TODO focal max method1
    #source: https://github.com/fuyw/Grace/blob/76bc27230fef581ae90f12a267764ac176637ba9/utils.py
    loss_per_px = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    pred_y = tf.nn.softmax(logits, axis=-1)
    p_t = labels * pred_y + (1 - labels) * (1 - pred_y)
    p_t = tf.reduce_max(pred_y*labels, axis=-1)
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    focal_loss = modulating_factor * alpha * loss_per_px

    # _ = tf.identity(focal_loss, name="loss_per_px")
    # _ = tf.identity(mask, name="mask_per_px")
    #
    # flpp = tf.boolean_mask(focal_loss, mask)

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    focal_loss *= mask

    # return tf.reduce_mean(flpp, name=name)
    return tf.reduce_mean(focal_loss, name=name)


def _focal_loss(logits, labels, ignore_label, gamma, alpha, name):
    #TODO focal max method1
    # labels = tf.to_int64(labels)
    # labels = tf.convert_to_tensor(labels, tf.int64)
    # logits = tf.convert_to_tensor(logits, tf.float32)
    num_classes = logits.get_shape().as_list()[-1]
    one_hot_target = tf.one_hot(tf.cast(labels, tf.int32),
                            num_classes, on_value=1.0, off_value=0.0)
    one_hot_target = tf.squeeze(one_hot_target, axis=[3])
    model_out = logits #tf.add(logits, epsilon)
    ce = tf.multiply(one_hot_target, -tf.log(tf.clip_by_value(model_out, 1.e-5,1.0)))
    weight = tf.multiply(one_hot_target, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1, keepdims=True)
    not_ignore_mask = tf.to_float(
                tf.not_equal(labels, ignore_label))
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return tf.reduce_mean(reduced_fl * not_ignore_mask, name=name)


def optimize(loss, global_step, otype, name):
    lr = tf.compat.v1.placeholder_with_default(FLAGS.learning_rate, shape=(), name="learning_rate")
    beta1 = tf.compat.v1.placeholder_with_default(FLAGS.beta1, shape=(), name="beta1")
    beta2 = tf.compat.v1.placeholder_with_default(FLAGS.beta2, shape=(), name="beta2")

    if otype == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2,
            epsilon=FLAGS.epsilon
        )
    elif otype == 'nadam':
        optimizer = tf.contrib.opt.NadamOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2,
            epsilon=FLAGS.epsilon
        )
    elif otype == "adamW":
        ##TODO - need further work
        optimizer = tf.contrib.opt.AdamWOptimizer(
            learning_rate=lr, beta1=beta1, beta2=beta2,
            epsilon=FLAGS.epsilon, weight_decay=FLAGS.weight_decay
        )

    # method 1
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        return optimizer.minimize(loss, global_step=global_step, name=name)

    #method 2 - slower source: https://github.com/tensorflow/tensorflow/issues/25057
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # minimize_op = optimizer.minimize(loss=loss, global_step=global_step,name=name)
    # train_op = tf.group(minimize_op, update_ops)

    # return train_op


def eval_oa(logits, labels, mask):

    prediction_scores = tf.nn.softmax(logits=logits, name="prediction_scores")
    predictions = tf.argmax(prediction_scores, 3, name="predictions")

    targets = tf.argmax(labels, 3, name="targets")

    correctly_predicted = tf.equal(tf.boolean_mask(predictions, mask), tf.boolean_mask(targets, mask),
                                   name="correctly_predicted")

    overall_accuracy = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32), name="overall_accuracy")

    ###Single-GPU
    overall_accuracy_sum = tf.Variable(tf.zeros(shape=([]), dtype=tf.float32),
                                       trainable=False,
                                       name="overall_accuracy_result",
                                       collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES]
                                       )

    ###Multi-GPU
    # overall_accuracy_sum = tf.Variable(tf.zeros(shape=([]), dtype=tf.float32),
    #                                    trainable=False,
    #                                    name="overall_accuracy_result",
    #                                    aggregation=tf.VariableAggregation.SUM,
    #                                    collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES]
    #                                    )

    update_op =  tf.compat.v1.assign_add(overall_accuracy_sum, overall_accuracy)

    return(overall_accuracy, update_op)


def summary(metric_op, loss_op):
    """
    minimal summaries for training @ monitoring
    """

    tf.compat.v1.summary.scalar("accuracy", metric_op)
    tf.compat.v1.summary.scalar("loss", loss_op)

    return tf.compat.v1.summary.merge_all()


def input(features, labels):

    with tf.name_scope("raw"):
        x250, x500, doy, year = features

        x250 = tf.cast(x250, tf.float32, name="x250")
        x500 = tf.cast(x500, tf.float32, name="x500")
        doy = tf.cast(doy, tf.float32, name="doy")
        year = tf.cast(year, tf.float32, name="year")

    y = tf.cast(labels, tf.int32, name="y")

    # integer sequence lenths per batch for dynamic_rnn masking
    sequence_lengths = tf.reduce_sum(tf.cast(x250[:, :, 0, 0, 0] > 0, tf.int32), axis=1,
                                                name="sequence_lengths")

    def resize(tensor, new_height, new_width):
        b = tf.shape(tensor)[0]
        t = tf.shape(tensor)[1]
        h = tf.shape(tensor)[2]
        w = tf.shape(tensor)[3]
        d = tf.shape(tensor)[4]

        # stack batch on times to fit 4D requirement of resize_tensor
        stacked_tensor = tf.reshape(tensor, [b * t, h, w, d])
        reshaped_stacked_tensor = tf.image.resize(stacked_tensor, size=(new_height, new_width))
        return tf.reshape(reshaped_stacked_tensor, [b, t, new_height, new_width, d])

    def expand3x(vector):
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        return vector

    with tf.name_scope("reshaped"):
        b = tf.shape(x250)[0]
        t = tf.shape(x250)[1]
        px = tf.shape(x250)[2]


        x500 = tf.identity(resize(x500, px, px), name="x500")

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(x250, name="x250"))
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x500)

        # expand
        doymat = tf.multiply(expand3x(doy), tf.ones((b, t, px, px, 1)), name="doy")
        yearmat = tf.multiply(expand3x(year), tf.ones((b, t, px, px, 1)), name="year")

        x = tf.concat((x250, x500, doymat, yearmat), axis=-1, name="x")
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)

        if FLAGS.experiment == 'bands':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bandswodoy':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bands250m':
            bands250m = 2
            bands500m = 1
        elif FLAGS.experiment == 'bandswoblue':
            bands250m = 2
            bands500m = 4
        elif FLAGS.experiment == 'bandsaux':
            bands250m = 5
            bands500m = 5
        elif FLAGS.experiment == 'evi2':
            bands250m = 1
            bands500m = 1
        elif FLAGS.experiment == 'indices':
            bands250m = 7
            bands500m = 5
        else:
            bands250m = 10
            bands500m = 5

        FLAGS.num_bands_250m = bands250m
        FLAGS.num_bands_500m = bands500m

        # set depth of x for convolutions
        depth = FLAGS.num_bands_250m + FLAGS.num_bands_500m + 2  # doy and year
        # depth = bands250m + bands500m + 2  # doy and year

        # dynamic shapes. Fill for debugging
        x.set_shape([None, None, FLAGS.pix250m, FLAGS.pix250m, depth])
        y.set_shape((None, None, FLAGS.pix250m, FLAGS.pix250m))
        sequence_lengths.set_shape(None)

    return (x, sequence_lengths), (y,)


def input_eval(features):
    with tf.name_scope("raw"):
        x250, x500, doy, year = features

        x250 = tf.cast(x250, tf.float32, name="x250")
        x500 = tf.cast(x500, tf.float32, name="x500")
        doy = tf.cast(doy, tf.float32, name="doy")
        year = tf.cast(year, tf.float32, name="year")

    # integer sequence lengths per batch for dynamic_rnn masking
    sequence_lengths = tf.reduce_sum(tf.cast(x250[:, :, 0, 0, 0] > 0, tf.int32), axis=1,
                                                name="sequence_lengths")

    def resize(tensor, new_height, new_width):
        b = tf.shape(tensor)[0]
        t = tf.shape(tensor)[1]
        h = tf.shape(tensor)[2]
        w = tf.shape(tensor)[3]
        d = tf.shape(tensor)[4]

        # stack batch on times to fit 4D requirement of resize_tensor
        stacked_tensor = tf.reshape(tensor, [b * t, h, w, d])
        reshaped_stacked_tensor = tf.image.resize(stacked_tensor, size=(new_height, new_width))
        return tf.reshape(reshaped_stacked_tensor, [b, t, new_height, new_width, d])

    def expand3x(vector):
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        vector = tf.expand_dims(vector, -1)
        return vector

    with tf.name_scope("reshaped"):
        b = tf.shape(x250)[0]
        t = tf.shape(x250)[1]
        px = tf.shape(x250)[2]

        x500 = tf.identity(resize(x500, px, px), name="x500")

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(x250, name="x250"))
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x500)

        # expand
        doymat = tf.multiply(expand3x(doy), tf.ones((b, t, px, px, 1)), name="doy")
        yearmat = tf.multiply(expand3x(year), tf.ones((b, t, px, px, 1)), name="year")

        x = tf.concat((x250, x500, doymat, yearmat), axis=-1, name="x")
        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)

        if FLAGS.experiment == 'bands':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bandswodoy':
            bands250m = 2
            bands500m = 5
        elif FLAGS.experiment == 'bands250m':
            bands250m = 2
            bands500m = 1
        elif FLAGS.experiment == 'bandswoblue':
            bands250m = 2
            bands500m = 4
        elif FLAGS.experiment == 'bandsaux':
            bands250m = 5
            bands500m = 5
        elif FLAGS.experiment == 'evi2':
            bands250m = 1
            bands500m = 1
        elif FLAGS.experiment == 'indices':
            bands250m = 7
            bands500m = 5
        else:
            bands250m = 10
            bands500m = 5

        FLAGS.num_bands_250m = bands250m
        FLAGS.num_bands_500m = bands500m

        # set depth of x for convolutions
        depth = FLAGS.num_bands_250m + FLAGS.num_bands_500m + 2  # doy and year
        # depth = bands250m + bands500m + 2  # doy and year

        # dynamic shapes. Fill for debugging
        x.set_shape([None, None, FLAGS.pix250m, FLAGS.pix250m, depth])
        sequence_lengths.set_shape(None)

    return x, sequence_lengths


def _model_fn(features, labels, mode, params):
  """MTLCC model

  Args:
    features: a batch of features.
    labels: a batch of labels or None if predicting.
    mode: an instance of tf.estimator.ModeKeys.
    params: a dict of additional params.

  Returns:
    A tf.estimator.EstimatorSpec that fully defines the model that will be run
      by an Estimator.
  """

  def parse(var, dtype):
      if type(var) == dtype:
          return var
      else:
          return eval(var)

  num_classes = parse(FLAGS.num_classes, int)

  if mode == tf.estimator.ModeKeys.PREDICT:
      # input pipeline
      with tf.name_scope("input"):
          x, sequence_lengths = input_eval(features)

      logits = inference(input=(x, sequence_lengths), num_classes=num_classes)

      prediction_scores = tf.nn.softmax(logits=logits, name="prediction_scores")
      pred = tf.argmax(prediction_scores, 3, name="predictions")

      predictions = {
          "pred": pred,
          "pred_sc": prediction_scores}

      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  else:
      with tf.name_scope("input"):
          (x, sequence_lengths), (alllabels,) = input(features, labels)
          alllabels = alllabels

      logits = inference(input=(x, sequence_lengths), num_classes=num_classes)

      # take first label -> assume labels do not change over timeseries
      first_labelmap = alllabels[ :, 0 ]

      # create one-hot labelmap from 0-num_classes
      labels = tf.one_hot(first_labelmap, num_classes + 1)

      # # mask out class 0 -> unknown
      unknown_mask = tf.cast(labels[ :, :, :, 0 ], tf.bool)
      not_unknown_mask = tf.logical_not(unknown_mask)

      # keep rest of labels
      labels = labels[ :, :, :, 1: ]

      global_step = tf.compat.v1.train.get_or_create_global_step()

      loss_op = loss(logits=logits, labels=labels, mask=not_unknown_mask, name="loss")
      # loss_op = softmax_focal_loss(logits=logits, labels=labels, mask=not_unknown_mask, gamma=2, alpha=0.25, name="loss")

      ao_ops = eval_oa(logits=logits, labels=labels, mask=not_unknown_mask)
      summary(ao_ops[0], loss_op)

      if mode == tf.estimator.ModeKeys.EVAL:
          eval_metric_ops = {"accuracy": ao_ops}
          return tf.estimator.EstimatorSpec(mode, loss=loss_op, eval_metric_ops=eval_metric_ops)

      elif mode == tf.estimator.ModeKeys.TRAIN:
          print("building optimizer...")
          train_op = optimize(loss_op, global_step, FLAGS.optimizertype, name="train_op")
          logging_hook = tf.estimator.LoggingTensorHook({"accuracy": ao_ops[0], 'loss': loss_op,  'global_step': global_step}, every_n_iter=64)

          # write FLAGS to file
          cfg = configparser.ConfigParser()
          flags_dict = dict()
          for name in FLAGS:
              flags_dict[name] = str(FLAGS[name].value)

          cfg["flags"] = flags_dict  # FLAGS.__flags #  broke tensorflow=1.5

          if not os.path.exists(params.model_dir):
              os.makedirs(params.model_dir)
          path = os.path.join(params.model_dir, MODEL_CFG_FILENAME)

          print("writing parameters to {}".format(path))
          with file_io.FileIO(path, 'w') as configfile:  # gcp
              cfg.write(configfile)

          return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op, training_hooks=[logging_hook])

      else:
          raise NotImplementedError('Unknown mode {}'.format(mode))


def convrnn_layer(input, filter, is_train=True, kernel=FLAGS.convrnn_kernel, sequence_lengths=None, bidirectional=True,
                  convcell='gru', scope="convrnn"):
    with tf.compat.v1.variable_scope(scope):

        x = input

        px = x.get_shape()[3]

        if FLAGS.convrnn_compression_filters > 0:
            x = conv_bn_relu(input=x, is_train=is_train, filter=FLAGS.convrnn_compression_filters, kernel=(1, 1, 1),
                             dilation_rate=(1, 1, 1), var_scope="comp")

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(input, "input"))

        if convcell == 'gru':
            cell = utils.ConvGRUCell((px, px), filter, kernel, activation=tf.nn.tanh,
                                       normalize=FLAGS.convrnn_normalize)

        elif convcell == 'lstm':
            cell = utils.ConvLSTMCell((px, px), filter, kernel, activation=tf.nn.tanh,
                                        normalize=FLAGS.convrnn_normalize, peephole=FLAGS.peephole)
        else:
            raise ValueError("convcell argument {} not valid either 'gru' or 'lstm'".format(convcell))

        ## add dropout wrapper to cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=FLAGS.recurrent_dropout_i,
                                             state_keep_prob=FLAGS.recurrent_dropout_i)

        if bidirectional:
            outputs, final_states = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=x,
                                                                              sequence_length=sequence_lengths,
                                                                              dtype=tf.float32, time_major=False,
                                                                              swap_memory=FLAGS.swap_memory)

            concat_outputs = tf.concat(outputs, -1)

            if convcell == 'gru':
                concat_final_state = tf.concat(final_states, -1)
            elif convcell == 'lstm':
                fw_final, bw_final = final_states
                concat_final_state = LSTMStateTuple(
                    c=tf.concat((fw_final.c, bw_final.c), -1),
                    h=tf.concat((fw_final.h, bw_final.h), -1)
                )

        else:
            concat_outputs, concat_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=input,
                                                                   sequence_length=sequence_lengths,
                                                                   dtype=tf.float32, time_major=False)

        if convcell == 'lstm':
            concat_final_state = concat_final_state.c

        else:
            tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME,
                                           tf.identity(concat_outputs, name="outputs"))
            tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME,
                                           tf.identity(concat_final_state, name="final_states"))

        return concat_outputs, concat_final_state


def conv_bn_relu(var_scope="name_scope", is_train=True, **kwargs):
    with tf.compat.v1.variable_scope(var_scope):

        if FLAGS.cnn_activation == 'relu':
            activation_function = tf.nn.relu
        elif FLAGS.cnn_activation == 'leaky_relu':
            activation_function = tf.nn.leaky_relu
        else:
            raise ValueError("please provide valid 'cnn_activation' FLAG. either 'relu' or 'leaky_relu'")

        is_train = tf.constant(is_train, dtype=tf.bool)
        x = conv_layer(**kwargs)
        # x = Batch_Normalization(x, is_train, "bn") #deprecated replaced by keras API BN
        bn = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, center=True, moving_mean_initializer='zeros') #source https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/batch_normalization/mnist_classifier/trainer/model.py
        x = bn(x, training=is_train)
        x = activation_function(x)

        tf.compat.v1.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)
        return x


def conv_layer(input, filter, kernel, dilation_rate=(1, 1, 1), stride=1, conv_fun=tf.keras.layers.Conv3D,
               layer_name="conv"):  # based on https://github.com/tensorflow/tensorflow/issues/26145

    with tf.name_scope(layer_name):
        # pad input to required sizes for same output dimensions
        input = pad(input, kernel, dilation_rate, padding="REFLECT")

        network = conv_fun(use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='VALID',
                           dilation_rate=dilation_rate)(inputs=input)

        return network


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   # updates_collections=tf.GraphKeys.UPDATE_OPS,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def pad(input, kernel, dilation, padding="REFLECT"):
    """https://www.tensorflow.org/api_docs/python/tf/pad"""

    # determine required padding sizes
    def padsize(kernel, dilation):
        p = []
        for k, d in zip(kernel, dilation):
            p.append(int(int(k / 2) * d))
            # p.append(k/2)
        return p

    padsizes = padsize(kernel, dilation)

    # [bleft,bright], [tleft,tright], [hleft,hright], [wleft,wright],[dleft,dright]
    paddings = tf.constant([[0, 0]] + [[p, p] for p in padsizes] + [[0, 0]], dtype=tf.int32)

    return tf.pad(input, paddings, padding)


def build_estimator(run_config):
    """Returns TensorFlow estimator."""

    estimator = tf.estimator.Estimator(
        model_fn=_model_fn,
        config=run_config,
        params=run_config,
    )

    return estimator