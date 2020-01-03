import os
import pickle
import logging
import random
from pprint import pformat

from absl import flags
import numpy as np
import tensorflow as tf

from fin.model import model_utils
from fin.model import fin_model
from fin.model import anml_model
from fin.gpu_utils import assign_to_gpu, average_grads_and_vars
from fin.data_utils import data_handler

# Test info
flags.DEFINE_string("info", default=None,
                    help="Show on log.")
# GPU config
flags.DEFINE_integer("num_hosts", default=1,
                     help="Number of hosts")
flags.DEFINE_integer("num_core_per_host", default=1,
                     help="Number of cores per host")
flags.DEFINE_bool("use_tpu", default=False,
                  help="Whether to use TPUs for training.")
flags.DEFINE_string("gpu", default=None,
                    help="which gpu pci bus id to use. ex: 0,1")
# Checkpoints
flags.DEFINE_string("model_dir", default=None,
                    help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model.")
flags.DEFINE_string("tfserving_config_path", default=None,
                    help="tfserving config path for config files.")
flags.DEFINE_string("log_tensorboard_dir", default=None,
                    help="TensorBoard summary log path.")
flags.DEFINE_bool("log_gradient", default=False,
                  help="Whether to log gradients for training.")
# Data config
flags.DEFINE_integer("batch_size", default=16,
                     help="Size of train batch.")
flags.DEFINE_string("file_dir", default=None,
                    help="training data file path dir.")
flags.DEFINE_integer("seq_len", default=32,
                     help="Number of time sequence title data to process.")

# Model
flags.DEFINE_string("config_path", default=None,
                    help="config path for config files.")

flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
                  help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("ff_activation", default="gelu",
                    help="Activation type used in position-wise feed-forward.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# training
flags.DEFINE_integer("train_steps", default=25000,
                     help="Number of training steps")
flags.DEFINE_integer("iterations", default=10,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=1000,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("warmup_steps", default=100, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("weight_decay", default=0., help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")


FLAGS = flags.FLAGS


def log_std(st, file):
    with open(file, "a+") as f:
        f.write(st + "\n")


def get_model_fn(session):
    def model_fn(tensors, is_training):
        inp_value, inp_cate, inp_mask, price, inp_trace_mask, target = tensors

        nm_name = "neuromodulatory"
        p_name = "prediction"
        with tf.variable_scope(nm_name):
            _, nm_summary = fin_model.build_train_anml_trace_net(
                    FLAGS, inp_value, inp_cate, inp_mask, inp_trace_mask)
        with tf.variable_scope(p_name):
            _, p_summary = fin_model.build_train_anml_trace_net(
                    FLAGS, inp_value, inp_cate, inp_mask, inp_trace_mask)

        model = anml_model.ANMLModel(
            session, nm_name, p_name, nm_summary, p_summary,
            tf.random_normal_initializer(mean=0.0, stddev=0.02))

        predict_tensor = model.build_predict_tensor(1)
        predict_value = tf.nn.sigmoid(predict_tensor)

        target = target[:, None]
        trace_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=predict_tensor, labels=target)
        trace_loss = tf.reduce_mean(trace_loss)

        # Check predict accuracy
        target_bool = tf.math.greater_equal(target, 0.5)
        predict_bool = tf.math.greater_equal(predict_value, 0.5)
        correct = tf.cast(tf.equal(target_bool, predict_bool), tf.float32)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # GPU
        total_loss = trace_loss

        # inner loop
        p_vars = tf.trainable_variables(p_name)
        p_grads = tf.gradients(total_loss, p_vars)
        p_clipped, p_gnorm = tf.clip_by_global_norm(p_grads, 1.0)
        inner_grads_and_vars = list(zip(p_clipped, p_vars))

        # outer loop
        nm_vars = tf.trainable_variables(nm_name)
        nm_grads = tf.gradients(total_loss, p_vars + nm_vars)
        nm_clipped, nm_gnorm = tf.clip_by_global_norm(nm_grads, 1.0)
        otter_grads_and_vars = list(zip(nm_clipped, p_vars + nm_vars))

        return (inner_grads_and_vars, otter_grads_and_vars, trace_loss, correct,
                predict_value, model)

    return model_fn


def single_core_graph(tensors, session):
    model_fn = get_model_fn(session)

    model_ret = model_fn(
            tensors=tensors,
            is_training=True)

    return model_ret


def train():
    # Get session and graph
    # graph = tf.get_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True, gpu_options=gpu_options)
    )
    # Load data
    prefix = FLAGS.file_dir
    value_keys_filepath = os.path.join(prefix, "value_keys.txt")
    csv_data_filepath = os.path.join(prefix, "indust_price_combined_data_20200110.csv")
    industry_filepath = os.path.join(prefix, "industry_cate.txt")

    value_keys = data_handler.load_value_keys(value_keys_filepath)
    _, max_value, max_price = data_handler.load_csv_trace_data(csv_data_filepath, value_keys)
    industry_mapping = data_handler.load_industry_cate(industry_filepath)

    train_pickle_path = os.path.join(prefix, "train_trace.pickle")
    valid_pickle_path = os.path.join(prefix, "valid_trace.pickle")
    test_pickle_path = os.path.join(prefix, "test_trace.pickle")
    with open(train_pickle_path, "rb") as f:
        train_data = pickle.load(f)
    with open(valid_pickle_path, "rb") as f:
        valid_data = pickle.load(f)
    with open(test_pickle_path, "rb") as f:
        test_data = pickle.load(f)

    # Build train and valid dataset
    train_gen_fn = data_handler.get_trace_generator_fn(
            train_data, max_value, max_price, industry_mapping, FLAGS.seq_len)
    valid_gen_fn = data_handler.get_trace_generator_fn(
            valid_data, max_value, max_price, industry_mapping, FLAGS.seq_len)
    test_gen_fn = data_handler.get_trace_generator_fn(
            test_data, max_value, max_price, industry_mapping, FLAGS.seq_len)
    total_data = {}
    total_data.update(train_data)
    total_data.update(valid_data)
    total_data.update(test_data)
    predict_gen_fn = data_handler.get_predict_generator_fn(
            total_data, max_value, max_price, industry_mapping, FLAGS.seq_len)

    # tensors = data_loader.input_tensors
    inp_value_ph = tf.placeholder(tf.float32, shape=[None, None, None, None])
    inp_cate_ph = tf.placeholder(tf.int32, shape=[None, None])
    inp_mask_ph = tf.placeholder(tf.float32, shape=[None, None, None])
    price_ph = tf.placeholder(tf.float32, shape=[None, None])
    inp_trace_mask_ph = tf.placeholder(tf.float32, shape=[None, None])
    target = tf.placeholder(tf.float32, shape=[None])

    tensors = (inp_value_ph, inp_cate_ph, inp_mask_ph, price_ph, inp_trace_mask_ph, target)

    split_num = FLAGS.num_core_per_host if FLAGS.num_core_per_host else 1
    assert split_num == 1
    split_tensors = [tf.split(tensor, split_num, 0) for tensor in tensors]

    models = []
    tower_inner_grads_and_vars = []
    tower_outer_grads_and_vars = []
    tower_trace_losses = []
    tower_corrects = []
    tower_predicts = []
    # Build multi gpu graph
    if FLAGS.num_core_per_host == 0:
        i = 0
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            inp_tensors = [ten[i] for ten in split_tensors]
            model_ret = single_core_graph(inp_tensors, sess)

            (inner_grads_and_vars_i, outer_grads_and_vars_i,
             trace_loss_i, correct_i, predict_value_i, model_i) = model_ret

            tower_inner_grads_and_vars.append(inner_grads_and_vars_i)
            tower_outer_grads_and_vars.append(outer_grads_and_vars_i)
            tower_trace_losses.append(trace_loss_i)
            tower_corrects.append(correct_i)
            tower_predicts.append(predict_value_i)
            models.append(model_i)

    for i in range(FLAGS.num_core_per_host):
        if i == 0:
            reuse = None
        else:
            reuse = True
        with tf.device(assign_to_gpu(i, "/gpu:0")), \
                tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            inp_tensors = [ten[i] for ten in split_tensors]
            model_ret = single_core_graph(inp_tensors, sess)

            (inner_grads_and_vars_i, outer_grads_and_vars_i,
             trace_loss_i, correct_i, predict_value_i, model_i) = model_ret

            tower_inner_grads_and_vars.append(inner_grads_and_vars_i)
            tower_outer_grads_and_vars.append(outer_grads_and_vars_i)
            tower_trace_losses.append(trace_loss_i)
            tower_corrects.append(correct_i)
            tower_predicts.append(predict_value_i)
            models.append(model_i)

    # average losses and gradients across towers
    if len(tower_inner_grads_and_vars) > 1:
        trace_loss = tf.add_n(tower_trace_losses) / len(tower_trace_losses)
        inner_grads_and_vars = average_grads_and_vars(tower_inner_grads_and_vars)
        outer_grads_and_vars = average_grads_and_vars(tower_outer_grads_and_vars)
        corrects = tf.concat(tower_corrects, axis=0)
        accuracy = tf.reduce_mean(corrects)
        predict_tensor = tf.concat(tower_predicts, axis=0)
    else:
        trace_loss = tower_trace_losses[0]
        inner_grads_and_vars = tower_inner_grads_and_vars[0]
        outer_grads_and_vars = tower_outer_grads_and_vars[0]
        corrects = tower_corrects[0]
        accuracy = tf.reduce_mean(corrects)
        predict_tensor = tf.concat(tower_predicts, axis=0)
        model = models[0]
    total_loss = trace_loss

    # Log gradients
    # if FLAGS.log_tensorboard_dir:
    #     if FLAGS.log_gradient:
    #         tf.logging.info("Log gradients:")
    #         for grad, var in grads_and_vars:
    #             var_name = var.name.split(":")[0]
    #             tf.logging.info("    Variable name: {}".format(var_name))
    #             tf.summary.histogram(var.name, grad)

    # get train op
    inner_learning_rate = 1e-4
    # outer_learning_rate = 2e-5
    inner_optimizer = tf.train.MomentumOptimizer(
            learning_rate=inner_learning_rate,
            momentum=0.)
    # outer_optimizer = tf.train.AdamOptimizer(
    #         learning_rate=outer_learning_rate,
    #         epsilon=FLAGS.adam_epsilon)

    outer_grads = []
    outer_vars = []
    outer_grad_placeholders = []
    grad_ph_and_vars = []
    for grad, var in outer_grads_and_vars:
        if grad is None:
            grad_ph_and_vars.append((None, var))
            continue
        if isinstance(grad, tf.IndexedSlices):
            ind = grad.indices
            values = grad.values
            indices_ph = tf.placeholder(ind.dtype.base_dtype, shape=ind.get_shape())
            values_ph = tf.placeholder(values.dtype.base_dtype, shape=values.get_shape())
            indexed_slice = tf.IndexedSlices(values_ph, indices_ph, grad.dense_shape)

            outer_grads.append(ind)
            outer_grads.append(values)
            outer_vars.append(var)

            outer_grad_placeholders.append(indices_ph)
            outer_grad_placeholders.append(values_ph)
            grad_ph_and_vars.append((indexed_slice, var))
        else:
            outer_grads.append(grad)
            outer_vars.append(vars)
            outer_grad_ph = tf.placeholder(grad.dtype.base_dtype, shape=grad.get_shape())
            outer_grad_placeholders.append(
                outer_grad_ph)
            grad_ph_and_vars.append((outer_grad_ph, var))

    inner_train_op = inner_optimizer.apply_gradients(inner_grads_and_vars)
    outer_train_op, learning_rate, gnorm = model_utils.get_train_op(
        FLAGS, None, grads_and_vars=grad_ph_and_vars)
    # outer_train_op = outer_optimizer.apply_gradients(grad_ph_and_vars)

    # saver = tf.train.Saver()

    # Init
    model_utils.init_from_checkpoint(FLAGS, global_vars=False)
    sess.run(tf.global_variables_initializer())

    # Start training loop.
    fetches = [total_loss, trace_loss, accuracy]

    def predict_data(data_gen, tensors, predict_t):
        predict_data = []

        i = 0

        for data in data_gen:
            i += 1
            print("Predict data {}".format(i), end="\r")
            (value_list_arr, industry_list_arr, mask_list_arr, price_list_arr,
             trace_mask_arr, current_mean, ticker) = data

            inp_data = (value_list_arr[None, :], industry_list_arr[None, :],
                        mask_list_arr[None, :], price_list_arr[None, :],
                        trace_mask_arr[None, :])

            inp_tensor_ph = tensors[:-1]
            feed_dict = dict(zip(inp_tensor_ph, inp_data))

            predict_np = sess.run(
                    [predict_t], feed_dict=feed_dict)

            predict_data.append((ticker, float(predict_np), current_mean))

        return predict_data

    def eval_data(data_gen, tensors, fetches):
        v_loss_np = 0.
        v_accuracy = 0.
        run_valid_step = 0
        for data in data_gen:
            (value_list_arr, industry_list_arr, mask_list_arr, price_list_arr,
             trace_mask_arr, target) = data
            data = (value_list_arr[None, :], industry_list_arr[None, :],
                    mask_list_arr[None, :], price_list_arr[None, :],
                    trace_mask_arr[None, :], np.array([target]))

            feed_dict = dict(zip(tensors, data))
            fetched = sess.run(
                    fetches, feed_dict=feed_dict)
            v_loss_np += fetched[0]
            v_accuracy += fetched[2]
            run_valid_step += 1

        v_loss_np = v_loss_np / run_valid_step
        v_accuracte = v_accuracy / run_valid_step

        return v_loss_np, v_accuracte

    # log summary
    # if FLAGS.log_tensorboard_dir:
    #     if not tf.gfile.Exists(FLAGS.log_tensorboard_dir):
    #         tf.gfile.MakeDirs(FLAGS.log_tensorboard_dir)
    #     summary_ops = tf.summary.merge_all()
    #     summary_writer = tf.summary.FileWriter(FLAGS.log_tensorboard_dir, sess.graph)
    #     fetches.append(summary_ops)

    # Model dir
    if FLAGS.model_dir:
        if not tf.gfile.Exists(FLAGS.model_dir):
            tf.gfile.MakeDirs(FLAGS.model_dir)
        # Save FLAGS
        flags_str = FLAGS.flags_into_string()
        flag_file = os.path.join(FLAGS.model_dir, "flag_file.txt")
        with open(flag_file, "w") as f:
            f.write(flags_str)

    # Log variable names.
    tf.logging.info(
        pformat(tf.trainable_variables(tf.get_variable_scope().name))
    )
    if FLAGS.info and FLAGS.model_dir:
        tf.logging.info(FLAGS.info)
        log_std(FLAGS.info, os.path.join(FLAGS.model_dir, "/log.txt"))

    outer_loss = 0.
    inner_loss = 0.
    outer_accuracy = 0.
    inner_accuracy = 0.
    mean_outer_loss = 0.
    mean_outer_accuracy = 0.
    prev_step = -1
    epoch = 1
    outer_num = 0
    log_format = ("Epoch: {:>3} - Outer: {:>3} - Inner: {:>2} | "
                  "Outer: loss {:>6.5f} | accuracy: {:.3f} | "
                  "Inner: loss {:>6.5f} | accuracy: {:.3f}        ")
    data_gen = train_gen_fn()

    while True:

        # Inner loop
        # Cache predict vars and restore before outer gradients update.

        inner_loss = 0.
        inner_accuracy = 0.
        trajectory_batch = []
        model.cache_p_values()
        model.init_final_vars()
        for _ in range(FLAGS.batch_size):
            try:
                data = next(data_gen)
            except StopIteration:
                data_gen = train_gen_fn()
                data = next(data_gen)
                epoch += 1
            trajectory_batch.append(data)

        for inner_i in range(20):
            random.shuffle(trajectory_batch)
            batch = trajectory_batch[:4]
            batch_data_arr = []
            for data_tuple in zip(*batch):
                batch_data_arr.append(np.array(data_tuple))

            feed = dict(zip(tensors, batch_data_arr))
            fetched = sess.run(fetches + [inner_train_op],
                               feed_dict=feed)
            total_loss_np, trace_loss_np, accuracy_np = fetched[:-1]

            inner_loss += total_loss_np
            inner_accuracy += accuracy_np

            log_str = log_format.format(
                epoch, outer_num, inner_i+1,
                mean_outer_loss,
                mean_outer_accuracy,
                total_loss_np, accuracy_np)
            print(log_str, end="\r")

        inner_loss = inner_loss / 20
        inner_accuracy = inner_accuracy / 20
        log_str = log_format.format(
                epoch, outer_num, inner_i+1,
                outer_loss / (outer_num - prev_step),
                outer_accuracy / (outer_num - prev_step),
                inner_loss, inner_accuracy)
        print(log_str, end="\r")

        # Outer loop
        sample_batch = []
        for i in range(FLAGS.batch_size):
            try:
                data = next(data_gen)
            except StopIteration:
                data_gen = train_gen_fn()
                data = next(data_gen)
                epoch += 1
            sample_batch.append(data)
        outer_batch = trajectory_batch + sample_batch

        batch_data_arr = []
        for data_tuple in zip(*outer_batch):
            batch_data_arr.append(np.array(data_tuple))

        feed = dict(zip(tensors, batch_data_arr))

        fetched = sess.run(fetches + outer_grads,
                           feed_dict=feed)
        total_loss_np, trace_loss_np, accuracy_np = fetched[:3]
        grads = fetched[3:]
        model.restore_p_values()
        sess.run(outer_train_op,
                 feed_dict=dict(zip(outer_grad_placeholders, grads)))

        outer_loss += total_loss_np
        outer_accuracy += accuracy_np

        mean_outer_loss = outer_loss / (outer_num - prev_step)
        mean_outer_accuracy = outer_accuracy / (outer_num - prev_step)

        # if FLAGS.log_tensorboard_dir:
        #     if curr_step % 100 == 0:
        #         summary_writer.add_summary(fetched[-1], curr_step)

        # print status
        if outer_num > 0 and outer_num % FLAGS.iterations == 0:
            # Run valid data
            # v_loss_np, v_price_loss_np, v_trace_loss_np = eval_data("valid")
            log_str = log_format.format(
                epoch, outer_num, inner_i+1,
                mean_outer_loss, mean_outer_accuracy,
                inner_loss, inner_accuracy)
            tf.logging.info(log_str)
            if FLAGS.info:
                tf.logging.info(FLAGS.info)
            if FLAGS.model_dir:
                log_std(log_str, os.path.join(FLAGS.model_dir, "/log.txt"))
            outer_loss = 0.
            inner_loss = 0.
            outer_accuracy = 0.
            inner_accuracy = 0.
            prev_step = outer_num

        if outer_num > 0 and outer_num % (FLAGS.iterations*10) == 0:
            gen = valid_gen_fn()
            v_loss_np, v_accuracte = eval_data(gen, tensors, fetches)
            valid_str = "Valid loss: {:.3f} | accuracy: {:.3f}".format(v_loss_np, v_accuracte)
            tf.logging.info(valid_str)

        outer_num += 1

        # save model
        # if FLAGS.save_steps and FLAGS.model_dir:
        #     if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
        #         save_path = os.path.join(FLAGS.model_dir, "model_{}.ckpt".format(curr_step))
        #         saver.save(sess, save_path)
        #         tf.logging.info("Model saved in path: {}".format(save_path))

        if outer_num >= FLAGS.train_steps:
            tf.logging.info("Finished training.")
            # tf.logging.info("Run eval...")
            # v_loss_np, v_price_loss_np, v_trace_loss_np = eval_data("valid")
            # t_loss_np, t_price_loss_np, t_trace_loss_np = eval_data("test")
            # if (curr_step - prev_step) > 0:
            #     mean_total_loss = total_loss / (curr_step - prev_step)
            #     mean_trace_loss = trace_loss / (curr_step - prev_step)
            #     mean_price_loss = price_loss / (curr_step - prev_step)
            #     mean_gnorm = gnorm / (curr_step - prev_step)
            # log_status_str = "Epoch: {:>3} [{}] | gnorm {:>4.2f} lr {:>9.3e}".format(
            #         epoch, curr_step, gnorm, lr_np)
            # log_str = ("{} : loss {:>6.5f} | trace_loss {:>6.5f} | price_loss {:>6.5f}")
            # train_log = log_str.format("Train", mean_total_loss, mean_trace_loss, mean_price_loss)
            # valid_log = log_str.format("Valid", v_loss_np, v_trace_loss_np, v_price_loss_np)
            # test_log = log_str.format("Test ", t_loss_np, t_trace_loss_np, t_price_loss_np)
            # tf.logging.info(log_status_str)
            # tf.logging.info(train_log)
            # tf.logging.info(valid_log)
            # tf.logging.info(test_log)
            gen = test_gen_fn()

            model.init_final_vars()
            data_gen = train_gen_fn()
            batch = []
            while True:
                try:
                    data = next(data_gen)
                except StopIteration:
                    break
                batch.append(data)

                if len(batch) >= FLAGS.batch_size:
                    batch_data_arr = []
                    for data_tuple in zip(*batch):
                        batch_data_arr.append(np.array(data_tuple))

                    feed = dict(zip(tensors, batch_data_arr))
                    sess.run(inner_train_op, feed_dict=feed)
                    batch = []
            if len(batch) > 0:
                batch_data_arr = []
                for data_tuple in zip(*batch):
                    batch_data_arr.append(np.array(data_tuple))

                feed = dict(zip(tensors, batch_data_arr))
                sess.run(inner_train_op, feed_dict=feed)

            v_loss_np, v_accuracte = eval_data(gen, tensors, fetches)
            valid_str = "Test loss: {:.3f} | accuracy: {:.3f}".format(v_loss_np, v_accuracte)
            tf.logging.info(valid_str)

            data_gen = predict_gen_fn()
            p_data = predict_data(data_gen, tensors, predict_tensor)

            with open("test_predict.txt", "w") as f:
                for ticker, pred, current_mean in p_data:
                    f.write(ticker + "\t" + str(pred) + "\t" + str(current_mean))

            break


def main(unused_argv):
    del unused_argv
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.gpu:
        FLAGS.num_core_per_host = len(FLAGS.gpu.split(","))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    else:
        FLAGS.num_core_per_host = 0

    train()


if __name__ == "__main__":
    # Add time stamp to logging.
    logger = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # logger.handlers[0].setFormatter(formatter)
    tf.app.run()
