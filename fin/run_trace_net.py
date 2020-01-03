import os
import pickle
import logging
from pprint import pformat

from absl import flags
import numpy as np
import tensorflow as tf

from fin.model import model_utils
from fin.model import fin_model
from fin.gpu_utils import assign_to_gpu, average_grads_and_vars
from fin.data_utils import base_data_loader, data_handler

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
flags.DEFINE_integer("train_steps", default=20000,
                     help="Number of training steps")
flags.DEFINE_integer("iterations", default=500,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=1000,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("warmup_steps", default=1000, help="number of warmup steps")
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


def get_model_fn():
    def model_fn(tensors, is_training):
        inp_value, inp_cate, inp_mask, price, inp_trace_mask, target = tensors

        output, summary, price_loss, trace_loss, predict_logits = fin_model.build_train_trace_net(
                FLAGS, inp_value, inp_cate, inp_mask, inp_trace_mask, price, target)

        # Check predict accuracy
        predict_value = tf.nn.sigmoid(predict_logits)
        target_bool = tf.math.greater_equal(target, 0.5)
        predict_bool = tf.math.greater_equal(predict_value, 0.5)
        correct = tf.cast(tf.equal(target_bool, predict_bool), tf.float32)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # GPU
        assert is_training
        total_loss = price_loss * 0.1 + trace_loss
        all_vars = tf.trainable_variables()
        grads = tf.gradients(total_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))

        return (grads_and_vars, price_loss, trace_loss, correct)

    return model_fn


def single_core_graph(tensors):
    model_fn = get_model_fn()

    model_ret = model_fn(
            tensors=tensors,
            is_training=True)

    return model_ret


def train():
    # Get session and graph
    graph = tf.get_default_graph()
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

    # Split train and valid data
    # random.seed(777)
    # valid_percent = 0.05  # 5% valid data
    # total_keys = list(data.keys())
    # total_len = len(total_keys)
    # valid_len = int(total_len * valid_percent)

    # random.shuffle(total_keys)
    # valid_titles = total_keys[:valid_len]
    # train_titles = total_keys[valid_len:]

    # valid_data = {}
    # train_data = {}
    # for title in valid_titles:
    #     valid_data[title] = data[title]
    # for title in train_titles:
    #     train_data[title] = data[title]
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
    train_dataset = data_handler.get_trace_dataset(train_gen_fn)
    valid_dataset = data_handler.get_trace_dataset(valid_gen_fn)
    test_dataset = data_handler.get_trace_dataset(test_gen_fn)

    data_loader = base_data_loader.BaseDataLoader(graph, sess, None)
    data_loader.add_datasets({"train": train_dataset}, batch_size=FLAGS.batch_size)
    data_loader.add_datasets({"valid": valid_dataset}, batch_size=FLAGS.batch_size)
    data_loader.add_datasets({"test": test_dataset}, batch_size=1)

    tensors = data_loader.input_tensors

    split_num = FLAGS.num_core_per_host if FLAGS.num_core_per_host else 1
    assert split_num == 1
    split_tensors = [tf.split(tensor, split_num, 0) for tensor in tensors]

    tower_grads_and_vars = []
    tower_price_losses = []
    tower_trace_losses = []
    tower_corrects = []
    # Build multi gpu graph
    if FLAGS.num_core_per_host == 0:
        i = 0
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            inp_tensors = [ten[i] for ten in split_tensors]
            model_ret = single_core_graph(inp_tensors)

            (grads_and_vars_i, price_loss_i, trace_loss_i, correct_i) = model_ret

            tower_price_losses.append(price_loss_i)
            tower_trace_losses.append(trace_loss_i)
            tower_grads_and_vars.append(grads_and_vars_i)
            tower_corrects.append(correct_i)

    for i in range(FLAGS.num_core_per_host):
        if i == 0:
            reuse = None
        else:
            reuse = True
        with tf.device(assign_to_gpu(i, "/gpu:0")), \
                tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            inp_tensors = [ten[i] for ten in split_tensors]
            model_ret = single_core_graph(inp_tensors)

            (grads_and_vars_i, price_loss_i, trace_loss_i, correct_i) = model_ret

            tower_price_losses.append(price_loss_i)
            tower_trace_losses.append(trace_loss_i)
            tower_grads_and_vars.append(grads_and_vars_i)
            tower_corrects.append(correct_i)

    # average losses and gradients across towers
    if len(tower_grads_and_vars) > 1:
        price_loss = tf.add_n(tower_price_losses) / len(tower_price_losses)
        trace_loss = tf.add_n(tower_trace_losses) / len(tower_trace_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
        corrects = tf.concat(tower_corrects, axis=0)
        accuracy = tf.reduce_mean(corrects)
    else:
        price_loss = tower_price_losses[0]
        trace_loss = tower_trace_losses[0]
        grads_and_vars = tower_grads_and_vars[0]
        corrects = tower_corrects[0]
        accuracy = tf.reduce_mean(corrects)

    total_loss = price_loss + trace_loss

    # Log gradients
    if FLAGS.log_tensorboard_dir:
        if FLAGS.log_gradient:
            tf.logging.info("Log gradients:")
            for grad, var in grads_and_vars:
                var_name = var.name.split(":")[0]
                tf.logging.info("    Variable name: {}".format(var_name))
                tf.summary.histogram(var.name, grad)

    # get train op
    train_op, learning_rate, gnorm = model_utils.get_train_op(
            FLAGS, None, grads_and_vars=grads_and_vars)
    global_step = tf.train.get_global_step()

    saver = tf.train.Saver()

    # Init
    model_utils.init_from_checkpoint(FLAGS, global_vars=False)
    sess.run(tf.global_variables_initializer())
    sess.run(data_loader.get_initializer("train"))
    sess.run(data_loader.get_initializer("valid"))
    sess.run(data_loader.get_initializer("test"))

    # Start training loop.
    fetches = [total_loss, global_step, learning_rate, gnorm,
               price_loss, trace_loss, accuracy, train_op]

    def eval_data(data_name):
        run_valid = True
        v_loss_np = 0.
        v_price_loss_np = 0.
        v_trace_loss_np = 0.
        v_accuracy_np = 0.
        run_valid_step = 0
        while run_valid:
            try:
                fetched = sess.run(
                        fetches[:-1], feed_dict=data_loader.get_feed_dict(data_name))
                run_valid_step += 1
                v_loss_np += fetched[0]
                v_price_loss_np += fetched[4]
                v_trace_loss_np += fetched[5]
                v_accuracy_np += fetched[6]
            except tf.errors.OutOfRangeError:
                run_valid = False
                sess.run(data_loader.get_initializer(data_name))
        v_loss_np = v_loss_np / run_valid_step
        v_price_loss_np = v_price_loss_np / run_valid_step
        v_trace_loss_np = v_trace_loss_np / run_valid_step
        v_accuracy_np = v_accuracy_np / run_valid_step

        return v_loss_np, v_price_loss_np, v_trace_loss_np, v_accuracy_np

    # log summary
    if FLAGS.log_tensorboard_dir:
        if not tf.gfile.Exists(FLAGS.log_tensorboard_dir):
            tf.gfile.MakeDirs(FLAGS.log_tensorboard_dir)
        summary_ops = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_tensorboard_dir, sess.graph)
        fetches.append(summary_ops)

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

    total_loss = 0.
    trace_loss = 0.
    price_loss = 0.
    acc = 0.
    gnorm = 0.
    prev_step = -1
    epoch = 1
    log_str_format = ("Epoch: {:>3} [{}] | gnorm {:>4.2f} lr {:>9.3e} "
                      "| loss {:>6.5f} acc {:>.3f} | trace/price loss {:>6.5f}/{:>6.5f}       ")
    while True:
        train_name = "train"
        try:
            fetched = sess.run(fetches, feed_dict=data_loader.get_feed_dict(train_name))
        except tf.errors.OutOfRangeError:
            sess.run(data_loader.get_initializer(train_name))
            epoch += 1
            fetched = sess.run(fetches, feed_dict=data_loader.get_feed_dict(train_name))

        (loss_np, curr_step, lr_np, gnorm_np,
         price_loss_np, trace_loss_np, accuracy_np) = fetched[:-1]
        total_loss += loss_np
        trace_loss += trace_loss_np
        price_loss += price_loss_np
        acc += accuracy_np
        gnorm += gnorm_np
        log_str = log_str_format.format(
            epoch, curr_step, gnorm_np, lr_np,
            loss_np, accuracy_np, trace_loss_np, price_loss_np
        )

        if FLAGS.log_tensorboard_dir:
            if curr_step % 100 == 0:
                summary_writer.add_summary(fetched[-1], curr_step)

        print(log_str, end="\r")

        # print example
        if curr_step <= 2:
            tf.logging.info("loss_np: {}".format(loss_np))
            tf.logging.info("lr_np: {}".format(lr_np))
            tf.logging.info("price_loss_np: {}".format(price_loss_np))
            tf.logging.info("trace_loss_np: {}".format(trace_loss_np))

        # print status
        if curr_step > 0 and curr_step % FLAGS.iterations == 0:
            # Run valid data
            v_loss_np, v_price_loss_np, v_trace_loss_np, v_accuracy_np = eval_data("valid")
            mean_total_loss = total_loss / (curr_step - prev_step)
            mean_trace_loss = trace_loss / (curr_step - prev_step)
            mean_price_loss = price_loss / (curr_step - prev_step)
            mean_acc = acc / (curr_step - prev_step)
            mean_gnorm = gnorm / (curr_step - prev_step)
            log_str = ("Epoch: {:>3} [{}] | gnorm {:>4.2f} lr {:>9.3e} "
                       "| loss {:>6.5f} acc {:>.3f} | v_loss {:>6.5f} v_acc {:>.3f}"
                       "| trace/price loss {:>6.5f}/{:>6.5f} "
                       "| v_trace/price loss {:>6.5f}/{:>6.5f}  ")
            log_str = log_str.format(
                epoch, curr_step, mean_gnorm, lr_np,
                mean_total_loss, mean_acc, v_loss_np, v_accuracy_np, mean_trace_loss,
                mean_price_loss, v_trace_loss_np, v_price_loss_np)
            tf.logging.info(log_str)
            if FLAGS.info:
                tf.logging.info(FLAGS.info)
            if FLAGS.model_dir:
                log_std(log_str, os.path.join(FLAGS.model_dir, "/log.txt"))
            total_loss = 0.
            trace_loss = 0.
            price_loss = 0.
            gnorm = 0.
            acc = 0.
            prev_step = curr_step

        # save model
        if FLAGS.save_steps and FLAGS.model_dir:
            if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
                save_path = os.path.join(FLAGS.model_dir, "model_{}.ckpt".format(curr_step))
                saver.save(sess, save_path)
                tf.logging.info("Model saved in path: {}".format(save_path))

        if curr_step >= FLAGS.train_steps:
            tf.logging.info("Finished training.")
            tf.logging.info("Run eval...")
            v_loss_np, v_price_loss_np, v_trace_loss_np, v_accuracy_np = eval_data("valid")
            t_loss_np, t_price_loss_np, t_trace_loss_np, t_accuracy_np = eval_data("test")
            if (curr_step - prev_step) > 0:
                mean_total_loss = total_loss / (curr_step - prev_step)
                mean_trace_loss = trace_loss / (curr_step - prev_step)
                mean_price_loss = price_loss / (curr_step - prev_step)
                mean_acc = acc / (curr_step - prev_step)
                mean_gnorm = gnorm / (curr_step - prev_step)
            log_status_str = "Epoch: {:>3} [{}] | gnorm {:>4.2f} lr {:>9.3e}".format(
                    epoch, curr_step, gnorm, lr_np)
            log_str = ("{} : loss {:>6.5f} acc {:>.3f} | trace_loss {:>6.5f} | price_loss {:>6.5f}")
            train_log = log_str.format("Train", mean_total_loss, mean_acc,
                                       mean_trace_loss, mean_price_loss)
            valid_log = log_str.format("Valid", v_loss_np, v_accuracy_np,
                                       v_trace_loss_np, v_price_loss_np)
            test_log = log_str.format("Test ", t_loss_np, t_accuracy_np,
                                      t_trace_loss_np, t_price_loss_np)
            tf.logging.info(log_status_str)
            tf.logging.info(train_log)
            tf.logging.info(valid_log)
            tf.logging.info(test_log)

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
