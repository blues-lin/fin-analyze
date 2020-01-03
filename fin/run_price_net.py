import os
import math
import random
import logging
from pprint import pformat

from absl import flags
import numpy as np
import tensorflow as tf

from fin.model import model_utils
from fin.model import fin_model
from fin.gpu_utils import assign_to_gpu, average_grads_and_vars
from fin.data_utils import base_data_loader, data_handler


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
flags.DEFINE_string("config_path", default=None,
                    help="config path for config files.")
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
flags.DEFINE_integer("num_predict", default=32,
                     help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer('perm_size', default=64,
                     help='perm size.')
flags.DEFINE_integer("n_token", 15000, help="Vocab size")

# Price net
flags.DEFINE_string("pre_act_fn", None,
                    help="Activate function in pre_net.")
flags.DEFINE_string("post_act_fn", None,
                    help="Activate function in post_net.")
flags.DEFINE_integer("category_num", 72,
                     help="Number of categories in industry id.")

# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_string("usermodel_config_path", default=None,
                    help="Usermodel config path.")
flags.DEFINE_string("summary_type", default="attn",
                    help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")
flags.DEFINE_bool("pre_ln", True,
                  help="Whether to use Pre-LN attention net.")
flags.DEFINE_integer("mem_len", default=128,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_integer("n_layer", default=12,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=128,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=128,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=8,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=64,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=512,
                     help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_integer("pos_len", default=256,
                     help="Number of maximum input values.")
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
flags.DEFINE_integer("train_steps", default=1000000,
                     help="Number of training steps")
flags.DEFINE_integer("iterations", default=1000,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=None,
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
        inp_value, inp_cate, inp_mask, target = tensors

        price_loss, output, summary, value = fin_model.build_train_title_net(
                FLAGS, inp_value, inp_cate, inp_mask, target)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # GPU
        assert is_training
        total_loss = price_loss
        all_vars = tf.trainable_variables()
        grads = tf.gradients(total_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))

        return (price_loss, grads_and_vars, value)

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
    data_filepath = os.path.join(prefix, "indust_price_combined_data_20200110.csv")
    industry_filepath = os.path.join(prefix, "industry_cate.txt")

    value_keys = data_handler.load_value_keys(value_keys_filepath)
    data, max_value, max_price = data_handler.load_csv_data(data_filepath, value_keys)
    industry_mapping = data_handler.load_industry_cate(industry_filepath)

    # Split train and valid data
    random.seed(777)
    valid_precent = 0.05  # 5% valid data
    total_titles = list(data.keys())
    total_len = len(total_titles)
    valid_len = int(total_len * valid_precent)

    random.shuffle(total_titles)
    valid_titles = total_titles[:valid_len]
    train_titles = total_titles[valid_len:]

    valid_data = {}
    train_data = {}
    for title in valid_titles:
        valid_data[title] = data[title]
    for title in train_titles:
        train_data[title] = data[title]

    # Build train and valid dataset
    train_gen_fn = data_handler.get_generator_fn(
            train_data, max_value, max_price, industry_mapping)
    valid_gen_fn = data_handler.get_generator_fn(
            valid_data, max_value, max_price, industry_mapping)
    train_dataset = data_handler.get_dataset(train_gen_fn)
    valid_dataset = data_handler.get_dataset(valid_gen_fn)

    data_loader = base_data_loader.BaseDataLoader(graph, sess, None)
    data_loader.add_datasets({"train": train_dataset}, batch_size=FLAGS.batch_size)
    data_loader.add_datasets({"valid": valid_dataset}, batch_size=FLAGS.batch_size)

    tensors = data_loader.input_tensors

    split_num = FLAGS.num_core_per_host if FLAGS.num_core_per_host else 1
    split_tensors = [tf.split(tensor, split_num, 0) for tensor in tensors]

    tower_losses, tower_grads_and_vars = [], []
    tower_values = []
    # Build multi gpu graph
    if FLAGS.num_core_per_host == 0:
        i = 0
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            inp_tensors = [ten[i] for ten in split_tensors]
            model_ret = single_core_graph(inp_tensors)

            (price_loss_i, grads_and_vars_i, value_i) = model_ret

            tower_losses.append(price_loss_i)
            tower_grads_and_vars.append(grads_and_vars_i)
            tower_values.append(value_i)

    for i in range(FLAGS.num_core_per_host):
        if i == 0:
            reuse = None
        else:
            reuse = True
        with tf.device(assign_to_gpu(i, "/gpu:0")), \
                tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            inp_tensors = [ten[i] for ten in split_tensors]
            model_ret = single_core_graph(inp_tensors)

            (price_loss_i, grads_and_vars_i, value_i) = model_ret

            tower_losses.append(price_loss_i)
            tower_grads_and_vars.append(grads_and_vars_i)
            tower_values.append(value_i)

    # average losses and gradients across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]

    # Log gradients
    if FLAGS.log_tensorboard_dir:
        if FLAGS.log_gradient:
            tf.logging.info("Log gradients:")
            for grad, var in grads_and_vars:
                var_name = var.name.split(":")[0]
                tf.logging.info("    Variable name: {}".format(var_name))
                tf.summary.histogram(var.name, grad)

    # concat value outputs
    batch_value = tf.concat(tower_values, axis=0)

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

    # Start training loop.
    fetches = [loss, global_step, learning_rate, gnorm,
               batch_value, train_op]

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

    total_loss = 0.
    prev_step = -1
    epoch = 1
    log_str_format = ("Epoch: {:>3} [{}] | gnorm {:>4.2f} lr {:>9.3e} "
                      "| loss {:>6.6f} | value diff {:>5.1f}       ")
    while True:
        train_name = "train"
        try:
            fetched = sess.run(fetches, feed_dict=data_loader.get_feed_dict(train_name))
        except tf.errors.OutOfRangeError:
            sess.run(data_loader.get_initializer(train_name))
            epoch += 1
            fetched = sess.run(fetches, feed_dict=data_loader.get_feed_dict(train_name))
        (loss_np, curr_step, lr_np, gnorm_np,
         batch_value_np) = fetched[:-1]
        total_loss += loss_np
        log_str = log_str_format.format(
            epoch, curr_step, gnorm_np, lr_np,
            loss_np, loss_np*max_price
        )

        if FLAGS.log_tensorboard_dir:
            if curr_step % 100 == 0:
                summary_writer.add_summary(fetched[-1], curr_step)

        print(log_str, end="\r")

        # print example
        if curr_step <= 2:
            tf.logging.info("loss_np: {}".format(loss_np))
            tf.logging.info("lr_np: {}".format(lr_np))
            tf.logging.info("value_np: {}".format(batch_value_np))

        # print status
        if curr_step > 0 and curr_step % FLAGS.iterations == 0:
            # Run valid data
            try:
                fetched = sess.run(fetches[:-1], feed_dict=data_loader.get_feed_dict("valid"))
            except tf.errors.OutOfRangeError:
                sess.run(data_loader.get_initializer("valid"))
                fetched = sess.run(fetches[:-1], feed_dict=data_loader.get_feed_dict("valid"))
            v_loss_np = fetched[0]
            curr_loss = total_loss / (curr_step - prev_step)
            log_str = ("Epoch: {:>3} [{}] | gnorm {:>4.2f} lr {:>9.3e} "
                       "| loss {:>6.6f}/{:>5.1f} | v_loss {:>6.6f}/{:>5.1f}       ")
            log_str = log_str.format(
                epoch, curr_step, gnorm_np, lr_np,
                curr_loss, curr_loss*max_price, v_loss_np, v_loss_np*max_price)
            tf.logging.info(log_str)
            if FLAGS.model_dir:
                log_std(log_str, FLAGS.model_dir + "/log.txt")
            total_loss = 0.
            prev_step = curr_step

        # save model
        if FLAGS.save_steps:
            if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
                save_path = os.path.join(FLAGS.model_dir, "model_{}.ckpt".format(curr_step))
                saver.save(sess, save_path)
                tf.logging.info("Model saved in path: {}".format(save_path))

        if curr_step >= FLAGS.train_steps:
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
    if FLAGS.config_path:
        FLAGS.model_config_path = os.path.join(FLAGS.config_path, "config.json")
    if FLAGS.init_checkpoint:
        FLAGS.model_config_path = os.path.join(FLAGS.init_checkpoint, "config.json")

    train()


if __name__ == "__main__":
    # Add time stamp to logging.
    logger = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    # logger.handlers[0].setFormatter(formatter)
    tf.app.run()
