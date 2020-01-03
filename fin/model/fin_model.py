import os

import tensorflow as tf

from fin.model import fin_modeling, xlnet_model, modeling, config_utils


def build_train_anml_trace_net(FLAGS, inp_values, inp_cate, price_inp_mask, trace_inp_mask):
    """
    Args:
        inp_values: np.array, (bsz, seq_len, num_value, 2), time sequence value.
        inp_cate: np.array, (bsz, seq_len)
        price_inp_mask: np.array, (bsz, seq_len, num_value), 0 for real tokens and 1 for padding.
        trace_inp_mask: np.array, (bsz, seq_len), 0 for real tokens and 1 for padding.
        price_target: np.array, (bsz, seq_len), target for price net.
        trace_target: np.array, (bsz,), target for trace net.
    """
    # XLNetConfig contains hyperparameters that are specific to a model checkpoint.
    config_dir = FLAGS.config_path
    price_xlnet_config_path = os.path.join(config_dir, "price_xlnet_config.json")
    price_xlnet_config = config_utils.XLNetConfig(json_path=price_xlnet_config_path)
    price_config_path = os.path.join(config_dir, "price_config.json")
    price_config = config_utils.PriceNetConfig(json_path=price_config_path)
    trace_xlnet_config_path = os.path.join(config_dir, "trace_xlnet_config.json")
    trace_xlnet_config = config_utils.XLNetConfig(json_path=trace_xlnet_config_path)
    # RunConfig contains hyperparameters that could be different between pretraining and finetuning.
    run_config = xlnet_model.create_run_config(
            is_training=True, is_finetune=True, FLAGS=FLAGS)

    # split seq
    inp_values_list = tf.unstack(inp_values, axis=1, num=FLAGS.seq_len)
    inp_cate_list = tf.unstack(inp_cate, axis=1, num=FLAGS.seq_len)
    price_inp_mask_list = tf.unstack(price_inp_mask, axis=1, num=FLAGS.seq_len)

    price_outputs = []
    price_summary = []
    price_values = []
    for inp, cate, mask in zip(inp_values_list, inp_cate_list, price_inp_mask_list):
        _output, _summary, value = title_net(inp, cate, mask, price_config,
                                             price_xlnet_config, run_config)
        price_outputs.append(_output)
        price_summary.append(_summary)
        price_values.append(value)

    trace_inp = tf.stack(price_summary, axis=1)
    output, summary, predict_logits = title_trace_net(
            trace_inp, trace_inp_mask, trace_xlnet_config, run_config)

    return price_summary, summary


def build_train_trace_net(FLAGS, inp_values, inp_cate, price_inp_mask, trace_inp_mask,
                          price_target, trace_target):
    """
    Args:
        inp_values: np.array, (bsz, seq_len, num_value, 2), time sequence value.
        inp_cate: np.array, (bsz, seq_len)
        price_inp_mask: np.array, (bsz, seq_len, num_value), 0 for real tokens and 1 for padding.
        trace_inp_mask: np.array, (bsz, seq_len), 0 for real tokens and 1 for padding.
        price_target: np.array, (bsz, seq_len), target for price net.
        trace_target: np.array, (bsz,), target for trace net.
    """
    # XLNetConfig contains hyperparameters that are specific to a model checkpoint.
    config_dir = FLAGS.config_path
    price_xlnet_config_path = os.path.join(config_dir, "price_xlnet_config.json")
    price_xlnet_config = config_utils.XLNetConfig(json_path=price_xlnet_config_path)
    price_config_path = os.path.join(config_dir, "price_config.json")
    price_config = config_utils.PriceNetConfig(json_path=price_config_path)
    trace_xlnet_config_path = os.path.join(config_dir, "trace_xlnet_config.json")
    trace_xlnet_config = config_utils.XLNetConfig(json_path=trace_xlnet_config_path)
    # RunConfig contains hyperparameters that could be different between pretraining and finetuning.
    run_config = xlnet_model.create_run_config(
            is_training=True, is_finetune=True, FLAGS=FLAGS)

    # split seq
    inp_values_list = tf.unstack(inp_values, axis=1, num=FLAGS.seq_len)
    inp_cate_list = tf.unstack(inp_cate, axis=1, num=FLAGS.seq_len)
    price_inp_mask_list = tf.unstack(price_inp_mask, axis=1, num=FLAGS.seq_len)

    price_outputs = []
    price_summary = []
    price_values = []
    for inp, cate, mask in zip(inp_values_list, inp_cate_list, price_inp_mask_list):
        _output, _summary, value = title_net(inp, cate, mask, price_config,
                                             price_xlnet_config, run_config)
        price_outputs.append(_output)
        price_summary.append(_summary)
        price_values.append(value)

    trace_inp = tf.stack(price_summary, axis=1)
    output, summary, predict_logits = title_trace_net(
            trace_inp, trace_inp_mask, trace_xlnet_config, run_config)

    # build loesses
    price_values_tensor = tf.concat(price_values, axis=1)
    price_loss_mask = 1 - trace_inp_mask
    price_loss = fin_modeling.price_mse_loss(
            price_values_tensor, price_target, mask=price_loss_mask)

    trace_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=predict_logits, labels=trace_target)
    trace_loss = tf.reduce_mean(trace_loss)

    return output, summary, price_loss, trace_loss, predict_logits


def build_train_title_net(FLAGS, inp_value, inp_cate, inp_mask, target):
    # XLNetConfig contains hyperparameters that are specific to a model checkpoint.
    if FLAGS.config_path:
        price_xlnet_config_path = os.path.join(FLAGS.config_path, "price_xlnet_config.json")
        price_xlnet_config = xlnet_model.XLNetConfig(json_path=price_xlnet_config_path)
        price_config_path = os.path.join(FLAGS.config_path, "price_config.json")
        price_config = config_utils.PriceNetConfig(json_path=price_config_path)
    else:
        price_xlnet_config = xlnet_model.XLNetConfig(FLAGS=FLAGS)
        price_config = config_utils.PriceNetConfig(FLAGS=FLAGS)
        if FLAGS.model_dir:
            price_xlnet_config_path = os.path.join(FLAGS.model_dir, "price_xlnet_config.json")
            price_config_path = os.path.join(FLAGS.model_dir, "price_config.json")

            price_xlnet_config.to_json(price_xlnet_config_path)
            price_config.to_json(price_config_path)

    # RunConfig contains hyperparameters that could be different between pretraining and finetuning.
    price_run_config = xlnet_model.create_run_config(
            is_training=True, is_finetune=True, FLAGS=FLAGS)

    output, summary, value = title_net(inp_value, inp_cate, inp_mask, price_config,
                                       price_xlnet_config, price_run_config)

    price_loss = fin_modeling.price_mse_loss(value, target)

    return price_loss, output, summary, value


def build_train_env_net(FLAGS, title_inp_dict, env_target):
    # Build config.
    if FLAGS.config_path:
        env_xlnet_config_path = os.path.join(FLAGS.config_path, "env_xlnet_config.json")
        price_xlnet_config_path = os.path.join(FLAGS.config_path, "price_xlnet_config.json")
        price_config_path = os.path.join(FLAGS.config_path, "price_config.json")

        env_xlnet_config = xlnet_model.XLNetConfig(json_path=env_xlnet_config_path)
        price_xlnet_config = xlnet_model.XLNetConfig(json_path=price_xlnet_config_path)
        price_config = config_utils.PriceNetConfig(json_path=price_config_path)
    else:
        env_xlnet_config = xlnet_model.XLNetConfig(FLAGS=FLAGS)
        price_xlnet_config = xlnet_model.XLNetConfig(FLAGS=FLAGS)
        price_config = config_utils.PriceNetConfig(FLAGS=FLAGS)
        if FLAGS.model_dir:
            env_xlnet_config_path = os.path.join(FLAGS.config_path, "env_xlnet_config.json")
            price_xlnet_config_path = os.path.join(FLAGS.model_dir, "price_xlnet_config.json")
            price_config_path = os.path.join(FLAGS.model_dir, "price_config.json")

            env_xlnet_config.to_json(env_xlnet_config_path)
            price_xlnet_config.to_json(price_xlnet_config_path)
            price_config.to_json(price_config_path)

    run_config = xlnet_model.create_run_config(is_training=True, is_finetune=True, FLAGS=FLAGS)
    initializer = xlnet_model._get_initializer(run_config)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        title_value = title_inp_dict["title_value"]
        title_cate = title_inp_dict["title_cate"]
        title_mask = title_inp_dict["title_mask"]
        title_target = title_inp_dict["title_target"]

        predict_title_values = []
        title_summaries = []
        for value, cate, mask in zip(title_value, title_cate, title_mask):
            output, summary, value = title_net(
                    value, cate, mask, price_config, price_xlnet_config, run_config)
            title_summaries.append(summary)
            predict_title_values.append(value)

        env_inp = tf.stack(title_summaries, axis=1)
        title_target_stack = tf.stack(title_target, axis=1)  # shape [bsz, num_title]
        title_value_stack = tf.stack(predict_title_values, axis=1)  # shape [bsz, num_title]

        title_loss = fin_modeling.price_mse_loss(title_value_stack, title_target_stack)

        summary_dict = env_net(env_inp, None, env_xlnet_config, run_config, ["mean_value"])

        predict_value_summary = summary_dict["mean_value"]

        # shape [bsz, 1]
        value = fin_modeling.postnet(
            predict_value_summary, env_xlnet_config.d_model, initializer, price_config.post_act_fn)

        mean_value_target = tf.reduce_mean(title_target_stack, axis=1, keepdims=True)

        env_loss = fin_modeling.price_mse_loss(value, mean_value_target)

    return title_loss, env_loss


def env_net(env_values, input_mask, xlnet_config, run_config, summary_names,
            summary_type="attn", scope="env_net"):
    """Process multiple title encoding.

    args:
        env_values: shape [bsz, num_value, d_model] input tensor.
        input_mask: shape [bsz, num_value].
        summary_names: list of string, scope names for summay layers.

    return:
        summaries: dictionary with summary_name as key and tensor [bsz, d_model] as value.
    """
    initializer = xlnet_model._get_initializer(run_config)
    tfm_args = dict(
            n_token=xlnet_config.n_token,
            initializer=initializer,
            attn_type="bi",
            n_layer=xlnet_config.n_layer,
            d_model=xlnet_config.d_model,
            n_head=xlnet_config.n_head,
            d_head=xlnet_config.d_head,
            d_inner=xlnet_config.d_inner,
            ff_activation=xlnet_config.ff_activation,
            untie_r=xlnet_config.untie_r,
            pre_ln=xlnet_config.pre_ln,
            pos_len=xlnet_config.pos_len,
            pre_act_fn=xlnet_config.pre_act_fn,

            is_training=run_config.is_training,
            use_bfloat16=run_config.use_bfloat16,
            use_tpu=run_config.use_tpu,
            dropout=run_config.dropout,
            dropatt=run_config.dropatt,

            mem_len=run_config.mem_len,
            reuse_len=run_config.reuse_len,
            bi_data=run_config.bi_data,
            clamp_len=run_config.clamp_len,
            same_length=run_config.same_length
    )
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        output = tf.transpose(env_values, [1, 0, 2])
        if input_mask is not None:
            input_mask = tf.transpose(input_mask, [1, 0])
        input_args = dict(
                inp_e=output,
                seg_id=None,
                input_mask=input_mask,
                mems=None,
                perm_mask=None,
                target_mapping=None,
                inp_q=None)
        tfm_args.update(input_args)

        output = fin_modeling.transformer_encoding_inp(**tfm_args)

        summary_dict = {}

        for summary_name in summary_names:

            summary = modeling.summarize_sequence(
                    summary_type=summary_type,
                    hidden=output,
                    d_model=xlnet_config.d_model,
                    n_head=xlnet_config.n_head,
                    d_head=xlnet_config.d_head,
                    dropout=run_config.dropout,
                    dropatt=run_config.dropatt,
                    is_training=run_config.is_training,
                    input_mask=input_mask,
                    initializer=initializer,
                    use_proj=True,
                    scope=summary_name)

            summary_dict[summary_name] = summary

    return summary_dict


def title_net(inp_value, inp_cate, input_mask, price_config, xlnet_config, run_config,
              summary_type="attn", scope="title_net"):
    """Process single title property into logits.

    args:
        inp_value: shape [bsz, num_value] input tensor.
        inp_cate: shape [bsz].
        input_mask: shape [bsz, num_value].

    return: shape [bsz, num_value, d_model]
    """

    initializer = xlnet_model._get_initializer(run_config)
    tfm_args = dict(
            n_token=xlnet_config.n_token,
            initializer=initializer,
            attn_type="bi",
            n_layer=xlnet_config.n_layer,
            d_model=xlnet_config.d_model,
            n_head=xlnet_config.n_head,
            d_head=xlnet_config.d_head,
            d_inner=xlnet_config.d_inner,
            ff_activation=xlnet_config.ff_activation,
            untie_r=xlnet_config.untie_r,
            pre_ln=xlnet_config.pre_ln,
            pos_len=xlnet_config.pos_len,

            is_training=run_config.is_training,
            use_bfloat16=run_config.use_bfloat16,
            use_tpu=run_config.use_tpu,
            dropout=run_config.dropout,
            dropatt=run_config.dropatt,

            mem_len=run_config.mem_len,
            reuse_len=run_config.reuse_len,
            bi_data=run_config.bi_data,
            clamp_len=run_config.clamp_len,
            same_length=run_config.same_length
    )

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        output = fin_modeling.prenet(
            inp_value, tfm_args["d_model"], tfm_args["initializer"],
            act_fn=price_config.pre_act_fn)

        if inp_cate is not None:
            cate_output = fin_modeling.category_net(
                inp_cate, tfm_args["d_model"], price_config.category_num, tfm_args["initializer"])
            cate_output = cate_output[:, None, :]
            output = tf.concat([output, cate_output], axis=1)

        output = tf.transpose(output, [1, 0, 2])
        if input_mask is not None:
            input_mask = tf.transpose(input_mask, [1, 0])
        input_args = dict(
                inp_e=output,
                seg_id=None,
                input_mask=input_mask,
                mems=None)
        tfm_args.update(input_args)

        output = fin_modeling.transformer_encoding_inp(**tfm_args)

        summary = modeling.summarize_sequence(
                summary_type=summary_type,
                hidden=output,
                d_model=xlnet_config.d_model,
                n_head=xlnet_config.n_head,
                d_head=xlnet_config.d_head,
                dropout=run_config.dropout,
                dropatt=run_config.dropatt,
                is_training=run_config.is_training,
                input_mask=input_mask,
                initializer=initializer,
                use_proj=True)

        value = fin_modeling.postnet(
            summary, tfm_args["d_model"], initializer, price_config.post_act_fn)

    output = tf.transpose(output, [1, 0, 2])

    return output, summary, value


def title_trace_net(inp_value, input_mask, xlnet_config, run_config,
                    summary_type="attn", scope="trace_net"):
    """
        Process multiple title logits in sequences.

    args:
        inp_value: shape [bsz, seq_len, d_model] input tensor.
        input_mask: shape [bsz, seq_len]. 0 for real tokens and 1 for padding.

    return: shape [bsz, seq_len, d_model]
    """

    initializer = xlnet_model._get_initializer(run_config)
    tfm_args = dict(
            n_token=xlnet_config.n_token,
            initializer=initializer,
            attn_type="uni",
            n_layer=xlnet_config.n_layer,
            d_model=xlnet_config.d_model,
            n_head=xlnet_config.n_head,
            d_head=xlnet_config.d_head,
            d_inner=xlnet_config.d_inner,
            ff_activation=xlnet_config.ff_activation,
            untie_r=xlnet_config.untie_r,
            pre_ln=xlnet_config.pre_ln,
            pos_len=xlnet_config.pos_len,

            is_training=run_config.is_training,
            use_bfloat16=run_config.use_bfloat16,
            use_tpu=run_config.use_tpu,
            dropout=run_config.dropout,
            dropatt=run_config.dropatt,

            mem_len=run_config.mem_len,
            reuse_len=run_config.reuse_len,
            bi_data=run_config.bi_data,
            clamp_len=run_config.clamp_len,
            same_length=run_config.same_length
    )

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        inp = tf.transpose(inp_value, [1, 0, 2])
        if input_mask is not None:
            input_mask = tf.transpose(input_mask, [1, 0])
        input_args = dict(
                inp_e=inp,
                seg_id=None,
                input_mask=input_mask,
                mems=None)
        tfm_args.update(input_args)

        output = fin_modeling.transformer_xl_encoding_inp(**tfm_args)

        summary = modeling.summarize_sequence(
                summary_type=summary_type,
                hidden=output,
                d_model=xlnet_config.d_model,
                n_head=xlnet_config.n_head,
                d_head=xlnet_config.d_head,
                dropout=run_config.dropout,
                dropatt=run_config.dropatt,
                is_training=run_config.is_training,
                input_mask=input_mask,
                initializer=initializer,
                use_proj=True)

        predict_logits = fin_modeling.postnet(
            summary, tfm_args["d_model"], initializer, None)

        predict_logits = tf.squeeze(predict_logits, axis=[1])

    output = tf.transpose(output, [1, 0, 2])

    return output, summary, predict_logits
