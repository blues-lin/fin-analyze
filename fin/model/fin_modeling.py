import tensorflow as tf

from fin.model import modeling


def _create_mask(qlen, mlen, dtype=tf.float32, same_length=False):
    """create causal attention mask."""
    attn_mask = tf.ones([qlen, qlen], dtype=dtype)
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

    return ret


def price_mse_loss(x, y, mask=None):
    """mask: 0 for pad, 1 for real value."""
    squ = tf.square(x - y)
    sqr = tf.sqrt(squ)
    if mask is not None:
        sqr = sqr*mask
        _sum = tf.reduce_sum(sqr, axis=1)
        valid_value_num = tf.reduce_sum(mask, axis=1)
        loss = _sum / valid_value_num
        loss = tf.reduce_mean(loss)
    else:
        loss = tf.reduce_mean(sqr)

    return loss


def postnet(logits, d_model, initializer, act_fn=None, scope="postnet", reuse=None):
    """Process logits into single value.

    args:
        logits: shape [bsz, d_model] input tensor.
        d_model: the hidden size.

    return: shape [bsz].
    """
    with tf.variable_scope(scope, reuse=reuse):
        act_w = tf.get_variable('act_weight', [d_model, d_model],
                                dtype=tf.float32, initializer=initializer)
        act_b = tf.get_variable('act_bias', [d_model],
                                dtype=tf.float32, initializer=initializer)

        logits = tf.einsum('bd,dv->bv', logits, act_w) + act_b

        if act_fn == "tanh":
            logits = tf.tanh(logits)
        elif act_fn == "gelu":
            logits = modeling.gelu(logits)
        else:
            logits = logits

        w = tf.get_variable('weight', [d_model, 1],
                            dtype=tf.float32, initializer=initializer)
        b = tf.get_variable('bias', [1],
                            dtype=tf.float32, initializer=initializer)

        value = tf.einsum('bd,dv->bv', logits, w) + b

    return value


def prenet(tensor_in, dim, initializer, act_fn=None, scope="prenet", reuse=None):
    """Process single value into multi dim vector.

    args:
        tensor_in: shape [bsz, num_value, 2] input tensor with negtive sign.
        dim: the size of vector.

    return: shape [bsz, num_value, dim]
    """
    with tf.variable_scope(scope, reuse=reuse):
        l1_w = tf.get_variable('prenet_1', [2, dim],
                               dtype=tf.float32, initializer=initializer)
        l1_b = tf.get_variable('prenet_1_bias', [dim],
                               dtype=tf.float32, initializer=initializer)
        l2_w = tf.get_variable('prenet_2', [dim, dim],
                               dtype=tf.float32, initializer=initializer)
        l2_b = tf.get_variable('prenet_2_bias', [dim],
                               dtype=tf.float32, initializer=initializer)

        logits = tf.einsum('bvn,nd->bvd', tensor_in, l1_w) + l1_b
        logits = tf.einsum('bvd,di->bvi', logits, l2_w) + l2_b
        if act_fn == "tanh":
            output = tf.tanh(logits)
        elif act_fn == "gelu":
            output = modeling.gelu(logits)
        else:
            output = logits

    return output


def category_net(tensor_in, emb_dim, num_cate, initializer, scope="catenet", reuse=None):
    """Process category data into dim vector.

    args:
        tensor_in: int shape [bsz] input tensor.
        emb_dim: the size of vector.
        num_cate: number of category.

    return: shape [bsz, emb_dim]
    """
    with tf.variable_scope(scope, reuse=reuse):
        emb_martix = tf.get_variable('emb_m', [num_cate, emb_dim],
                                     dtype=tf.float32, initializer=initializer)

        return tf.nn.embedding_lookup(emb_martix, tensor_in)


def transformer_xl_encoding_inp(inp_e, n_layer, d_model, n_head,
                                d_head, d_inner, dropout, dropatt, attn_type,
                                initializer, is_training,
                                same_length=False, clamp_len=-1, untie_r=False,
                                input_mask=None, seg_id=None, reuse_len=None,
                                ff_activation='gelu',
                                use_bfloat16=False, scope='transformer',
                                pre_ln=False, **kwargs):
    """
        With relative_positional_encoding.
        Defines a Transformer-XL computation graph with additional
        support for XLNet.

        Args:

        inp_k: int32 Tensor in shape [len, bsz], the input token IDs.
        seg_id: int32 Tensor in shape [len, bsz], the input segment IDs.
        input_mask: float32 Tensor in shape [len, bsz], the input mask.
            0 for real tokens and 1 for padding.
        mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
            from previous batches. The length of the list equals n_layer.
            If None, no memory is used.
        perm_mask: float32 Tensor in shape [len, len, bsz].
            If perm_mask[i, j, k] = 0, i attend to j in batch k;
            if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
            If None, each position attends to all the others.
        target_mapping: float32 Tensor in shape [num_predict, len, bsz].
            If target_mapping[i, j, k] = 1, the i-th predict in batch k is
            on the j-th token.
            Only used during pretraining for partial prediction.
            Set to None during finetuning.
        inp_q: float32 Tensor in shape [len, bsz].
            1 for tokens with losses and 0 for tokens without losses.
            Only used during pretraining for two-stream attention.
            Set to None during finetuning.

        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        ff_activation: str, "relu" or "gelu".
        untie_r: bool, whether to untie the biases in attention.
        n_token: int, the vocab size.

        is_training: bool, whether in training mode.
        use_tpu: bool, whether TPUs are used.
        use_bfloat16: bool, use bfloat16 instead of float32.
        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
        summary_type: str, "last", "first", "mean", or "attn". The method
            to pool the input to get a vector representation.
        initializer: A tf initializer.
        scope: scope name for the computation graph.
    """
    tf_float = tf.bfloat16 if use_bfloat16 else tf.float32

    with tf.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
        else:
            r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                       dtype=tf_float, initializer=initializer)

        bsz = tf.shape(inp_e)[1]
        qlen = tf.shape(inp_e)[0]
        mlen = 0
        klen = mlen + qlen

        # Attention mask
        # causal attention mask
        if attn_type == 'uni':
            attn_mask = _create_mask(qlen, mlen, tf_float, same_length)
            attn_mask = attn_mask[:, :, None, None]
        elif attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(attn_type))

        # data mask: input mask
        if input_mask is not None:
            data_mask = input_mask[None]
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=tf_float)

        if attn_mask is not None:
            non_tgt_mask = -tf.eye(qlen, dtype=tf_float)
            non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=tf_float),
                                     non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                                   dtype=tf_float)
        else:
            non_tgt_mask = None

        output_h = inp_e

        # Segment embedding
        seg_mat = None

        # Positional encoding
        pos_emb = modeling.relative_positional_encoding(
                qlen, klen, d_model, clamp_len, attn_type, False,
                bsz=bsz, dtype=tf_float)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

        # Attention layers
        mems = [None] * n_layer

        for i in range(n_layer):
            # not use seg
            # segment bias
            r_s_bias_i = None
            seg_embed_i = None

            with tf.variable_scope('layer_{}'.format(i)):
                if pre_ln:
                    output_h = tf.contrib.layers.layer_norm(
                        output_h, begin_norm_axis=-1, scope='LayerNorm')

                output_h = modeling.rel_multihead_attn(
                        h=output_h,
                        r=pos_emb,
                        r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                        r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                        seg_mat=seg_mat,
                        r_s_bias=r_s_bias_i,
                        seg_embed=seg_embed_i,
                        attn_mask=non_tgt_mask,
                        mems=mems[i],
                        d_model=d_model,
                        n_head=n_head,
                        d_head=d_head,
                        dropout=dropout,
                        dropatt=dropatt,
                        is_training=is_training,
                        kernel_initializer=initializer,
                        reuse=None,
                        pre_ln=pre_ln)

                output_h = modeling.positionwise_ffn(
                        inp=output_h,
                        d_model=d_model,
                        d_inner=d_inner,
                        dropout=dropout,
                        kernel_initializer=initializer,
                        activation_type=ff_activation,
                        is_training=is_training,
                        reuse=None,
                        pre_ln=pre_ln)

        output = tf.layers.dropout(output_h, dropout, training=is_training)
        if pre_ln:
            output_h = tf.contrib.layers.layer_norm(
                output_h, begin_norm_axis=-1, scope='LayerNorm')

        return output


def transformer_encoding_inp(inp_e, n_token, n_layer, d_model, n_head,
                             d_head, d_inner, dropout, dropatt, attn_type,
                             bi_data, initializer, is_training,
                             pos_len=None, mem_len=None, mems=None,
                             same_length=False, clamp_len=-1, untie_r=False,
                             use_tpu=True, input_mask=None,
                             seg_id=None, reuse_len=None,
                             ff_activation='gelu',
                             use_bfloat16=False, scope='transformer',
                             pre_ln=False, **kwargs):
    """
        Without relative_positional_encoding.
        Defines a Transformer-XL computation graph with additional
        support for XLNet.

        Args:

        inp_e: float32 Tensor in shape [len, bsz, d_model], the input vector.
        seg_id: int32 Tensor in shape [len, bsz], the input segment IDs.
        input_mask: float32 Tensor in shape [len, bsz], the input mask.
            0 for real tokens and 1 for padding.
        mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
            from previous batches. The length of the list equals n_layer.
            If None, no memory is used.
        perm_mask: float32 Tensor in shape [len, len, bsz].
            If perm_mask[i, j, k] = 0, i attend to j in batch k;
            if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
            If None, each position attends to all the others.
        target_mapping: float32 Tensor in shape [num_predict, len, bsz].
            If target_mapping[i, j, k] = 1, the i-th predict in batch k is
            on the j-th token.
            Only used during pretraining for partial prediction.
            Set to None during finetuning.
        inp_q: float32 Tensor in shape [len, bsz].
            1 for tokens with losses and 0 for tokens without losses.
            Only used during pretraining for two-stream attention.
            Set to None during finetuning.

        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        ff_activation: str, "relu" or "gelu".
        untie_r: bool, whether to untie the biases in attention.
        n_token: int, the vocab size.

        is_training: bool, whether in training mode.
        use_tpu: bool, whether TPUs are used.
        use_bfloat16: bool, use bfloat16 instead of float32.
        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.
        init: str, the initialization scheme, either "normal" or "uniform".
        init_range: float, initialize the parameters with a uniform distribution
            in [-init_range, init_range]. Only effective when init="uniform".
        init_std: float, initialize the parameters with a normal distribution
            with mean 0 and stddev init_std. Only effective when init="normal".
        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
            and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
            Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
            -1 means no clamping.
        same_length: bool, whether to use the same attention length for each token.
        summary_type: str, "last", "first", "mean", or "attn". The method
            to pool the input to get a vector representation.
        initializer: A tf initializer.
        scope: scope name for the computation graph.
        pre_ln: bool, whether use pre-layer_norm.
    """
    tf_float = tf.bfloat16 if use_bfloat16 else tf.float32

    with tf.variable_scope(scope):
        qlen = tf.shape(inp_e)[0]

        attn_mask = None

        # data mask: input mask
        if input_mask is None:
            data_mask = None
        else:
            data_mask = input_mask[None]
        if data_mask is not None:
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=tf_float)

        # Position embedding
        if pos_len:
            pos_int = tf.range(qlen, dtype=tf.int32)
            pos_int = pos_int[:, None]
            pos_emb, lookup_table = modeling.embedding_lookup(
                    x=pos_int,
                    n_token=pos_len,
                    d_embed=d_model,
                    initializer=initializer,
                    use_tpu=use_tpu,
                    dtype=tf_float,
                    scope='pos_embedding')
            pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

            output_h = inp_e + pos_emb
        else:
            output_h = inp_e

        # Attention layers
        for i in range(n_layer):
            with tf.variable_scope('layer_{}'.format(i)):
                if pre_ln:
                    output_h = tf.contrib.layers.layer_norm(
                        output_h, begin_norm_axis=-1, scope='LayerNorm')

                output_h = modeling.multihead_attn(
                        output_h, output_h, output_h,
                        attn_mask, d_model, n_head, d_head, dropout,
                        dropatt, is_training, initializer, residual=True,
                        scope='abs_attn', pre_ln=pre_ln, reuse=None)

                output_h = modeling.positionwise_ffn(
                        inp=output_h,
                        d_model=d_model,
                        d_inner=d_inner,
                        dropout=dropout,
                        kernel_initializer=initializer,
                        activation_type=ff_activation,
                        is_training=is_training,
                        reuse=None,
                        pre_ln=pre_ln)

        output_h = tf.layers.dropout(output_h, dropout, training=is_training)
        if pre_ln:
            output_h = tf.contrib.layers.layer_norm(output_h, begin_norm_axis=-1,
                                                    scope='LayerNorm_output')

        return output_h
