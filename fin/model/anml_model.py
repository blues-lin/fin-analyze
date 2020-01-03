import tensorflow as tf

from fin.model import reptile_utils


"""A Neuromodulated Meta-Learning algorithm (ANML) https://arxiv.org/pdf/2002.09571.pdf"""


class ANMLModel:

    def __init__(self, session, nm_name, p_name, nm_logits, p_logits, initializer):
        self.session = session
        self.nm_name = nm_name
        self.p_name = p_name
        self.nm_logits = nm_logits
        self.p_logits = p_logits
        self.nm_vars = tf.trainable_variables(nm_name)
        self.p_state = reptile_utils.VariableState(
            session, tf.trainable_variables(p_name))
        self.final_vars = []

    def cache_p_values(self):
        self.p_values = self.p_state.export_variables()

    def restore_p_values(self):
        self.p_state.import_variables(self.p_values)

    def init_final_vars(self):
        self.session.run(self.final_var_init_op)

    def build_predict_tensor(self, output_dim):
        with tf.variable_scope(self.p_name):
            gate = tf.nn.sigmoid(self.nm_logits)
            predict_logits = self.p_logits * gate

            logits_dim = self.p_logits.get_shape().as_list()[1]

            w = tf.get_variable("predict_w_{}".format(
                output_dim), [logits_dim, output_dim])
            b = tf.get_variable("predict_b_{}".format(
                output_dim), [output_dim])

            self.final_var_init_op = tf.variables_initializer([w, b])

            logits = tf.einsum('bd,dv->bv', predict_logits, w) + b

        return logits

    def build_inner_loop_op(self, loss, learning_rate):
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        p_vars = tf.trainable_variables(self.p_name)
        grads = tf.gradients(loss, p_vars)

        clipped, gnorm = tf.clip_by_global_norm(grads, 1.0)

        train_op = optimizer.apply_gradients(zip(clipped, p_vars))

        return train_op, learning_rate, gnorm

    def build_outer_loop_op(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        p_vars = tf.trainable_variables(self.p_name)
        nm_vars = tf.trainable_variables(self.nm_name)
        vars = p_vars + nm_vars

        grads = tf.gradients(loss, vars)
        clipped, gnorm = tf.clip_by_global_norm(grads, 1.0)

        train_op = optimizer.apply_gradients(zip(clipped, vars))

        return train_op, learning_rate, gnorm
