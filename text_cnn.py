import tensorflow as tf
import numpy as np

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length_arg1, sequence_length_arg2, num_classes, pos_vocab_size, word_vocab_size,
      pos_embedding_size, word_embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x_arg1_word = tf.placeholder(tf.int32, [None, sequence_length_arg1], name="input_x_arg1_word")
        self.input_x_arg2_word = tf.placeholder(tf.int32, [None, sequence_length_arg2], name="input_x_arg2_word")
        self.input_x_arg1_pos = tf.placeholder(tf.int32, [None, sequence_length_arg1], name="input_x_arg1_pos")
        self.input_x_arg2_pos = tf.placeholder(tf.int32, [None, sequence_length_arg2], name="input_x_arg2_pos")

        self.input_y = tf.placeholder(tf.int64, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        embedding_size = pos_embedding_size + word_embedding_size
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self._W_emb_pos = _variable_on_cpu(
                name='pos_embedding',
                shape=[pos_vocab_size, pos_embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
            )
            self.embedded_chars_arg1_pos = tf.nn.embedding_lookup(self._W_emb_pos, self.input_x_arg1_pos)
            self.embedded_chars_arg2_pos = tf.nn.embedding_lookup(self._W_emb_pos, self.input_x_arg2_pos)
            # self.embedded_chars_expanded_arg1_pos = tf.expand_dims(self.embedded_chars_arg1_pos, -1)
            # self.embedded_chars_expanded_arg2_pos = tf.expand_dims(self.embedded_chars_arg2_pos, -1)

            self._W_emb_word = _variable_on_cpu(name='word_embedding', shape=[word_vocab_size, word_embedding_size],  initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            self.embedded_chars_arg1_word = tf.nn.embedding_lookup(self._W_emb_word, self.input_x_arg1_word)
            self.embedded_chars_arg2_word = tf.nn.embedding_lookup(self._W_emb_word, self.input_x_arg2_word)

            # self.embedded_chars_expanded_arg1_word = tf.expand_dims(self.embedded_chars_arg1_word, -1)
            # self.embedded_chars_expanded_arg2_word = tf.expand_dims(self.embedded_chars_arg2_word, -1)

            self.embedded_chars_arg1 = tf.concat(
                [self.embedded_chars_arg1_word, self.embedded_chars_arg1_pos],2)
            self.embedded_chars_arg2 = tf.concat(
                [self.embedded_chars_arg2_word, self.embedded_chars_arg2_pos], 2)
            print(self.embedded_chars_arg1.shape)
            self.embedded_chars_expanded_arg1 = tf.expand_dims(self.embedded_chars_arg1, -1)
            self.embedded_chars_expanded_arg2 = tf.expand_dims(self.embedded_chars_arg2, -1)
            print(self.embedded_chars_expanded_arg1.shape)


        # Create a convolution + maxpool layer for each filter size
            #print(self.embedded_chars_expanded_arg1_pos.shape)
            #print(self.embedded_chars_expanded_arg1_word.shape)

            #print(self.embedded_chars_expanded_arg1.shape)






        pooled_outputs_arg1 = []
        pooled_outputs_arg2 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.random_uniform( shape=[num_filters]), name="b")
                conv_arg1 = tf.nn.conv2d(
                    self.embedded_chars_expanded_arg1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_arg1")

                conv_arg2 = tf.nn.conv2d(
                    self.embedded_chars_expanded_arg2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_arg2")






                # Apply nonlinearity
                h_arg1 = tf.nn.relu(tf.nn.bias_add(conv_arg1, b), name="relu_arg1")
                h_arg2 = tf.nn.relu(tf.nn.bias_add(conv_arg2, b), name="relu_arg2")
                # h_arg1 = tf.nn.tanh(conv1, name="relu_arg1")
                # h_arg2 = tf.nn.tanh(conv2, name="relu_arg2")

                # Maxpooling over the outputs
                pooled_arg1 = tf.nn.max_pool(
                    h_arg1,
                    ksize=[1, sequence_length_arg1 - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_arg1")

                pooled_arg2 = tf.nn.max_pool(
                    h_arg2,
                    ksize=[1, sequence_length_arg2 - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_arg2")


                pooled_outputs_arg1.append(pooled_arg1)
                pooled_outputs_arg2.append(pooled_arg2)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_arg1 = tf.concat(pooled_outputs_arg1, 3)
        self.h_pool_arg2 = tf.concat(pooled_outputs_arg2, 3)

        self.h_pool_flat_arg1 = tf.reshape(self.h_pool_arg1, [-1, num_filters_total])
        self.h_pool_flat_arg2 = tf.reshape(self.h_pool_arg2, [-1, num_filters_total])

        self.h_pool_flat = tf.concat([self.h_pool_flat_arg1, self.h_pool_flat_arg2], axis=1)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)






        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[2 * num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_uniform(shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b,name="scores")#dropout changed
            self.soft_socres = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.soft_socres, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

            self.loss = tf.add(tf.reduce_mean(losses),l2_reg_lambda * l2_loss,name="losses")

            #self.predictions = tf.argmax(self.input_y, 1)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            all_positive_indexes = tf.where(self.predictions > 0)
            all_positive_indexes = tf.reshape(all_positive_indexes, [-1])

            positive_pred = tf.nn.embedding_lookup(self.predictions, all_positive_indexes)
            positive_label = tf.nn.embedding_lookup(tf.argmax(self.input_y, 1), all_positive_indexes)



            correct_positive_predictions = tf.equal(positive_pred, positive_label)

            self.precision = tf.reduce_mean(tf.cast(correct_positive_predictions, dtype=tf.float32), name="precision")

            tp = tf.reduce_sum(tf.cast(correct_positive_predictions, dtype=tf.float32))

            self.label_index = tf.cast(tf.argmax(self.input_y, 1), dtype=tf.int32)


            tpfn = tf.reduce_sum(tf.where(self.label_index > 0, tf.ones_like(self.label_index, dtype=tf.float32), tf.zeros_like(self.label_index, dtype=tf.float32)))

            # tpfn = tf.cast(tpfn, dtype=tf.float32)

            self.recall = tf.truediv(tp, tpfn, name='recall')

    def assign_embedding(self, session, pretrained_pos_embedding, pretrained_word_embedding):
        session.run(tf.assign(self._W_emb_pos, pretrained_pos_embedding))
        session.run(tf.assign(self._W_emb_word, pretrained_word_embedding))