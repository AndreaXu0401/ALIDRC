#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import gensim
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 512, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints",10 , "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("use_pretrain",True,"use pretrained embedding")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer("max_document_length",100,"Max document length")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================
#load pos file
x_text_train_arg1_pos,x_text_train_arg2_pos,y_train = data_helpers.load_pos_pkl('train_pos_file')
x_text_dev_arg1_pos, x_text_dev_arg2_pos, y_dev = data_helpers.load_pos_pkl('dev_pos_file')

#load word file
x_text_train_arg1_word, x_text_train_arg2_word, y_train = data_helpers.load_word_pkl('train_word_file')
x_text_dev_arg1_word, x_text_dev_arg2_word, y_dev = data_helpers.load_word_pkl('dev_word_file')


pos_vocab,pos_embd = data_helpers.build_pos_vocab_embd('fourway_data_wu_NOTUNK/pos_list.pkl')
word_vocab,word_embd = data_helpers.build_word_vocab_embd('fourway_data_wu_NOTUNK/word_list.pkl')


# file_pkl = open("./pos_vocab_embd_word_vocab_embd.pkl", "wb")
# pickle.dump([pos_vocab,pos_embd,word_vocab,word_embd],file_pkl)
# file_pkl.close()


x_train_arg1_pos = data_helpers.build_input_data(x_text_train_arg1_pos,pos_vocab,FLAGS.max_document_length)
x_train_arg2_pos = data_helpers.build_input_data(x_text_train_arg2_pos,pos_vocab,FLAGS.max_document_length)
x_dev_arg1_pos = data_helpers.build_input_data(x_text_dev_arg1_pos,pos_vocab,FLAGS.max_document_length)
x_dev_arg2_pos = data_helpers.build_input_data(x_text_dev_arg2_pos,pos_vocab,FLAGS.max_document_length)

x_train_arg1_word = data_helpers.build_input_data(x_text_train_arg1_word, word_vocab,FLAGS.max_document_length)
x_train_arg2_word = data_helpers.build_input_data(x_text_train_arg2_word, word_vocab,FLAGS.max_document_length)
x_dev_arg1_word = data_helpers.build_input_data(x_text_dev_arg1_word,word_vocab,FLAGS.max_document_length)
x_dev_arg2_word = data_helpers.build_input_data(x_text_dev_arg2_word,word_vocab,FLAGS.max_document_length)


with open("./vocab_embd.txt","w",encoding="utf-8") as write_object:
    for v in word_vocab:
        write_object.write(str(v)+"\n")

with open("./x_train_arg1.txt", "w") as write_object:
    for x in x_train_arg1_word:
        write_object.write(str(x) + "\n")


pos_embedding_dim =len(pos_embd[0])
word_embedding_dim = len(word_embd[0])

print(len(pos_embd),len(pos_vocab))
print(len(word_embd),len(word_vocab))

pretrained_pos_embedding = np.array(pos_embd)
pretrained_word_embedding = np.array(word_embd)


print("data load successfully!")

# Split train/test set
# TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6))
    sess = tf.Session(config=session_conf)
    print(np.array(y_train))
    with sess.as_default():
        cnn = TextCNN(
            sequence_length_arg1=x_train_arg1_word.shape[1],
            sequence_length_arg2=x_train_arg2_word.shape[1],
            num_classes=np.array(y_train).shape[1],
            pos_vocab_size=len(pos_vocab),
            word_vocab_size=len(word_vocab),
            pos_embedding_size=pos_embedding_dim ,
            word_embedding_size=word_embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        if FLAGS.use_pretrain:
            cnn.assign_embedding(sess, pretrained_pos_embedding,pretrained_word_embedding)
            #cnn.assign_word_embedding(sess,pretrained_word_embedding)



        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')
        # var_list1 = [var for var in tf.trainable_variables() if
        #                'word_embedding' in var.op.name]
        # var_list2 = [var for var in tf.trainable_variables() if
        #                'word_embedding' not in var.op.name]
        # opt1 = tf.train.AdamOptimizer(0.001)
        # opt2 = tf.train.AdamOptimizer(0.004)
        #
        #
        # grads1 = opt1.compute_gradients(cnn.loss,var_list1)
        # grads2 = opt2.compute_gradients(cnn.loss,var_list2)
        # train_op1 = opt1.apply_gradients(grads1,global_step=global_step, name='train_op1')
        # train_op2 = opt2.apply_gradients(grads2, name='train_op1')
        # train_op = tf.group(train_op1, train_op2)
        # # Keep track of gradient values and sparsity (optional)
        # grads_and_vars = grads1+grads2
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "explicit_" + timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "explicit-model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch_arg1_pos, x_batch_arg2_pos,x_batch_arg1_word,x_batch_arg2_word, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x_arg1_pos: x_batch_arg1_pos,
              cnn.input_x_arg2_pos: x_batch_arg2_pos,
              cnn.input_x_arg1_word:x_batch_arg1_word,
              cnn.input_x_arg2_word:x_batch_arg2_word,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, precision, recall ,predictions= sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall,cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("comparison {}: step {}, loss {:g}"
                  .format(time_str, step, loss))
            train_summary_writer.add_summary(summaries, step)






        def dev_step(x_arg1_pos, x_arg2_pos, x_arg1_word, x_arg2_word, y):
            total_loss = 0.0
            batches = data_helpers.batch_iter(
                list(zip(x_arg1_pos, x_arg2_pos, x_arg1_word, x_arg2_word, y)),
                FLAGS.batch_size,
                1,
                False
            )
            all_predicts = []
            for batch in batches:
                x_batch_arg1_pos, x_batch_arg2_pos, x_batch_arg1_word, x_batch_arg2_word, y_batch = zip(*batch)
                feed_dict = {
                    cnn.input_x_arg1_pos: x_batch_arg1_pos,
                    cnn.input_x_arg2_pos: x_batch_arg2_pos,
                    cnn.input_x_arg1_word: x_batch_arg1_word,
                    cnn.input_x_arg2_word: x_batch_arg2_word,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1
                }
                batch_loss, batch_predicts = sess.run(
                    [cnn.loss, cnn.predictions],
                    feed_dict
                )

                total_loss += batch_loss
                all_predicts.extend(batch_predicts)

            target_names = ["Temporal","Comparison","Contingency","Expansion"]
            print(classification_report(np.argmax(y, axis=1), all_predicts, target_names=target_names,digits=6))
            write2 = open("result.txt","a")
            if len(all_predicts)== 1011:
                write2.write(str(np.argmax(y,axis=1).tolist())+"\n")
                write2.write(str(all_predicts)+"\n")
            write2.close()
            acc = accuracy_score(np.argmax(y, axis=1), all_predicts)
            fscore = f1_score(np.argmax(y, axis=1), all_predicts,average="macro")
            #print("acc:"+accuracy_score(np.argmax(y, axis=1), all_predicts))
            # precision, recall, fscore, _ = precision_recall_fscore_support(
            #     np.argmax(y, axis=1),
            #     all_predicts,
            #     labels=[1, 0],
            #     pos_label=1,
            #     average="binary"
            # )
            
            print("acc",acc)
            print("micro-fscore",fscore)
            print('')
            print('total loss: ', total_loss)
            #print('precision {:g}, recall {:g}, fscore {:g}'.format(precision, recall, fscore))
            print('')
            return fscore, acc







        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train_arg1_pos, x_train_arg2_pos,x_train_arg1_word,x_train_arg2_word, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        max_f1 = 0.0
        max_acc=0.0
        for batch in batches:
            x_batch_arg1_pos, x_batch_arg2_pos,x_batch_arg1_word,x_batch_arg2_word, y_batch = zip(*batch)
            # Training
            train_step(x_batch_arg1_pos, x_batch_arg2_pos, x_batch_arg1_word, x_batch_arg2_word, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            # dev step
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                print("Train: ")
                dev_step(x_train_arg1_pos, x_train_arg2_pos, x_train_arg1_word, x_train_arg2_word, y_train)

                print("Dev: ")
                f1,acc =dev_step(
                    x_dev_arg1_pos,
                    x_dev_arg2_pos,
                    x_dev_arg1_word,
                    x_dev_arg2_word,
                    y_dev)

                if f1 > max_f1 or acc>max_acc:
                    if f1>max_f1:
                        max_f1 = f1
                    if acc>max_acc:
                        max_acc = acc


                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    print("")



