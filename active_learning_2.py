#! /usr/bin/env python
#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/explicit_1515300803/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_integer("max_document_length",100,"Max document length")
# Misc Parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,,,,,,,,
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#
#load pos file
x_text_train_arg1_pos, x_text_train_arg2_pos, y_train = data_helpers.load_pos_pkl('train_pos_file')
x_text_dev_arg1_pos, x_text_dev_arg2_pos, y_dev = data_helpers.load_pos_pkl('dev_pos_file')
x_text_explicit_arg1_pos, x_text_explicit_arg2_pos, y_explicit = data_helpers.load_pos_pkl('explicit_pos_file')
#load word file
x_text_train_arg1_word, x_text_train_arg2_word, y_train = data_helpers.load_word_pkl('train_word_file')
x_text_dev_arg1_word, x_text_dev_arg2_word, y_dev = data_helpers.load_word_pkl('dev_word_file')
x_text_explicit_arg1_word, x_text_explicit_arg2_word,y_explicit = data_helpers.load_word_pkl('explicit_word_file')
#load vocab and embd

file_pkl = open("./pos_vocab_embd_word_vocab_embd.pkl", "rb")
pos_vocab,pos_embd, word_vocab, word_embd = pickle.load(file_pkl)
file_pkl.close()


x_train_arg1_pos = data_helpers.build_input_data(x_text_train_arg1_pos, pos_vocab, FLAGS.max_document_length)
x_train_arg2_pos = data_helpers.build_input_data(x_text_train_arg2_pos, pos_vocab, FLAGS.max_document_length)
x_dev_arg1_pos = data_helpers.build_input_data(x_text_dev_arg1_pos, pos_vocab, FLAGS.max_document_length)
x_dev_arg2_pos = data_helpers.build_input_data(x_text_dev_arg2_pos, pos_vocab, FLAGS.max_document_length)
x_explicit_arg1_pos = data_helpers.build_input_data(x_text_explicit_arg1_pos, pos_vocab, FLAGS.max_document_length)
x_explicit_arg2_pos = data_helpers.build_input_data(x_text_explicit_arg2_pos, pos_vocab, FLAGS.max_document_length)

x_train_arg1_word = data_helpers.build_input_data(x_text_train_arg1_word, word_vocab, FLAGS.max_document_length)
x_train_arg2_word = data_helpers.build_input_data(x_text_train_arg2_word, word_vocab, FLAGS.max_document_length)
x_dev_arg1_word = data_helpers.build_input_data(x_text_dev_arg1_word, word_vocab, FLAGS.max_document_length)
x_dev_arg2_word = data_helpers.build_input_data(x_text_dev_arg2_word, word_vocab, FLAGS.max_document_length)
x_explicit_arg1_word = data_helpers.build_input_data(x_text_explicit_arg1_word, word_vocab, FLAGS.max_document_length)
x_explicit_arg2_word = data_helpers.build_input_data(x_text_explicit_arg2_word, word_vocab, FLAGS.max_document_length)




graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6))
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        global_step = 0
        max_f1 = 0.0

        for i in range(10):

            checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.checkpoint_dir, "checkpoints"))
            # checkpoint_file = 'runs/1498291496/checkpoints/active-4-model-20'
            print('checkpoint_file: ', checkpoint_file)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print('model restored...')
            # Get the placeholders from the graph by name
            # Get the placeholders from the graph by name
            input_x_arg1_pos = graph.get_operation_by_name("input_x_arg1_pos").outputs[0]
            input_x_arg2_pos = graph.get_operation_by_name("input_x_arg2_pos").outputs[0]
            input_x_arg1_word = graph.get_operation_by_name("input_x_arg1_word").outputs[0]
            input_x_arg2_word = graph.get_operation_by_name("input_x_arg2_word").outputs[0]

            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            scores = graph.get_operation_by_name("output/scores").outputs[0]

            train_op = graph.get_operation_by_name("train_op").outputs[0]

            precision = graph.get_operation_by_name("accuracy/precision").outputs[0]
            recall = graph.get_operation_by_name("accuracy/recall").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

            loss = graph.get_operation_by_name("loss/losses").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]


            def train_step(x_batch_arg1_pos, x_batch_arg2_pos,x_batch_arg1_word,x_batch_arg2_word, y_batch):


                feed_dict = {
                    input_x_arg1_pos: x_batch_arg1_pos,
                    input_x_arg2_pos: x_batch_arg2_pos,
                    input_x_arg1_word:x_batch_arg1_word,
                    input_x_arg2_word:x_batch_arg2_word,
                    input_y: y_batch,
                    dropout_keep_prob: 0.5

                }
                _, _precision, _recall, _accuracy= sess.run(
                    [train_op, precision, recall, accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()

                print("{}: step {}, precision {:g}, recall {:g}, accuracy{:g}".format(time_str, global_step, _precision, _recall,_accuracy))


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
                        input_x_arg1_pos: x_batch_arg1_pos,
                        input_x_arg2_pos: x_batch_arg2_pos,
                        input_x_arg1_word: x_batch_arg1_word,
                        input_x_arg2_word: x_batch_arg2_word,
                        input_y: y_batch,
                        dropout_keep_prob: 1
                    }
                    batch_loss, batch_predicts = sess.run(
                        [loss, predictions],
                        feed_dict
                    )

                    total_loss += batch_loss
                    all_predicts.extend(batch_predicts)

                target_names = ["Temporal", "Comparison", "Contingency", "Expansion"]
                print(classification_report(np.argmax(y, axis=1), all_predicts, target_names=target_names, digits=6))
                acc = accuracy_score(np.argmax(y, axis=1), all_predicts)
                fscore = f1_score(np.argmax(y, axis=1), all_predicts, average="macro")
                # print("acc:"+accuracy_score(np.argmax(y, axis=1), all_predicts))
                # precision, recall, fscore, _ = precision_recall_fscore_support(
                #     np.argmax(y, axis=1),
                #     all_predicts,
                #     labels=[1, 0],
                #     pos_label=1,
                #     average="binary"
                # )
                print(str)
                print("acc", acc)
                print("micro-fscore", fscore)
                print('')
                print('total loss: ', total_loss)
                # print('precision {:g}, recall {:g}, fscore {:g}'.format(precision, recall, fscore))
                print('')
                return fscore, acc


            print("initial f score")

            #Also use this step to evaluate the final result on test set;
            max_f1, max_acc = dev_step(x_dev_arg1_pos, x_dev_arg2_pos, x_dev_arg1_word, x_dev_arg2_word, y_dev)



            #
            #
            # pick index....

            print('active learning... ', i)

            all_scores = None

            batches = data_helpers.batch_iter(list(zip(x_explicit_arg1_pos, x_explicit_arg2_pos,x_explicit_arg1_word,x_explicit_arg2_word,y_explicit)),
                                              FLAGS.batch_size, 1,
                                              shuffle=False)



            for batch in batches:
                x_explicit_batch_arg1_pos, x_explicit_batch_arg2_pos,x_explicit_batch_arg1_word,x_explicit_batch_arg2_word,y_batch = zip(*batch)
                #print(x_test_batch_arg1)
                feed_dict = {
                    input_x_arg1_pos: x_explicit_batch_arg1_pos,
                    input_x_arg2_pos: x_explicit_batch_arg2_pos,
                    input_x_arg1_word:x_explicit_batch_arg1_word,
                    input_x_arg2_word:x_explicit_batch_arg2_word,
                    dropout_keep_prob: 1.0,

                }
                batch_scores = sess.run(scores, feed_dict)
                #print(batch_scores)
                batch_scores_op = tf.nn.softmax(batch_scores)
                batch_scores = sess.run(batch_scores_op)
               # print(batch_scores)

                if all_scores is None:
                    all_scores = batch_scores
                else:
                    all_scores = np.concatenate([all_scores,batch_scores])

            all_scores = np.array(all_scores)
            print(all_scores)

            index_picked = np.where(np.sum(np.multiply(-all_scores, np.log2(all_scores)), axis=1)> 0.95)

            index_picked = index_picked[0].tolist()

            print('pick index', len(index_picked))

            # y_add1 = np.array(y_explicit_array_label, dtype=np.int64)[index_picked]
            # y_add2 = np.array(y_explicit_array_label, dtype=np.int64)[index_picked]
            # print(y_add1)
            # print(y_add2)





            print('pick index', len(index_picked))

            #sampling without replacement
            add_train_pos_arg1 = np.array(x_explicit_arg1_pos, dtype=np.int64)[index_picked]
            add_train_pos_arg2 = np.array(x_explicit_arg2_pos, dtype=np.int64)[index_picked]
            add_train_word_arg1 = np.array(x_explicit_arg1_word, dtype=np.int64)[index_picked]
            add_train_word_arg2 = np.array(x_explicit_arg2_word, dtype=np.int64)[index_picked]
            y_add = np.array(y_explicit, dtype=np.int64)[index_picked]

            #sampling with replacement
            # x_explicit_arg1_pos = np.delete(x_explicit_arg1_pos, index_picked, axis=0)
            # x_explicit_arg1_pos = np.delete(x_explicit_arg1_pos, index_picked, axis=0)
            # x_explicit_arg2_pos = np.delete(x_explicit_arg2_pos, index_picked, axis=0)
            # x_explicit_arg1_word= np.delete(x_explicit_arg1_word, index_picked, axis=0)
            # x_explicit_arg2_word=np.delete(x_explicit_arg2_word, index_picked, axis=0)
            # y_explicit = np.delete(y_explicit, index_picked, axis=0)
            #y_explicit_array_label = np.delete(y_explicit_array_label, index_picked_leverage, axis=0)

            x_train_arg1_pos_add= np.concatenate((x_train_arg1_pos, add_train_pos_arg1))
            x_train_arg2_pos_add = np.concatenate((x_train_arg2_pos, add_train_pos_arg2))
            x_train_arg1_word_add = np.concatenate((x_train_arg1_word, add_train_word_arg1))
            x_train_arg2_word_add = np.concatenate((x_train_arg2_word, add_train_word_arg2))
            y_train_label_add = np.concatenate((y_train, y_add))

            add_batches = data_helpers.batch_iter(list(zip(x_train_arg1_pos_add, x_train_arg2_pos_add,
                                                           x_train_arg1_word_add, x_train_arg2_word_add, y_train_label_add)),
                                                  FLAGS.batch_size, FLAGS.batch_size, shuffle=True)

            # saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            for batch in add_batches:

                global_step += 1
                x_add_batch_arg1_pos, x_add_batch_arg2_pos,x_add_batch_arg1_word, x_add_batch_arg2_word, y_batch = zip(*batch)
                train_step(x_add_batch_arg1_pos, x_add_batch_arg2_pos,x_add_batch_arg1_word, x_add_batch_arg2_word, y_batch)

                if global_step % 100 == 0:
                    print("")
                    f1, acc = dev_step(x_dev_arg1_pos, x_dev_arg2_pos,x_dev_arg1_word, x_dev_arg2_word, y_dev)
                    if f1 > max_f1 and acc>max_acc:

                        path = saver.save(sess, os.path.join(os.path.join(FLAGS.checkpoint_dir, "checkpoints"),
                                                             'start4-active-' + str(i) + "-" + "model"),
                                          global_step=global_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    print("")

