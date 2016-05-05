import random
import os
import time
from data.load_fb15k237 import load_fb15k, load_fb15k_type_constraints, split_relations
from sampler import *
from eval import eval_triples
from model import *
from model.comp_models import *
import sys
from kb import subsample_kb
import shutil
import json
from tensorflow.models.rnn.rnn_cell import *


# data loading specifics
tf.app.flags.DEFINE_string('fb15k_dir', "data_rel", 'data dir containing extracted files of fb15k dataset.')

# model
tf.app.flags.DEFINE_integer("size", 50, "hidden size of model")

# training
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
tf.app.flags.DEFINE_float("l2_lambda", 0, "L2-regularization raten (only for batch training).")
tf.app.flags.DEFINE_float("sample_text_prob", 0.935,
                          "Probability of sampling text triple (default is ratio of text (emnlp) to kb triples.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("num_neg", 200, "Number of negative examples for training.")
tf.app.flags.DEFINE_integer("pos_per_batch", 100, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", -1, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_integer("subsample_kb", -1, "num of entities in subsampled kb. if <= 0 use whole kb")
tf.app.flags.DEFINE_boolean("kb_only", False, "Only load and train on FB relations, ignoring text.")
tf.app.flags.DEFINE_boolean("batch_train", False, "Use batch training.")
tf.app.flags.DEFINE_boolean("type_constraint", False, "Use type constraint during sampling.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("model", "ModelE",
                           "Model architecture or combination thereof split by comma of: "
                           "'DistMult', 'DistMult', 'ModelE', 'ModelO', 'ModelN', 'WeightedModelO'")
tf.app.flags.DEFINE_string("observed_sets", "train_text", "Which sets to observe for observed models.")
tf.app.flags.DEFINE_string("valid_mode", "a", "[a,t,nt] are possible. a- validate on all triples, "
                                              "t- validate only on triples with text mentions, "
                                              "nt- validate only on triples without text mentions")
tf.app.flags.DEFINE_string("composition", "BiRNN", "'LSTM', 'GRU', 'RNN', 'BoW', 'BiLSTM', 'BiGRU', 'BiRNN'")

FLAGS = tf.app.flags.FLAGS

if "," in FLAGS.model:
    FLAGS.model = FLAGS.model.split(",")

FLAGS.observed_sets = FLAGS.observed_sets.split(",")

assert (not FLAGS.batch_train or FLAGS.ckpt_its <= -1), "Do not define checkpoint iterations when doing batch training."

if FLAGS.batch_train:
    print("Batch training!")

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

kb = load_fb15k(FLAGS.fb15k_dir, with_text=not FLAGS.kb_only)
if FLAGS.subsample_kb > 0:
    kb = subsample_kb(kb, FLAGS.subsample_kb)

if FLAGS.type_constraint:
    print("Loading type constraints...")
    load_fb15k_type_constraints(kb, os.path.join(FLAGS.fb15k_dir, "types"))

num_kb = 0
num_text = 0

for f in kb.get_all_facts():
    if f[2] == "train":
        num_kb += 1
    elif f[2] == "train_text":
        num_text += 1

print("Loaded data. %d kb triples. %d text_triples." % (num_kb, num_text))
batch_size = (FLAGS.num_neg+1) * FLAGS.pos_per_batch * 2  # x2 because subject and object loss training

fact_sampler = BatchNegTypeSampler(kb, FLAGS.pos_per_batch, which_set="train", neg_per_pos=FLAGS.num_neg, type_constraint=FLAGS.type_constraint)
if not FLAGS.kb_only:
    text_sampler = BatchNegTypeSampler(kb, FLAGS.pos_per_batch, which_set="train_text", neg_per_pos=FLAGS.num_neg, type_constraint=False)
print("Created Samplers.")

train_dir = os.path.join(FLAGS.save_dir, "train")

i = 0

subsample_validation = map(lambda x: x[0], kb.get_all_facts_of_arity(2, "valid"))
if len(subsample_validation) > 5000:
    subsample_validation = random.sample(subsample_validation, 5000)


if FLAGS.ckpt_its <= 0:
    print "Setting checkpoint iteration to size of whole epoch."
    FLAGS.ckpt_its = text_sampler.epoch_size

with tf.Session() as sess:
    print "Creating model ...",FLAGS.composition
    model = create_model(kb, FLAGS.size, batch_size, num_neg=FLAGS.num_neg, learning_rate=FLAGS.learning_rate,
                         l2_lambda=FLAGS.l2_lambda, is_batch_training=FLAGS.batch_train, type=FLAGS.model,
                         observed_sets=FLAGS.observed_sets, composition=FLAGS.composition)

    print "Created model: " + model.name()

    if os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: ".ckpt" in x, os.listdir(train_dir))), key=os.path.getctime)
        print "Loading from checkpoint " + newest
        model.saver.restore(sess, newest)
    else:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        sess.run(tf.initialize_all_variables())

    num_params = reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
    print("Num params: %d" % num_params)


    print("Initialized model.")
    loss = 0.0
    step_time = 0.0
    previous_mrrs = list()
    mrr2modelpath = dict()
    e = 0
    mode = "update"
    if FLAGS.batch_train:
        mode = "accumulate"

    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    end_of_epoch = False
    def sample_next_batch():
        #if FLAGS.kb_only or random.random() >= FLAGS.sample_text_prob:
         #   return fact_sampler.get_batch_async()
        #else:
        return text_sampler.get_batch_async()

    next_batch = sample_next_batch()

    while FLAGS.max_iterations < 0 or i < FLAGS.max_iterations:
        i += 1
        start_time = time.time()
        pos, negs = next_batch.get()
        end_of_epoch = text_sampler.end_of_epoch()
        current_ct = text_sampler.count
        # already fetch next batch parallel to running model
        next_batch = sample_next_batch()

        loss += model.step(sess, pos, negs, mode) #update the paratmers and return the accumulated loss
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f" %
                         (float((i-1) % FLAGS.ckpt_its + 1.0)*100.0 / FLAGS.ckpt_its,
                          loss / float((i-1) % FLAGS.ckpt_its + 1.0)))
        sys.stdout.flush()


        if (end_of_epoch and not text_sampler.end_of_epoch() and FLAGS.batch_train) or (not FLAGS.batch_train and i % FLAGS.ckpt_its == 0):
            if not FLAGS.batch_train:
                loss /= FLAGS.ckpt_its
                print ""
                print "%d%% in epoch done." % (100*current_ct/fact_sampler.epoch_size)
            # Print statistics for the previous epoch.
	    e+=1
            step_time /= FLAGS.ckpt_its
            print "global step %d learning rate %.4f, step-time %.3f, loss %.4f" % (model.global_step.eval(),
                                                                                    model.learning_rate.eval(),
                                                                                    step_time, loss)
            step_time, loss = 0.0, 0.0
            valid_loss = 0.0
	    print "epoch ",e
            # Run evals on development set and print their perplexity.
            if e >= 8:
                print "Stop learning!"
                break
                #lr = model.learning_rate.eval()
                #sess.run(model.learning_rate.assign(lr * FLAGS.learning_rate_decay))
                #print "Decaying learning rate to: %.4f" % model.learning_rate.eval()

            # Save checkpoint and zero timer and loss.
            path = model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print "####################################"

    print "##############################"
