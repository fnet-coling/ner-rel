import tensorflow as tf
from models import *
from comp_models import *
from comp_functions import *
from data.load_fb15k237 import split_relations

def default_init():
    return tf.random_normal_initializer(0.0, 0.1) #Returns an initializer that generates tensors with a normal distribution. args -> mean,stddev




def create_model(kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2,
                 l2_lambda=0.0, is_batch_training=False, type="DistMult",
                 observed_sets=["train_text"], composition=None, num_buckets= 10):
    '''
    Factory Method for all models
    :param type: any or combination of "ModelF", "DistMult", "ModelE", "ModelO", "ModelN"
    :param composition: "Tanh", "LSTM", "GRU", "BiTanh", "BiLSTM", "BiGRU", "BoW" or None
    :return: Model(s) of type "type"
    '''
    if not isinstance(type, list):
        if not composition:
            composition = ""
        with vs.variable_scope(type+"_" + composition):#set the scope of shared variables, e.g., DistMult_RNN
            comp_size = 2*size if type == "ModelE" else size#the dimentions for modelE is double the size of other methods
            if composition == "RNN":
                composition = TanhRNNCompF(kb, comp_size, num_buckets, split_relations, batch_size / (num_neg + 1), learning_rate)
            elif composition == "LSTM":
                composition = LSTMCompF(kb, comp_size, num_buckets, split_relations, batch_size / (num_neg + 1), learning_rate)
            elif composition == "GRU":
                composition = GRUCompF(kb, comp_size, num_buckets, split_relations, batch_size / (num_neg + 1), learning_rate)
            elif composition == "BiRNN":
                composition = BiTanhRNNCompF(kb, comp_size, num_buckets, split_relations, batch_size / (num_neg + 1), learning_rate)
            elif composition == "BiLSTM":
                composition = BiLSTMCompF(kb, comp_size, num_buckets, split_relations, batch_size / (num_neg + 1), learning_rate)
            elif composition == "BiGRU":
                composition = BiGRUCompF(kb, comp_size, num_buckets, split_relations, batch_size / (num_neg + 1), learning_rate)
            elif composition == "BoW":
                composition = BoWCompF(kb, comp_size, num_buckets, split_relations, batch_size / (num_neg + 1), learning_rate)
            else:
                composition = None

        if type == "ModelF":
            return ModelF(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        elif type == "DistMult":
            if composition:
                return CompDistMult(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
            else:
                return DistMult(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        elif type == "ModelE":
            if composition:
                return CompModelE(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
            else:
                return ModelE(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        elif type == "ModelO":
            if composition:
                return CompModelO(kb, size, batch_size, composition, is_train, num_neg, learning_rate, observed_sets)
            else:
                return ModelO(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training, observed_sets)
        elif type == "WeightedModelO":
            if composition:
                return CompWeightedModelO(kb, size, batch_size, composition, is_train, num_neg, learning_rate, observed_sets)
            else:
                return WeightedModelO(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training, observed_sets)
        elif type == "BlurWeightedModelO":
            return BlurWeightedModelO(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training, observed_sets)
        elif type == "ModelN":
            return ModelN(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        else:
            raise NameError("There is no model with type %s. "
                            "Possible values are 'ModelF', 'DistMult', 'ModelE', 'ModelO', 'ModelN'." % type)
    else:
        if composition:
            return CompCombinedModel(type, kb, size, batch_size, is_train, num_neg,
                                     learning_rate, l2_lambda, is_batch_training, composition)
        else:
            return CombinedModel(type, kb, size, batch_size, is_train, num_neg,
                                 learning_rate, l2_lambda, is_batch_training, composition)


#@tf.ops.RegisterGradient("SparseToDense")
def _tf_sparse_to_dense_grad(op, grad):
    grad_flat = tf.reshape(grad, [-1]) #flatten grad to a 1-D vector, e.g., [0,1,2,3,4,5...]
    sparse_indices = op.inputs[0] 
    d = tf.gather(tf.shape(sparse_indices), [0])#get the dimensions
    shape = op.inputs[1]
    cols = tf.gather(shape, [1])
    ones = tf.expand_dims(tf.ones(d, dtype=tf.int64), 1)#insert one dimention to [d,1]
    cols = ones * cols
    conc = tf.concat(1, [cols, ones])
    sparse_indices = tf.reduce_sum(tf.mul(sparse_indices, conc), 1)
    in_grad = tf.nn.embedding_lookup(grad_flat, sparse_indices)
    return None, None, in_grad, None
