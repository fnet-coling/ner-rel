
from gensim.models import Word2Vec
from gensim.matutils import argsort
from numpy import array,asarray
import numpy as np
import sys
class WARPModel:
    def __init__(self,label_vec_f,feature_vec_f,binary=False):
        label2vec = Word2Vec.load_word2vec_format(label_vec_f,binary=binary)
        self.label_embed =  label2vec.syn0
        self.dictionary = label2vec.index2word
        self.vocab = label2vec.vocab
        self.feat_embed = Word2Vec.load_word2vec_format(feature_vec_f,binary=binary).syn0
    def scores(self,X):
        return np.dot(np.sum(self.feat_embed[X],axis=0),self.label_embed.T)
    def label_rank(self,X):
        return argsort(self.scores(X),reverse=True)
    def batch_predict(self,Xs,topk=2,use_text=False):
        if use_text:
            return [map(lambda x:self.dictionary[x],self.label_rank(X[1])[:topk]) for X in Xs]
        else:
            return [refine(self.label_rank(X[1])[:topk]) for X in Xs]
        
def read_data(fn):
    data = []
    with open(fn,'r') as f:
        for ln in f:
            splited = ln.rstrip().split()
            data.append([splited[0],map(int,splited[1].split(','))])
    return data
def build_mention_map(fn):
    mmap = dict()
    with open(fn,'r') as f:
        for ln in f:
            splited= ln.rstrip().split()
            mmap[splited[0]] = splited[1]
    return mmap
def refine(labels,maxDepth=2,delim='/'):
    keep = [""]*maxDepth
    for l in labels:
        path = getPath(l,delim)
        for i in range(len(path)):
            if keep[i] =="" :
                keep[i] = path[i]
            elif keep[i] != path[i]:
                break
    results = []
    tmp= ''
    for l in keep:
        if l!="":
            tmp+=delim
            tmp +=l
            results.append(tmp)
        
    return results        
def getPath(label,delim='/'):
    return label.split(delim)[1:] 
def main():
    label_embed_fn = sys.argv[1]
    feature_embed_fn = sys.argv[2]
    data_fn = sys.argv[3]
    mention_map = build_mention_map(sys.argv[4])
    model = WARPModel(label_embed_fn,feature_embed_fn)
    data = read_data(data_fn)
    delimer=sys.argv[7]
    out = open(sys.argv[5],'wb')
    predictions = model.batch_predict(data,use_text=True,topk=int(sys.argv[6]))
    for i in xrange(len(predictions)):
        for l in refine(predictions[i],delim=delimer):
            out.write("%s\t%s\t1\n" % (mention_map[data[i][0]],str(model.vocab[l].index)))
    out.close()


if __name__ == "__main__":
	    main()      
