import random
import numpy as np
import sys
from random import randint
from data_load import *
in_dir= "/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN"
a=MentionData('/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/train_x_new.txt',
              "/Users/mayk/working/figer/baseline/PLE/Intermediate/BBN/train_y.txt",
             in_dir+"/feature.txt",in_dir+"/type.txt")

import numpy as np

cimport numpy as np
from random import randint
import sys
import cython

import math
from libc.stdlib cimport malloc, free

from libc.math cimport exp
from libc.math cimport log
from libc.math cimport sqrt
from libc.string cimport memset
import random
# scipy <= 0.15

import scipy.linalg.blas as fblas
ctypedef np.float32_t REAL_t
cdef int ONE = 1


REAL = np.float32
cdef extern from "/Users/mayk/working/figer/baseline/PLE/Model/warp/voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)
DEF MAX_SENTENCE_LEN = 10000
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil


cdef scopy_ptr scopy = <scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
DEF EXP_TABLE_SIZE = 10000
DEF MAX_EXP = 50

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef REAL_t ONEF = <REAL_t>1.0

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < 50 by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]
cdef REAL_t cvdot(vec1,vec2,size):
    cdef int csize = size
    f= dsdot(&csize,<REAL_t *>(np.PyArray_DATA(vec1)),&ONE,<REAL_t *>(np.PyArray_DATA(vec2)),&ONE)
    return f
def csaxpy(vec1,vec2,alpha,size):
    cdef int csize = size
    cdef float calpha = alpha
    f= our_saxpy_noblas(&csize,&calpha,<REAL_t *>(np.PyArray_DATA(vec1)),&ONE,<REAL_t *>(np.PyArray_DATA(vec2)),&ONE)
    return f
cdef REAL_t crank(int k):
    cdef REAL_t loss = 0.
    cdef int i = 1
    for i in xrange(1,k+1):
        loss += ONEF/i
    return loss
cdef REAL_t vsum(REAL_t *vec,int *size):
    cdef int i
    cdef REAL_t product
    product = <REAL_t>0.0
    for i from 0 <= i < size[0] by 1:
        product += vec[i] * vec[i]
    return np.sqrt(product)
def cnorm(vec):
    cdef int size
    size  = len(vec)
    return vsum(<REAL_t *>(np.PyArray_DATA(vec)),&size)
def init():
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
#init()


def ctrain(A,B,insts,size,lr,gradient,max_it =10):
    cdef float error


    for it in xrange(1,max_it+1):
        error = 0.
        for i,inst in enumerate(insts):
            error+=gradient(A,B,inst,size,lr=lr)
        
            if i % 1000 ==0:
                sys.stdout.write("\rIteration %d " % (it)+ "trained {0:.0f}%".format(float(i)*100/len(insts))+" Loss:{0:.2f}".format(error))
                sys.stdout.flush()
        sys.stdout.write("\n")
cdef void divide(REAL_t *vec, const float *alpha, const int *size):
    cdef int i
    cdef REAL_t product
    for i from 0 <= i < size[0] by 1:
        vec[i] = vec[i]/alpha[0]
def cdivide(vec,alpha):
    cdef int size
    size  = len(vec)
    cdef float r = alpha
    divide(<REAL_t *>(np.PyArray_DATA(vec)),&r,&size)

    
def max_margin_gradient(A,B,inst,size,lr=0.01):
    #print B
    #print B[0]-B[9]

    dA = np.zeros(size,dtype=REAL)
    dB = np.zeros([len(inst.labels),size],dtype=REAL)
    random.seed(1)
    x = np.sum(A[inst.features],axis=0)
    cdef REAL_t error = 0
    cdef REAL_t s1,s2
    cdef clr = lr
    cdef int N
    cdef int n_sample
    cdef int neg_num = len(inst.negative_labels)
    cdef REAL_t norm
    for l in inst.sparse_labels:
        s1= cvdot(x,B[l],50)
        N=1
        n_sample  = -1
        for k in xrange(neg_num):
            nl = inst.negative_labels[randint(0,neg_num-1)]
            s2 = cvdot(x,B[nl],50)
            if s1 - s2<1:
                n_sample = nl
                N = k+1
                break
        if n_sample!=-1:
            error += 1 + s2-s1
            csaxpy(B[l]-B[n_sample],dA,1.0,50)
            csaxpy(x,dB[l],1.0,50)
            csaxpy(x,dB[n_sample],-1.0,50)

    for f in inst.features:
        csaxpy(dA,A[f],clr,50)
        norm = cnorm(A[f])
        if norm >1:
            cdivide(A[f],norm)

    for i in xrange(len(B)):
        csaxpy(dB[i],B[i],clr,50)
        #B[i] += lr*dB[i]
        norm =  cnorm(B[i])
        if norm >1:
            cdivide(B[i],norm)
            #B[i] /=norm
    return error

def max_max_margin_gradient(A,B,inst,size,lr=0.01):
    #print B
    #print B[0]-B[9]

    dA = np.zeros(size,dtype=REAL)
    dB = np.zeros([len(inst.labels),size],dtype=REAL)
    random.seed(1)
    x = np.sum(A[inst.features],axis=0)
    cdef REAL_t error = 0
    cdef REAL_t s1,s2
    cdef clr = lr
    cdef int N
    cdef int neg_num = len(inst.negative_labels)
    cdef REAL_t norm
    cdef int max_l,max_nl
    cdef REAL_t max_s1= float('-inf'),max_s2=float('-inf')
    for l in inst.sparse_labels:
        s1= cvdot(x,B[l],50)
        if s1>=max_s1:
            max_l = l
            max_s1 = s1
    for nl in inst.negative_labels:
        s2= cvdot(x,B[nl],50)
        if s2>=max_s2:
            max_nl = nl
            max_s2 = s2
    if max_s1-max_s2<1:
        error += 1 + max_s2-max_s1
        csaxpy(B[max_l]-B[max_nl],dA,1.0,50)
        csaxpy(x,dB[max_l],1.0,50)
        csaxpy(x,dB[max_nl],-1.0,50)

    for f in inst.features:
        csaxpy(dA,A[f],clr,50)
        norm = cnorm(A[f])
        if norm >1:
            cdivide(A[f],norm)

    for i in xrange(len(B)):
        csaxpy(dB[i],B[i],clr,50)
        #B[i] += lr*dB[i]
        norm =  cnorm(B[i])
        if norm >1:
            cdivide(B[i],norm)
            #B[i] /=norm
    return error
def softmax_gradient(A,B,inst,size,lr=0.01):
    #print B
    #print B[0]-B[9]

    dA = np.zeros(size,dtype=REAL)
    random.seed(1)
    x = np.mean(A[inst.features],axis=0)
    cdef REAL_t error = 0
    cdef REAL_t s1,s2,logs1,logs2,g
    cdef clr = lr
    cdef int neg_num = len(inst.negative_labels)
    cdef REAL_t norm
    cdef REAL_t pos = ONEF
    cdef int csize =size
    for l in inst.sparse_labels:
        logs1= cvdot(x,B[l],50)
        if  logs1 <= -MAX_EXP or  logs1 >= MAX_EXP:
            continue
        s1 = EXP_TABLE[<int>((logs1 + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        error -=logs1
        g = (pos - s1) * clr
        csaxpy(B[l],dA,g,50)
        saxpy(&csize, &g, <REAL_t *>(np.PyArray_DATA(x)), &ONE, <REAL_t *>(np.PyArray_DATA(B[l])), &ONE)
        for k in xrange(neg_num):
            nl = inst.negative_labels[randint(0,neg_num-1)]
            logs2 = cvdot(x,B[nl],50)
            if  logs2 <= -MAX_EXP or  logs2 >= MAX_EXP:
                continue
            s2 = EXP_TABLE[<int>((logs2 + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            if s2 > s1:
                g =  (-s2) * clr
                csaxpy(B[nl],dA,g,50)
                saxpy(&csize, &g, <REAL_t *>(np.PyArray_DATA(x)), &ONE, <REAL_t *>(np.PyArray_DATA(B[nl])), &ONE)
                error += logs2
    

    for f in inst.features:
        csaxpy(dA,A[f],clr,50)
        norm = cnorm(A[f])
        #if norm >1:
         #   cdivide(A[f],norm)

    for i in xrange(len(B)):
        norm =  cnorm(B[i])
       # if norm >1:
        #    cdivide(B[i],norm)
            #B[i] /=norm
    return error



def pure_ctrain(A,B,insts,size,lr,label_size,max_it =10):
    cdef REAL_t error
    cdef REAL_t *cA = <REAL_t *>(np.PyArray_DATA(A))
    cdef REAL_t *cB = <REAL_t *>(np.PyArray_DATA(B))
    cdef int rows = label_size
    cdef REAL_t *dB = <REAL_t *>malloc(size*rows * sizeof(REAL_t))
    cdef REAL_t *dA =  <REAL_t *>malloc(size * sizeof(REAL_t))
    cdef REAL_t *x =  <REAL_t *>malloc(size * sizeof(REAL_t))
    feats = []
    for inst in insts:
        feats.extend(inst.features)
    cdef int *features =  <int *>(np.PyArray_DATA(np.array(feats,dtype=np.int32)))
    indices = []
    cnt = 0
    for inst in insts:
        indices.append(cnt)
        cnt += len(inst.features)
    indices.append(cnt)

    cdef int *f_idx = <int *>(np.PyArray_DATA(np.asarray(indices,dtype=np.int32)))
    cdef int *labels = <int *>(np.PyArray_DATA(np.asarray([inst.sparse_labels+inst.negative_labels for inst in insts],dtype=np.int32)))
    cdef int *pos_cnts = <int *>(np.PyArray_DATA(np.asarray([len(inst.sparse_labels) for inst in insts],dtype=np.int32)))
    cdef int cSize = size
    for it in xrange(1,max_it+1):
        error = 0.
        for i,inst in enumerate(insts):
            memset(x, 0, size * cython.sizeof(REAL_t))
            for f in inst.features:
                our_saxpy_noblas(&cSize,&ONEF,&cA[f*cSize],&ONE,x,&ONE)
            error+=cwarp_gradient(cA,cB,dB,dA,x,&features[f_idx[i]],f_idx[i+1]-f_idx[i],&labels[i*label_size],pos_cnts[i],label_size,cSize,lr)
        
            if i % 1000 ==0:
                sys.stdout.write("\rIteration %d " % (it)+ "trained {0:.0f}%".format(float(i)*100/len(insts))+" Loss:{0:.2f}".format(error))
                sys.stdout.flush()
        sys.stdout.write("\n")
   # free(f_idx)
  #  free(pos_cnts)
#    free(labels)
    free(dA)
    free(dB)
    free(x)
    return None
cdef REAL_t cwarp_gradient(REAL_t *cA,REAL_t *cB,REAL_t *dB,REAL_t *dA,
                                       REAL_t *x,int *features,int fcnt, int *label,int pos_cnt,int label_cnt,
                                       int cSize,float clr):
    memset(dB,0, cSize *label_cnt* cython.sizeof(REAL_t)) 
    memset(dA, 0, cSize * cython.sizeof(REAL_t))
    random.seed(1)
    cdef REAL_t error = 0.0
    cdef REAL_t s1,s2
    cdef float L,negL
    cdef int N,n_sample 
    cdef int neg_num = label_cnt-pos_cnt
    cdef REAL_t norm = <REAL_t>0.0
    
    

    for pos in range(pos_cnt):
        l = label[pos]
        s1 = dsdot(&cSize,&cB[l*cSize],&ONE,x,&ONE)
        N=1
        n_sample  = -1
        for k in xrange(neg_num):
            nl = label[pos_cnt+randint(0,neg_num-1)]
            s2 = dsdot(&cSize,&cB[nl*cSize],&ONE,x,&ONE)
            if s1 - s2<1:
                n_sample = nl
                N = k+1
                break
        if n_sample!=-1:
            L = crank(neg_num/N)
            negL = -L
            error += (1+s2-s1)*L            
            our_saxpy_noblas(&cSize,&L,&cB[l*cSize],&ONE,dA,&ONE)
            our_saxpy_noblas(&cSize,&negL,&cB[n_sample*cSize],&ONE,dA,&ONE)

            our_saxpy_noblas(&cSize,&L,x,&ONE,&dB[l*cSize],&ONE)
            our_saxpy_noblas(&cSize,&negL,x,&ONE,&dB[n_sample*cSize],&ONE)
            
    for i in range(fcnt):
        f = features[i]
    
        our_saxpy_noblas(&cSize,&clr,dA,&ONE,&cA[f*cSize],&ONE)
#         norm = <REAL_t>0.0

#         for m in range(cSize):
#             #print cA[f*cSize+m]
#             norm+= cA[f*cSize+m]*cA[f*cSize+m]
        
#         norm = sqrt(norm)
     
#         if norm >1:
#             divide(&cA[f*cSize],&norm,&cSize)

#     for i in xrange(label_cnt):
#         our_saxpy_noblas(&cSize,&clr,&dB[i*cSize],&ONE,&cB[i*cSize],&ONE)
#         norm = <REAL_t>0.0
         
#         for m in range(cSize):
#             norm+= cB[i*cSize+m]*cB[i*cSize+m]
#         norm = sqrt(norm)

#          #norm =  vsum(&cB[i*cSize],&cSize)
#         if norm >1:
#             divide(&cB[i*cSize],&norm,&cSize)


    return error

def warp_gradient(A,B,inst,size,lr=0.01):
    #print B
    #print B[0]-B[9]
    dA = np.zeros(size,dtype=REAL)
    dB = np.zeros([len(inst.labels),size],dtype=REAL)
    random.seed(1)
    x = np.sum(A[inst.features],axis=0)
    cdef REAL_t error = 0.

    cdef REAL_t clr = lr
    cdef int N,n_sample 
    cdef int neg_num = len(inst.negative_labels)
    cdef REAL_t norm
    cdef int cSize = size
    cdef REAL_t floats
    
   
    for l in inst.sparse_labels:
        s1= cvdot(x,B[l],50)
        N=1
        n_sample  = -1
        for k in xrange(neg_num):
            nl = inst.negative_labels[randint(0,neg_num-1)]
            s2 = cvdot(x,B[nl],50)
            
            if s1 - s2<1:
                n_sample = nl
                N = k+1
                break
        if n_sample!=-1:

            L = crank(len(inst.negative_labels)/N)
            negL = -L
            error += (1+s2-s1)*L

            csaxpy(B[l]-B[n_sample],dA,L,50)
            
            csaxpy(x,dB[l],L,50)
            csaxpy(x,dB[n_sample],-L,50)
            #print dB[l][0]
    for f in inst.features:
        csaxpy(dA,A[f],clr,50)
#         norm = cnorm(A[f])
#         if norm >1:
#             cdivide(A[f],norm)

#     for i in xrange(len(B)):
#         csaxpy(dB[i],B[i],clr,50)

#         #B[i] += lr*dB[i]
#         norm =  cnorm(B[i])
#         if norm >1:
#             cdivide(B[i],norm)
#             #B[i] /=norm
    return error
def save_to_text(matrix,output):
    shape = matrix.shape
    with open(output,'wb') as out:
        out.write("%d %d\n" % (shape))
        for row in matrix:
            x = " ".join(map(lambda x:"{0:.5}".format(x),row))
            out.write(x+"\n")


