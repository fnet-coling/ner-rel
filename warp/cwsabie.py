from cwsabie_inner import *
size=50
np.random.seed(1)
A= np.random.rand(len(a.feature2id),size).astype(np.float32)
for i in xrange(len(A)):
    norm =  cnorm(A[i])
    if norm >1:
        cdivide(A[i],norm)
B= np.random.rand(len(a.label2id),size).astype(np.float32)
for i in xrange(len(B)):
    norm =  cnorm(B[i])
    if norm >1:
        cdivide(B[i],norm)
pure_ctrain(A,B,a.data,50,0.01,len(a.label2id),max_it=2)
