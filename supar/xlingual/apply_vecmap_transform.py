import supar.xlingual.embeddings as embeddings
from supar.xlingual.cupy_utils import *
import numpy as np

def vecmap_orth(x, w):
    embeddings.normalize(x, ['unit', 'center', 'unit'])
    xw = np.empty_like(x)
    x.dot(w, out=xw)
    return xw

def vecmap(x, W2, s): #src: W2=wx2, trg: W2=wz2
    #xp = get_cupy()
    xp = np
    xp.random.seed(0)
    # STEP 0: Normalize
    x = xp.asarray(x)
    s = xp.asarray(s)
    W2 = xp.asarray(W2)
    embeddings.normalize(x, ['unit', 'center', 'unit'])
    xw = xp.empty_like(x)
    xw[:] = x

    # STEP 1: Whitening
    def whitening_transformation(m):
        u, s, vt = xp.linalg.svd(m, full_matrices=False)
        return vt.T.dot(xp.diag(1/s)).dot(vt)

    #W1 = whitening_transformation(xw)
    #xw = xw.dot(W1)

    # STEP 2: Orthogonal mapping
    xw = xw.dot(W2)

    # STEP 3: Re-weighting
    xw *= s**0.5

    # STEP 4: De-whitening
    #xw = xw.dot(W2.T.dot(xp.linalg.inv(W1)).dot(W2))

    return xw

def vecmap_white(x, W2, s):
    np.random.seed(0)
    # Normalize
    x = np.asarray(x)
    s = np.asarray(s)
    W2 = np.asarray(W2)
    embeddings.normalize(x, ['unit', 'center', 'unit'])
    xw = np.empty_like(x)
    xw[:] = x
    # Whitening
    def whitening(m):
        u,s,vt = np.linalg.svd(m, full_matrices=False)
        return vt.T.dot(np.diag(1/s)).dot(vt)
    W1 = whitening(xw)
    xw = xw.dot(W1)
    # Orthogonal mapping
    xw = xw.dot(W2)
    # Re-weighting
    xw *= s**0.5
    # De-whitening
    xw = xw.dot(W2.T.dot(np.linalg.inv(W1)).dot(W2))

    return xw

def batch_map(batch, W, s):
    tr_vec = []
    for v in batch:
        tr_vec.append(vecmap(np.array([v]), W, s))

    return np.array(tr_vec)

def batch_orth(batch, W):
    tr_vec = []
    for v in batch:
        tr_vec.append(vecmap_orth(np.array([v]), W))
    return np.array(tr_vec)
#def transform(vector, W, s):
#    tr_vec = vecmap(vector, W, s)
#    if lang == 'src':
#        #load wx2, s
#        tr_vec = vecmap(vector, wx2, s)
#    elif lang == 'trg':
#        #load wz2, s
#        tr_vec = vecmap(vector, wz2, s)
#    else:
#        raise ValueError
#
#    return tr_vec
