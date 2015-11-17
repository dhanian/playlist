
# coding: utf-8

# In[ ]:

import os
import scipy.sparse
import sys
import sqlite3
import time


# In[ ]:

print "Initializing program, please wait..."


# ### Read song tid index into a dictionary from database

# In[ ]:

index = {}
conn = sqlite3.connect("songs_index.db")
with conn:
    cur = conn.cursor()
    cur.execute("SELECT tid, idx FROM Songs")
    while True:
        song = cur.fetchone()
        if not song:
            break
        index[song[0]]=song[1]
conn.close()


# ### Read edge index and weight data (row,col,data) into list for coo_matrix generation 

# In[ ]:

conn = sqlite3.connect("lastfm_similars.db")
row,col,data=[],[],[]

with conn:
    cur = conn.cursor()
    cur.execute("SELECT tid, target FROM similars_src")
    while True:
        song = cur.fetchone()
        if not song:
            break
        similars = song[1].split(",")
        row_idx = index[song[0]]
        for i in range(0,len(similars),2):
            row.append(row_idx)
            col.append(index[similars[i]])
            data.append(float(similars[i+1]))
conn.close()


# ### Calculate the scipy csr format of the transition matrix

# In[ ]:

# number of nodes
N = len(index)
# calculate the graph adjacency matrix as a scipy sparse matrix
mtx = scipy.sparse.coo_matrix((data,(row,col)),shape=(N,N))
compress = "csr"
mtx = mtx.asformat(compress)
  
# normalize the matrix 
rowSum = scipy.array(mtx.sum(axis=1)).flatten()
rowSum[rowSum != 0] = 1./rowSum[rowSum != 0]
invDiag = scipy.sparse.spdiags(rowSum.T, 0, N, N, format=compress)
mtx = invDiag * mtx
# identify sinking nodes index
sinking = scipy.where(rowSum == 0)[0]


# ### Personalized PageRank function

# In[ ]:

# function to calculate the personalized pagerank by power iteration method using scipy's sparse matrix implementation
def PPR(index,mtx,sinking,alpha=0.85, seed=None ,max_iter=100, tol=1e-6):

    """
    [Parameters]
    index: dictionary. tid-index pair of songs
    mtx: scipy.sparse.csr.csr_matrix. The transition matrix in scipy csr format.
    sinking: numpy.ndarray. Index of songs that has no out edges
    seed: list. Seed song indexes

    [Return type]
    dictionary. songs tid - score pairs
    """
    # starting rank
    x = scipy.repeat(1./N, N)
    
    # personalization vector 
    if not seed:
        v = scipy.repeat(1./N, N)
    else:
        v = scipy.zeros(N)
        v[seed] = 1
        v /= v.sum()

    #power iteration:
    for _ in xrange(max_iter):
        xlast = x
        x = alpha*(x*mtx + sum(x[sinking])*v) + (1-alpha)*v
        if scipy.absolute(x-xlast).sum() < tol:
            #nodes = sorted(index, key=index.get, reverse=False)
            #return dict(zip(nodes,x))
            scores = {}
            for k,v in index.items():
                scores[k] = x[v]
            return scores
    raise RuntimeError('Power iteration failed to converge in %d iterations.' % max_iter)


# ### Generate playlist from a subset of all the songs

# In[ ]:

seed = raw_input("Pick some songs to start your playlist:")
seed = map(int,seed.split(","))
listLength = 20
t0 = time.time()
rank = PPR(index,mtx,sinking,seed=seed)
playlist = sorted(rank, key=rank.get, reverse=True)
t1 = time.time()
scores = [rank[i] for i in playlist]
print "Playlist generated in %.3f seconds" % (t1-t0)
print "Playlist(showing first 20 only)"
print playlist[:listLength]

