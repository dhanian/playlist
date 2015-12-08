# coding: utf-8

# ## Generate song index database, and Community partition dictionary

# ### Get all tid of all songs

# In[ ]:

import os
import sys
import sqlite3

nodes = set()
conn = sqlite3.connect("lastfm_similars.db")
with conn:
    cur = conn.cursor()
    cur.execute("SELECT tid, target FROM similars_src")
    while True:
        song = cur.fetchone()
        if not song:
            break
        nodes.add(song[0])
        similars = song[1].split(",")
        for i in range(0,len(similars),2):
            nodes.add(similars[i])
            #edges[(song[0],similars[i])] = float(similars[i+1])
conn.close()
nodes = list(nodes)


# ### Store the song tid index in a new database 

# In[ ]:

N = len(nodes)
index = dict(zip(nodes,range(N)))
conn = sqlite3.connect("songs_index.db")

with conn:
    cur =  conn.cursor()
    cur.execute("DROP TABLE IF EXISTS Songs")
    cur.execute("CREATE TABLE Songs(tid TEXT, idx INT)")
    for k,v in index.iteritems():
        cur.execute("INSERT INTO Songs VALUES(?,?)", (k,v))
conn.close()


# ### Read edge index and write it into a text file for community detection

# In[ ]:

conn = sqlite3.connect("lastfm_similars.db")
row,col,data=[],[],[]
outFile = open('edges.txt','w')

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
            if float(similars[i+1]) >= 0.5:
                ss = '%d,%d\n' % (row_idx,index[similars[i]])
                outFile.write(ss)
conn.close()
outFile.close()


# ### Community Detection using Louvain Method

# In[1]:

import community
import networkx as nx

G = nx.Graph()
G.add_nodes_from(range(663234))

inFile = open('edges.txt')
for line in inFile:
    fields = line.strip().split(',')
    G.add_edge(int(fields[0]),int(fields[1]))
inFile.close()

# Compute the best partition
partition = community.best_partition(G)
# Store the partition dictionary to file
outFile = open('partition.txt','w')
for nid,com in partition.iteritems():
    ss = '%d,%d\n' % (nid,com)
    outFile.write(ss)
outFile.close()


# #### Number of communities

# In[2]:

len(set(partition.values()))


# #### Modularity of the partition

# In[3]:

community.modularity(partition,G)

# coding: utf-8

# In[1]:

import os
import scipy.sparse
import sys
import sqlite3
import time


# In[2]:

print "Initializing program, please wait..."


# ### Read song tid index into a dictionary from database

# In[3]:

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


# ### Read community partition dictionary

# In[30]:

partition, comm = {}, {}
inFile = open('partition.txt')
for line in inFile:
    fields = line.strip().split(',')
    partition[int(fields[0])] = int(fields[1])
inFile.close()
    
for nid,com in partition.iteritems():
    if com not in comm:
        comm[com] = []
    comm[com].append(nid)


# ### Read edge index and weight data (row,col,data) into list for coo_matrix generation 

# In[8]:

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

# In[9]:

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

# In[77]:

# function to calculate the personalized pagerank by power iteration method using scipy's sparse matrix implementation
def PPR(index,mtx,sinking,v=None,alpha=0.85,max_iter=100, tol=1e-6):

    """
    [Parameters]
    index: dictionary. tid-index pair of songs
    mtx: scipy.sparse.csr.csr_matrix. The transition matrix in scipy csr format.
    sinking: numpy.ndarray. Index of songs that has no out edges
    seed: list. Seed song indexes

    [Return type]
    dictionary. songs tid - score pairs
    """
    N = len(index)
    
    # starting rank
    x = scipy.repeat(1./N, N)
    
    # personalization vector 
    if v is None:
        v = scipy.repeat(1./N, N)
    v /= v.sum()

    #power iteration:
    for _ in xrange(max_iter):
        xlast = x
        x = alpha*(x*mtx + sum(x[sinking])*v) + (1-alpha)*v
        if scipy.absolute(x-xlast).sum() < tol:
            #nodes = sorted(index, key=index.get, reverse=False)
            #return dict(zip(nodes,x))
            scores = {}
            for key,value in index.iteritems():
                scores[key] = x[value]
            return scores
    raise RuntimeError('Power iteration failed to converge in %d iterations.' % max_iter)


# ### Map track id to song metadata

# In[75]:

meta = {}
inFile = open('unique_tracks.txt')
for line in inFile:
    fields = line.strip().split('<SEP>')
    meta[fields[0]] = fields[2] + ': ' + fields[3]
inFile.close()


# ### Generate playlist from a subset of all the songs

# In[100]:

seed = raw_input("Pick some songs to start your playlist: ")
#seed = map(int,seed.split(","))

seed_raw = seed.strip().split(";")
seed = []
for tid,song in meta.iteritems():
    for track in seed_raw:
        if song == track:
            seed.append(index[tid])
if len(seed) != len(seed_raw):
    print "No such song(s)!"
discover_rate = float(input("Pick discovery rate from 0 to 1: "))
listLength = int(input("Playlist length: "))
 
t0 = time.time()
v = scipy.repeat(discover_rate*0.01/float(N),N)
for track in seed:
    for song in comm[partition[track]]:
        v[song] = 1./N
for track in seed:
    #v[track] = (N-len(seed)) / float(N) / len(seed)
    v[track] = len(comm[partition[track]])/float(N)
          
rank = PPR(index,mtx,sinking,v)
playlist = sorted(rank, key=rank.get, reverse=True)
t1 = time.time()
scores = [rank[i] for i in playlist]
print "Playlist generated in %.3f seconds" % (t1-t0)
print "Playlist(showing first 20 only)"
uniList = []
i,j = 0,0
while i < listLength:
    song = meta[playlist[j]]
    j += 1
    if song not in uniList:
        i += 1
        uniList.append(song)
        print song

