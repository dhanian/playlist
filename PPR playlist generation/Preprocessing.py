
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

