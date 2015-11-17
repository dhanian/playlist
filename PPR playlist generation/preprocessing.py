
# coding: utf-8

# ## Generate song index database

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
    for k,v in index.items():
        cur.execute("INSERT INTO Songs VALUES(?,?)", (k,v))
conn.close()

