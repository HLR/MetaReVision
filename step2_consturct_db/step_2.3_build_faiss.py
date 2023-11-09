"""
build faiss

we can refer:
1. chatgpt
2. knnLM: build_index

output:
element_cpt.indexed
"""
import torch
import numpy as np
import faiss
from ProjUtils.Constant import MSCOCODataStore, MSCOCOIndex

# create index
dim = 384
index = faiss.IndexFlatL2(dim)

# load datastore
data_store = torch.load(MSCOCODataStore).cpu().numpy()

# add vec to index
index.add(data_store)

# save the index
faiss.write_index(index, MSCOCOIndex)