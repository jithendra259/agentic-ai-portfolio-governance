import numpy as np
import pandas as pd
from pymongo import MongoClient
import os

client14 = MongoClient("mongodb+srv://kandulajithendrasubramanyam_db_user:6TzH8AQkMVvirJhv@cluster0.yqwhfm6.mongodb.net/?retryWrites=true&w=majority", 
                       tls=True, tlsAllowInvalidCertificates=True)
db14 = client14["Stock_data"]

universe_id = "U1"
bb_docs = list(db14["blackboard_mpi"].find(
    {"universe_id": universe_id},
    sort=[("window_number", 1)]
))

print(f"Total windows: {len(bb_docs)}")
I_t_list = []
lambda_t_list = []
for doc in bb_docs:
    I_t = doc.get("I_t")
    lam = doc.get("lambda_t")
    I_t_list.append(float(I_t) if I_t is not None else 0.0)
    lambda_t_list.append(float(lam) if lam is not None else 0.0)

I_arr = np.array(I_t_list)
crisis_count  = int((I_arr >= 0.85).sum())
elevated_count = int(((I_arr >= 0.50) & (I_arr < 0.85)).sum())
calm_count    = int((I_arr < 0.50).sum())

print(f"Calm: {calm_count}, Elevated: {elevated_count}, Crisis: {crisis_count}")

# Let's also check regime_transitions
rt_docs = list(db14["regime_transitions"].find({"universe_id": universe_id}))
rt_I_arr = np.array([float(d.get("I_t", 0.0)) for d in rt_docs])
print(f"Regime Transitions - Calm: {int((rt_I_arr < 0.50).sum())}, Elevated: {int(((rt_I_arr >= 0.50) & (rt_I_arr < 0.85)).sum())}, Crisis: {int((rt_I_arr >= 0.85).sum())}")
