import json, pickle
import os
from utils.parse import args

run_tag = args.checkpoint_tag or args.adapter_type
pred_path = f"data/{args.dataset}/tst_pred_{run_tag}.pkl"
ref_path = f"data/{args.dataset}/tst_ref_{run_tag}.pkl"

if not os.path.exists(pred_path):
    pred_path = f"data/{args.dataset}/tst_pred.pkl"
if not os.path.exists(ref_path):
    ref_path = f"data/{args.dataset}/tst_ref.pkl"

with open(pred_path, "rb") as f:
    predictions = pickle.load(f)
with open(ref_path, "rb") as f:
    references = pickle.load(f)
    
for i in range(len(predictions)):
    print(f"Prediction: {predictions[i]}")
    print(f"Reference: {references[i]}")
    print("-" * 50)
    if i == 5:
        break
