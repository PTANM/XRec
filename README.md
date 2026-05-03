### Feature Constrained Generation via Logit Biasing
Repo is based on XRec's initial repo. Documents for the added logit biasing feature are in the FCLB folder.


### Requirements:
Same as original XRec
- Requires **Hugging Face User Access Token** for downloading Llama 2 model

### Steps to running:

1. build_vocab.py
Change variables:
BASE_DIR

Ensure BASE_DIR has:
item_profile.json
user_profile.json
item_vocab.pkl

Run script:
python build_vocab.py

2. infer_biased.py

Change variables: 
base
self.pred_path
self.ref_path

Ensure base dir has:
user_converter.pkl
item_converter.pkl
item_vocab.pkl
predictions.pkl (your own name)
references.pkl (your own name)

Run script: 
python FCLB/infer_biased.py --dataset {dataset} --batch_size {batch size} --mode generate

3. eval_biased.py

Change variables:
base

Ensure base dir has:
tst.pkl
item_vocab.pkl
predictions.pkl (your own name)
references.pkl (your own name)

Run script: 
python FCLB/eval_biased.py --dataset {dataset} --pred {prediction file} --ref {references file}