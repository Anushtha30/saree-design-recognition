# Colour-invariant saree design recognition

Pipeline: pretrained timm backbone -> 256-d L2-normalised embedding. Trained with ArcFace (design id as class) +
colour-consistency loss between two colour-randomised views. Retrieval = cosine similarity.

Run order
1. `pip install -r requirements.txt`
2. Put data as `data/<dataset>/<design_id>/<images>` (data is NOT included or redistributed).
3. `python train.py --roots data/deeplure data/kaggle --epochs 15 --out runs/exp1`
4. `python evaluate.py --run runs/exp1`                          # trained model, held-out designs
5. `python evaluate.py --run runs/exp1 --baseline mobilenetv3_large_100`   # frozen ImageNet baseline
6. `python efficiency.py --ckpt runs/exp1/best.pt`
7. `python infer.py search --ckpt runs/exp1/best.pt --query q.jpg --gallery data/deeplure`

Disclosures: timm ImageNet-pretrained backbone; datasets = DeepLure corpus + Kaggle "Indian Saree Patterns".

## Visual demo (no install)
Open `demo/index.html` in a browser (or VS Code "Live Server"). It is an illustration with simulated scores
showing colour-invariant vs colour-sensitive retrieval; real results come from `evaluate.py`.
