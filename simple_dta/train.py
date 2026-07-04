"""Train + evaluate a simplified affinity-prediction model (CNN or GNN).

Usage:
    python train.py --model cnn --dataset davis --epochs 100
    python train.py --model gnn --dataset davis --epochs 100

Both models predict affinity only (single MSE objective) -- none of
DeepDTAGen's generation machinery (VAE / Transformer decoder / FetterGrad).
Reports MSE / RMSE / CI / rm2 / Pearson / Spearman and per-epoch wall time.
"""
import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from metrics import all_metrics
from models import CNNDTA, GNNDTA, count_params

SEED = 4221


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(model_kind, dataset, batch_size, featurizer="full"):
    if model_kind == "cnn":
        from torch.utils.data import DataLoader
        from data import CNNDataset
        train_ds = CNNDataset(dataset, "train")
        test_ds = CNNDataset(dataset, "test")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        from torch_geometric.loader import DataLoader as GeoLoader
        from data import build_graph_dataset
        print(f"Building molecular graphs (in-the-clear encoding, featurizer={featurizer})...")
        train_list = build_graph_dataset(dataset, "train", featurizer=featurizer)
        test_list = build_graph_dataset(dataset, "test", featurizer=featurizer)
        train_loader = GeoLoader(train_list, batch_size=batch_size, shuffle=True)
        test_loader = GeoLoader(test_list, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_device(batch, model_kind, device):
    if model_kind == "cnn":
        xd, xt, y = batch
        return (xd.to(device), xt.to(device), None), y.to(device)
    batch = batch.to(device)
    return batch, batch.y.view(-1)


def evaluate(model, loader, model_kind, device, dataset=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            inp, y = to_device(batch, model_kind, device)
            preds.append(model(inp).cpu().numpy())
            trues.append(y.cpu().numpy())
    P = np.concatenate(preds).flatten()
    G = np.concatenate(trues).flatten()
    return all_metrics(G, P, dataset=dataset), G, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "gnn"], required=True)
    ap.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pool", choices=["max", "mean", "maxmean"], default="max",
                    help="global pooling; mean is MPC-friendly, max matches baseline, "
                         "maxmean concatenates both (doubles tower out dim)")
    ap.add_argument("--drug-filters", type=int, default=32,
                    help="CNN: base filter count for the drug (SMILES) tower (f, 2f, 3f)")
    ap.add_argument("--prot-filters", type=int, default=32,
                    help="CNN: base filter count for the protein tower (f, 2f, 3f)")
    ap.add_argument("--proj-dim", type=int, default=0,
                    help="CNN: if >0, project each tower to this dim before the head "
                         "(compact joint binding embedding); 0 = concat raw tower outputs")
    ap.add_argument("--drug-channels", default="",
                    help="CNN: comma list of drug conv out-channels per layer "
                         "(sets depth AND shape); overrides --drug-filters. e.g. 16,32")
    ap.add_argument("--prot-channels", default="",
                    help="CNN: comma list of protein conv out-channels per layer; "
                         "overrides --prot-filters. e.g. 32,64,96,128")
    ap.add_argument("--drug-dilations", default="",
                    help="CNN: comma list of drug conv dilations, one per layer. e.g. 1,2,4")
    ap.add_argument("--drug-kernel", type=int, default=4,
                    help="conv kernel size for the drug/SMILES tower (default 4)")
    ap.add_argument("--prot-kernel", type=int, default=8,
                    help="conv kernel size for the protein tower (default 8)")
    ap.add_argument("--prot-dilations", default="",
                    help="CNN: comma list of protein conv dilations, one per layer. e.g. 1,2,4")
    ap.add_argument("--head-dim", type=int, default=1024,
                    help="width of first FC head layer (1024=baseline, 512=reduced)")
    ap.add_argument("--head-layers", type=int, choices=[1, 2], default=2,
                    help="number of hidden FC layers in head (2=baseline, 1=drop 2nd)")
    ap.add_argument("--node-feat", choices=["full", "small", "tiny"], default="full",
                    help="GNN atom-feature width: full(94)|small(12)|tiny(4)")
    ap.add_argument("--gcn-dim", type=int, default=128,
                    help="GNN: GCN hidden width per layer")
    ap.add_argument("--gcn-layers", type=int, default=2,
                    help="GNN: number of GCNConv layers (DeepDTAGen uses 3)")
    ap.add_argument("--embed-dim", type=int, default=128,
                    help="protein (and CNN drug) embedding dim; smaller = cheaper in MPC")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout prob in the prediction head (regularization)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam L2 weight decay (regularization)")
    ap.add_argument("--select-metric", choices=["mse", "balacc"], default="mse",
                    help="which metric picks the best checkpoint / dumped preds: "
                         "mse (minimize, default) or balacc (maximize balanced-acc mean)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed; vary across runs to average over inits")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the run tag so parallel runs don't collide")
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--cuda", type=int, default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if args.cuda is not None
                          and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model}_{args.dataset}{args.tag_suffix}"
    gnn_cfg = (f" | node_feat={args.node_feat} gcn_dim={args.gcn_dim} "
               f"gcn_layers={args.gcn_layers} embed_dim={args.embed_dim}"
               if args.model == "gnn" else f" | embed_dim={args.embed_dim}")
    print(f"=== {tag} | device={device} | pool={args.pool} | head_dim={args.head_dim} "
          f"| head_layers={args.head_layers} | seed={args.seed}{gnn_cfg} ===")

    train_loader, test_loader = build_loaders(args.model, args.dataset,
                                              args.batch_size, featurizer=args.node_feat)

    if args.model == "cnn":
        _il = lambda s: [int(x) for x in s.split(",")] if s else None
        model = CNNDTA(embed_dim=args.embed_dim, pool=args.pool, head_dim=args.head_dim,
                       head_layers=args.head_layers, drug_filters=args.drug_filters,
                       prot_filters=args.prot_filters, proj_dim=args.proj_dim,
                       dropout=args.dropout,
                       drug_channels=_il(args.drug_channels),
                       prot_channels=_il(args.prot_channels),
                       drug_dilations=_il(args.drug_dilations),
                       prot_dilations=_il(args.prot_dilations),
                       drug_kernel=args.drug_kernel,
                       prot_kernel=args.prot_kernel).to(device)
    else:
        from data import FEATURIZERS
        node_feat_dim = FEATURIZERS[args.node_feat][1]
        model = GNNDTA(embed_dim=args.embed_dim, gcn_dim=args.gcn_dim,
                       gcn_layers=args.gcn_layers, node_feat_dim=node_feat_dim,
                       pool=args.pool, head_dim=args.head_dim,
                       head_layers=args.head_layers, dropout=args.dropout).to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    # Which metric selects the saved checkpoint: MSE (minimize) or balacc_mean (maximize).
    sel_key = "balacc_mean" if args.select_metric == "balacc" else "MSE"
    sel_minimize = (sel_key == "MSE")
    best = {sel_key: float("inf") if sel_minimize else float("-inf")}
    history = []
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for batch in train_loader:
            inp, y = to_device(batch, args.model, device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            n += len(y)
        dt = time.time() - t0
        epoch_times.append(dt)
        train_mse = running / n

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            m, G, P = evaluate(model, test_loader, args.model, device,
                               dataset=args.dataset)
            history.append({"epoch": epoch, "train_mse": train_mse, **m})
            print(f"[{tag}] epoch {epoch:3d} | {dt:5.1f}s | train_mse {train_mse:.4f} "
                  f"| test MSE {m['MSE']:.4f} CI {m['CI']:.4f} rm2 {m['rm2']:.4f} "
                  f"Pearson {m['Pearson']:.4f} balAcc {m.get('balacc_mean', float('nan')):.4f}")
            cur = m.get(sel_key)
            improved = (cur is not None and not np.isnan(cur) and
                        (cur < best[sel_key] if sel_minimize else cur > best[sel_key]))
            if improved:
                best = {"epoch": epoch, **m}
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"{tag}_best.pth"))
                np.savetxt(os.path.join(args.out_dir, f"{tag}_pred.txt"), P)
                np.savetxt(os.path.join(args.out_dir, f"{tag}_true.txt"), G)

    summary = {
        "tag": tag, "model": args.model, "dataset": args.dataset,
        "epochs": args.epochs, "params": count_params(model),
        "mean_epoch_sec": float(np.mean(epoch_times)),
        "total_train_sec": float(np.sum(epoch_times)),
        "best": best, "pool": args.pool, "head_dim": args.head_dim,
        "head_layers": args.head_layers, "seed": args.seed,
        "node_feat": args.node_feat, "gcn_dim": args.gcn_dim,
        "gcn_layers": args.gcn_layers, "embed_dim": args.embed_dim,
        "drug_filters": args.drug_filters, "prot_filters": args.prot_filters,
        "proj_dim": args.proj_dim, "drug_channels": args.drug_channels,
        "prot_channels": args.prot_channels, "drug_dilations": args.drug_dilations,
        "prot_dilations": args.prot_dilations,
        "drug_kernel": args.drug_kernel, "prot_kernel": args.prot_kernel,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay, "select_metric": args.select_metric,
        "history": history,
    }
    with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BEST [{tag}] === epoch {best.get('epoch')} | "
          f"MSE {best['MSE']:.4f} | CI {best['CI']:.4f} | rm2 {best['rm2']:.4f}")
    print(f"params {summary['params']:,} | mean {summary['mean_epoch_sec']:.1f}s/epoch "
          f"| total {summary['total_train_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
