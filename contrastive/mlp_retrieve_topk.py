"""
Retrieve top-K doc_ids per query using the newly-trained MLP heads.
Outputs {qid: [doc_id1, doc_id2, ..., doc_id10]} per QA file.

Used downstream by the text/cache baseline scripts.
"""
import json, os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, '/home/eftychia/Financial-QA-Benchmark-with-KV-cache')

# Architecture must match the MLP-pool training script
class ProjectionHead(nn.Module):
    """Matches peiwen's MLP-train ProjectionHead exactly."""
    def __init__(self, in_dim, out_dim, num_layers=5):
        super().__init__()
        layers = []
        # First layer: in → out
        layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU(), nn.Dropout(p=0.1)]
        # Hidden layers (num_layers - 2)
        for _ in range(num_layers - 2):
            layers += [nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU(), nn.Dropout(p=0.1)]
        # Final: Linear + LayerNorm (no activation)
        layers += [nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim)]
        self.proj = nn.Sequential(*layers)
    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class PoolMLP(nn.Module):
    def __init__(self, in_dim, num_layers, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, num_layers),
        )
    def forward(self, x):
        return F.softmax(self.mlp(x), dim=-1)


def load_mlp_retriever(heads_path, pooled_kv_path, device='cuda'):
    pooled = torch.load(pooled_kv_path, map_location='cpu', weights_only=False)
    doc_ids = pooled['doc_ids']
    layer_vecs = pooled['layer_vectors']  # list[n_docs] of list[n_layers] of Tensor[hidden]
    if isinstance(layer_vecs, torch.Tensor):
        chunk_layer_matrix = layer_vecs.to(device, dtype=torch.float32)
    elif isinstance(layer_vecs, list) and isinstance(layer_vecs[0], list):
        # list[docs] of list[layers] of Tensor
        chunk_layer_matrix = torch.stack([torch.stack([t for t in doc_layers]) for doc_layers in layer_vecs]).to(device, dtype=torch.float32)
    else:
        chunk_layer_matrix = torch.stack([torch.as_tensor(lv) for lv in layer_vecs]).to(device, dtype=torch.float32)
    # ensure shape (n_docs, n_layers, hidden)
    if chunk_layer_matrix.ndim == 3 and chunk_layer_matrix.shape[0] != len(doc_ids):
        chunk_layer_matrix = chunk_layer_matrix.transpose(0, 1)
    print(f'  chunk_layer_matrix shape: {tuple(chunk_layer_matrix.shape)}')

    heads = torch.load(heads_path, map_location='cpu', weights_only=False)
    meta = heads.get('meta', {})
    proj_dim = meta.get('proj_dim', 1024)
    head_layers = meta.get('head_layers', 5)
    kv_pool_layers = meta.get('kv_pool_layers', 4)
    in_dim = chunk_layer_matrix.shape[-1]

    qh = ProjectionHead(2560, proj_dim, head_layers).to(device)
    ch = ProjectionHead(in_dim, proj_dim, head_layers).to(device)
    pool_mlp = PoolMLP(2560, kv_pool_layers).to(device)
    qh.load_state_dict(heads['question_head'])
    ch.load_state_dict(heads['chunk_head'])
    pool_mlp.load_state_dict(heads['pool_mlp'])
    qh.eval(); ch.eval(); pool_mlp.eval()

    # Pre-select the last kv_pool_layers
    selected_layers = chunk_layer_matrix[:, -kv_pool_layers:, :]  # (n_docs, kv_pool_layers, hidden)
    return doc_ids, selected_layers, qh, ch, pool_mlp, kv_pool_layers


def encode_question(question, tokenizer, model, device):
    tokens = tokenizer(question, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    # mean-pool last hidden state over tokens (excluding padding)
    mask = tokens['attention_mask'].unsqueeze(-1)
    h = out.hidden_states[-1] * mask
    pooled = h.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled  # (1, hidden)


@torch.no_grad()
def retrieve_top_k(question_repr, doc_ids, selected_layers, qh, ch, pool_mlp, top_k=10):
    """question_repr: (1, 2560). Returns list of (doc_id, score) sorted desc."""
    pool_w = pool_mlp(question_repr)  # (1, kv_pool_layers)
    # apply weights to selected_layers per doc: (n_docs, kv_pool_layers, hidden) × (1, kv_pool_layers) → (n_docs, hidden)
    chunk_pooled = torch.einsum('dkh,bk->dh', selected_layers, pool_w)  # (n_docs, hidden)
    q_proj = qh(question_repr)  # (1, proj_dim)
    d_proj = ch(chunk_pooled)  # (n_docs, proj_dim)
    sims = (d_proj @ q_proj.T).squeeze(-1)  # (n_docs,)
    top_scores, top_idx = torch.topk(sims, k=min(top_k, len(doc_ids)))
    return [(doc_ids[i], float(s)) for i, s in zip(top_idx.tolist(), top_scores.tolist())]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--heads', default='/home/eftychia/Financial-QA-Benchmark-with-KV-cache/retrieval_results/mlp_combined_then_merged_v1/20260531-041035/projection_heads.pt')
    p.add_argument('--pooled-kv', default='/home/eftychia/Financial-QA-Benchmark-with-KV-cache/retrieval_results/mlp_combined_then_merged_v1/pooled_kv_with_layers_p_compression.pt')
    p.add_argument('--out-dir', default='/home/eftychia/Financial-QA-Benchmark-with-KV-cache/retrieval_results/mlp_combined_then_merged_v1/topk_for_baselines')
    p.add_argument('--top-k', type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda'
    print(f'Loading retriever (heads={args.heads}, pooled_kv={args.pooled_kv}) ...')
    doc_ids, selected_layers, qh, ch, pool_mlp, kv_pool_layers = load_mlp_retriever(args.heads, args.pooled_kv, device=device)
    print(f'  retriever ready: {len(doc_ids)} docs, kv_pool_layers={kv_pool_layers}')

    print(f'Loading Qwen3-4b for question encoding ...')
    model_name = 'Qwen/Qwen3-4b'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    qa_files = [
        ('chunk_based_qa_VLO_PSX', 'qa/chunk_based_qa_VLO_PSX.json'),
        ('tracking_qa_VLO_PSX', 'qa/tracking_qa_VLO_PSX.json'),
        ('company_comparison_VLO_vs_PSX', 'qa/company_comparison_VLO_vs_PSX.json'),
    ]
    for name, qa_path in qa_files:
        qa = json.load(open(qa_path))
        print(f'\n=== {name}: {len(qa)} QAs ===')
        out_flat = {}
        for i, x in enumerate(qa):
            q_repr = encode_question(x['question'], tokenizer, model, device).to(torch.float32)
            topk = retrieve_top_k(q_repr, doc_ids, selected_layers, qh, ch, pool_mlp, top_k=args.top_k)
            out_flat[x['q_id']] = [d for d, _ in topk]
            if (i+1) % 25 == 0: print(f'  [{i+1}/{len(qa)}]')
        out_path = f'{args.out_dir}/{name}_top{args.top_k}.json'
        json.dump(out_flat, open(out_path, 'w'))
        print(f'  saved → {out_path}')


if __name__ == '__main__':
    main()
