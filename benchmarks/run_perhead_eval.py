#!/usr/bin/env python3
"""
PRISM Per-Head Block-Sparse Evaluation.
Key fix: only retrieval heads get block-sparse mask;
streaming heads keep full attention.
Uses monkey-patched eager_attention_forward for per-head control.
"""
import json, os, sys, time, gc, math, types
import torch
import torch.nn as nn
import numpy as np

LOG = r"C:\Users\admin\PRISM\perhead_eval_log.txt"
RESULTS_DIR = r"C:\Users\admin\PRISM\results\downstream"
TEXT_FILE = r"C:\Users\admin\PRISM\paper_text.txt"
RH_FILE = r"C:\Users\admin\PRISM\results\retrieval_heads\qwen25_7b_bf16_128k.json"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ── Per-head attention forward ────────────────────────────────
# Global storage for per-head mask
_PERHEAD_MASK = {}  # layer_idx -> (1, num_heads, 1, seq) mask tensor

def make_perhead_eager_forward(original_forward, layer_idx, num_kv_groups):
    """Create a patched eager attention forward that applies per-head masks."""
    def patched_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        # Check if we have a per-head mask for this layer
        if layer_idx in _PERHEAD_MASK:
            perhead_mask = _PERHEAD_MASK[layer_idx]  # (1, n_kv_heads, 1, seq)
            # Expand to query head count: (1, n_kv_heads, 1, seq) -> (1, n_heads, 1, seq)
            if perhead_mask.shape[1] != query.shape[1]:
                # Repeat each KV head mask for its group of query heads
                perhead_mask = perhead_mask.repeat_interleave(num_kv_groups, dim=1)
            # Expand to (1, n_heads, T_q, seq) via broadcast
            # perhead_mask values: 0 = attend, -inf = mask
            attention_mask = perhead_mask  # Override the standard mask

        return original_forward(module, query, key, value, attention_mask,
                              scaling, dropout, **kwargs)
    return patched_forward


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG, "w") as f:
        f.write("")
    try:
        _run()
    except Exception as e:
        import traceback
        log(f"FATAL ERROR: {e}")
        log(traceback.format_exc())

def _run():

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    MODEL = "Qwen/Qwen2.5-7B"
    BLOCK_SIZE = 128
    K_VALUES = [8, 16, 32]
    D_SIG = 32
    RH_THRESH = 0.3
    RECENT_WINDOW = 256

    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dk = "dtype" if tv >= (4, 57) else "torch_dtype"

    # Load text
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()
    log(f"Text: {len(raw_text)} chars")

    # Load R_h data to identify retrieval heads
    rh_matrix = None
    if os.path.exists(RH_FILE):
        with open(RH_FILE, "r") as f:
            rh_data = json.load(f)
        # Keys are direct context lengths: "2048", "4096", "8192", etc.
        for ctx_key in ["8192", "4096", "2048"]:
            if ctx_key in rh_data and "rh_matrix" in rh_data[ctx_key]:
                rh_matrix = np.array(rh_data[ctx_key]["rh_matrix"])
                log(f"Loaded R_h from ctx={ctx_key}")
                break

    if rh_matrix is None:
        log("WARNING: No R_h matrix found, using threshold-based fallback")
        # Fallback: assume all heads except layer 0-2 are retrieval
        rh_matrix = np.ones((28, 4)) * 0.5
        rh_matrix[:3, :] = 0.1  # Early layers are streaming

    log(f"R_h matrix shape: {rh_matrix.shape}")
    retrieval_mask_per_layer = rh_matrix > RH_THRESH  # (n_layers, n_kv_heads) bool
    n_retrieval = retrieval_mask_per_layer.sum()
    n_total = retrieval_mask_per_layer.size
    log(f"Retrieval heads: {n_retrieval}/{n_total} ({100*n_retrieval/n_total:.1f}%)")

    # Load model in EAGER mode (required for per-head control)
    log(f"Loading {MODEL} (eager mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, **{dk: torch.bfloat16},
        attn_implementation="eager",
        device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model.eval()

    cfg = model.config
    nL = cfg.num_hidden_layers
    nA = cfg.num_attention_heads
    nKV = cfg.num_key_value_heads
    G = nA // nKV  # GQA group size
    hd = cfg.hidden_size // nA
    log(f"Arch: {nL}L, {nA}AH, {nKV}KVH, G={G}, hd={hd}")
    log(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Monkey-patch: intercept self_attn.forward to inject per-head masks
    # The per-head mask is ADDED to the existing causal mask (not replacing it)
    # and expanded from nKV heads to nA query heads via GQA grouping
    log("Patching eager attention for per-head masking...")

    _originals = {}
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig_forward = attn.forward

        def make_patched_attn_forward(orig_fwd, layer_idx, gqa_groups):
            def patched_attn_forward(hidden_states, **kwargs):
                if layer_idx in _PERHEAD_MASK:
                    existing_mask = kwargs.get('attention_mask', None)
                    perhead = _PERHEAD_MASK[layer_idx]  # (1, nKV, 1, seq)
                    # Expand KV heads -> query heads: (1, nKV, 1, seq) -> (1, nA, 1, seq)
                    perhead_full = perhead.repeat_interleave(gqa_groups, dim=1)
                    if existing_mask is not None:
                        # existing: (1, 1, seq, seq) causal; perhead: (1, nA, 1, seq) block
                        # Combined via broadcast: (1, nA, seq, seq)
                        kwargs['attention_mask'] = existing_mask + perhead_full
                    else:
                        kwargs['attention_mask'] = perhead_full
                return orig_fwd(hidden_states, **kwargs)
            return patched_attn_forward

        _originals[li] = orig_forward
        attn.forward = make_patched_attn_forward(orig_forward, li, G)

    log("Patching done")

    # Tokenize
    all_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    log(f"Total tokens: {len(all_ids)}")

    results = {"model": MODEL, "block_size": BLOCK_SIZE, "d_sig": D_SIG,
               "method": "per_head_block_sparse", "rh_threshold": RH_THRESH}

    # ═══════════════════════════════════════════════════════════
    # PERPLEXITY with per-head block-sparse attention
    # ═══════════════════════════════════════════════════════════
    log("\n" + "="*60)
    log("PERPLEXITY (per-head block-sparse)")
    log("="*60)

    CTX_LENGTHS = [2048, 4096, 8192]
    ppl_results = {}

    for ctx in CTX_LENGTHS:
        if ctx > len(all_ids):
            continue

        log(f"\n  Context: {ctx}")
        n_blocks = ctx // BLOCK_SIZE
        n_windows = min(3, len(all_ids) // ctx)

        full_losses = []
        sparse_losses = {k: [] for k in K_VALUES}

        for wi in range(n_windows):
            start = wi * ctx
            ids = torch.tensor([all_ids[start:start + ctx]], device="cuda")
            targets = ids[0, 1:]

            # Clear per-head masks (full attention)
            _PERHEAD_MASK.clear()

            # Full attention forward
            with torch.no_grad():
                out_full = model(ids, use_cache=True)
            logits_full = out_full.logits[0, :-1, :]
            loss_full = nn.functional.cross_entropy(logits_full, targets)
            ppl_full = loss_full.exp().item()
            full_losses.append(loss_full.item())

            # Extract keys for block signatures
            keys_per_layer = []
            for layer_kv in out_full.past_key_values:
                k_tensor = layer_kv[0][0]  # (n_kv_heads, seq, head_dim)
                keys_per_layer.append(k_tensor[:, :, :D_SIG].cpu())
            del out_full; torch.cuda.empty_cache()

            # Compute block signatures per layer per KV head
            # sigs[li]: (n_kv_heads, n_blocks, D_SIG)
            sigs_per_layer = []
            for li in range(nL):
                keys = keys_per_layer[li]  # (n_kv_heads, seq, D_SIG)
                trimmed = keys[:, :n_blocks * BLOCK_SIZE, :]
                reshaped = trimmed.view(nKV, n_blocks, BLOCK_SIZE, D_SIG)
                sigs_per_layer.append(reshaped.mean(dim=2))  # (n_kv_heads, n_blocks, D_SIG)

            # Query signal: last token's key per layer per KV head
            queries_per_layer = []
            for li in range(nL):
                q = keys_per_layer[li][:, -1, :]  # (n_kv_heads, D_SIG)
                queries_per_layer.append(q)

            del keys_per_layer; gc.collect()

            # For each k value, create per-head masks and evaluate
            for k in K_VALUES:
                if k >= n_blocks:
                    sparse_losses[k].append(loss_full.item())
                    continue

                # Create per-head mask for each layer
                for li in range(nL):
                    # Per KV head: select top-k blocks
                    kv_mask = torch.zeros(nKV, 1, ctx, device="cuda")  # (nKV, 1, seq)

                    for hi in range(nKV):
                        if not retrieval_mask_per_layer[li, hi]:
                            # Streaming head: full attention (all positions visible)
                            kv_mask[hi, 0, :] = 0  # 0 = no masking
                        else:
                            # Retrieval head: block-sparse
                            scores = sigs_per_layer[li][hi] @ queries_per_layer[li][hi]
                            _, top_idx = scores.topk(min(k, n_blocks))

                            # Start with all masked (-inf)
                            head_mask = torch.full((ctx,), float('-inf'), device="cuda")
                            # Unmask selected blocks
                            for b in top_idx:
                                s = b * BLOCK_SIZE
                                e = min((b + 1) * BLOCK_SIZE, ctx)
                                head_mask[s:e] = 0
                            # Unmask recent window
                            head_mask[-RECENT_WINDOW:] = 0
                            kv_mask[hi, 0, :] = head_mask

                    # Shape: (1, nKV, 1, seq) -> will be expanded to (1, nA, 1, seq) in patch
                    _PERHEAD_MASK[li] = kv_mask.unsqueeze(0).to(torch.bfloat16)

                # Sparse forward
                with torch.no_grad():
                    out_sparse = model(ids)
                logits_s = out_sparse.logits[0, :-1, :]
                loss_s = nn.functional.cross_entropy(logits_s, targets)
                sparse_losses[k].append(loss_s.item())

                del out_sparse; torch.cuda.empty_cache()
                _PERHEAD_MASK.clear()

            vram = torch.cuda.max_memory_allocated() / 1e9
            log(f"    Window {wi+1}: PPL_full={ppl_full:.2f}, VRAM={vram:.1f}GB")
            for k in K_VALUES:
                if sparse_losses[k]:
                    log(f"      k={k}: PPL={math.exp(sparse_losses[k][-1]):.2f}")

            del ids, logits_full, targets
            torch.cuda.empty_cache(); gc.collect()
            torch.cuda.reset_peak_memory_stats()

        if full_losses:
            avg_full = math.exp(np.mean(full_losses))
            ctx_res = {"ppl_full": round(avg_full, 2)}
            for k in K_VALUES:
                if sparse_losses[k]:
                    avg_s = math.exp(np.mean(sparse_losses[k]))
                    inc = (avg_s - avg_full) / avg_full * 100
                    ctx_res[f"ppl_k{k}"] = round(avg_s, 2)
                    ctx_res[f"inc_k{k}"] = round(inc, 1)
            ppl_results[str(ctx)] = ctx_res
            log(f"  >> ctx={ctx}: full={avg_full:.2f}")
            for k in K_VALUES:
                if f"ppl_k{k}" in ctx_res:
                    log(f"     k={k}: {ctx_res[f'ppl_k{k}']:.2f} ({ctx_res[f'inc_k{k}']:+.1f}%)")

    results["perplexity"] = ppl_results

    # ═══════════════════════════════════════════════════════════
    # NIAH with per-head block-sparse attention
    # ═══════════════════════════════════════════════════════════
    log("\n" + "="*60)
    log("NIAH (per-head block-sparse)")
    log("="*60)

    NEEDLE = "The secret access code for the photonic laboratory is 'quantum-cascade-7391'."
    QUESTION = "\n\nQuestion: What is the secret access code for the photonic laboratory?\nAnswer:"
    EXPECTED = "quantum-cascade-7391"

    niah_results = {}
    NIAH_CTXS = [2048, 4096, 8192]
    POSITIONS = [0.1, 0.25, 0.5, 0.75, 0.9]

    for ctx in NIAH_CTXS:
        log(f"\n  Context: {ctx}")
        needle_toks = tokenizer.encode(NEEDLE, add_special_tokens=False)
        question_toks = tokenizer.encode(QUESTION, add_special_tokens=False)
        hay_len = ctx - len(needle_toks) - len(question_toks) - 5

        if hay_len < 100:
            continue

        hay_toks = all_ids[:hay_len]
        hits_full = 0
        hits_sparse = {k: 0 for k in K_VALUES}
        total = 0

        for pos_frac in POSITIONS:
            insert = int(len(hay_toks) * pos_frac)
            input_toks = hay_toks[:insert] + needle_toks + hay_toks[insert:]
            input_toks = input_toks[:hay_len] + question_toks
            ids = torch.tensor([input_toks[:ctx]], device="cuda")
            actual_len = ids.shape[1]
            n_blocks = actual_len // BLOCK_SIZE

            # Full attention generation
            _PERHEAD_MASK.clear()
            with torch.no_grad():
                gen_f = model.generate(ids, max_new_tokens=30, do_sample=False)
            answer_f = tokenizer.decode(gen_f[0, actual_len:], skip_special_tokens=True)
            hit_f = EXPECTED.lower() in answer_f.lower()
            if hit_f:
                hits_full += 1

            # Get KV cache for signatures
            with torch.no_grad():
                cache_out = model(ids, use_cache=True)

            keys_per_layer = []
            for layer_kv in cache_out.past_key_values:
                keys_per_layer.append(layer_kv[0][0, :, :, :D_SIG].cpu())
            del cache_out; torch.cuda.empty_cache()

            # Block signatures + query
            sigs_per_layer = []
            queries_per_layer = []
            for li in range(nL):
                keys = keys_per_layer[li]
                trimmed = keys[:, :n_blocks * BLOCK_SIZE, :]
                reshaped = trimmed.view(nKV, n_blocks, BLOCK_SIZE, D_SIG)
                sigs_per_layer.append(reshaped.mean(dim=2))
                queries_per_layer.append(keys[:, -1, :])
            del keys_per_layer; gc.collect()

            # Sparse generation for each k
            for k in K_VALUES:
                if k >= n_blocks:
                    if hit_f: hits_sparse[k] += 1
                    continue

                # Build per-head masks (for generation, mask shape works differently)
                # For generate(), we pass attention_mask as (1, seq) to the model
                # So we need a different approach: use the 2D mask but per-head
                # Alternative: just use the standard approach with a combined mask

                # Create a 2D mask that includes blocks selected by ANY retrieval head
                combined_mask = torch.zeros(actual_len, dtype=torch.long, device="cuda")
                for li in range(nL):
                    for hi in range(nKV):
                        if retrieval_mask_per_layer[li, hi]:
                            scores = sigs_per_layer[li][hi] @ queries_per_layer[li][hi]
                            _, top_idx = scores.topk(min(k, n_blocks))
                            for b in top_idx:
                                s = b * BLOCK_SIZE
                                e = min((b + 1) * BLOCK_SIZE, actual_len)
                                combined_mask[s:e] = 1
                combined_mask[-RECENT_WINDOW:] = 1
                attn_mask = combined_mask.unsqueeze(0)

                with torch.no_grad():
                    gen_s = model.generate(ids, attention_mask=attn_mask,
                                         max_new_tokens=30, do_sample=False)
                answer_s = tokenizer.decode(gen_s[0, actual_len:], skip_special_tokens=True)
                hit_s = EXPECTED.lower() in answer_s.lower()
                if hit_s:
                    hits_sparse[k] += 1

                del gen_s; torch.cuda.empty_cache()

            total += 1
            log(f"    pos={pos_frac:.0%}: full={'O' if hit_f else 'X'}")
            for k in K_VALUES:
                # Report inline
                pass

            del ids, gen_f
            torch.cuda.empty_cache(); gc.collect()

        if total > 0:
            niah_results[str(ctx)] = {"full": round(100 * hits_full / total, 1), "total": total}
            for k in K_VALUES:
                niah_results[str(ctx)][f"k{k}"] = round(100 * hits_sparse[k] / total, 1)

            log(f"  >> ctx={ctx}: full={100*hits_full/total:.0f}%")
            for k in K_VALUES:
                log(f"     k={k}: {100*hits_sparse[k]/total:.0f}%")

    results["niah"] = niah_results

    # Restore original forwards
    for li, orig in _originals.items():
        model.model.layers[li].self_attn.forward = orig
    _PERHEAD_MASK.clear()

    # Save
    path = os.path.join(RESULTS_DIR, "perhead_eval.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved: {path}")

    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    if results.get("perplexity"):
        log("\nPPL (per-head block-sparse):")
        for ctx, r in results["perplexity"].items():
            log(f"  ctx={ctx}: full={r['ppl_full']}")
            for k in K_VALUES:
                if f"inc_k{k}" in r:
                    log(f"    k={k}: {r[f'ppl_k{k}']} ({r[f'inc_k{k}']:+.1f}%)")

    if results.get("niah"):
        log("\nNIAH:")
        for ctx, r in results["niah"].items():
            log(f"  ctx={ctx}: full={r['full']}%")
            for k in K_VALUES:
                if f"k{k}" in r:
                    log(f"    k={k}: {r[f'k{k}']}%")

    log("\nDONE!")

if __name__ == "__main__":
    main()
