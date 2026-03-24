#!/usr/bin/env python3
"""
PRISM MRR-integrated NIAH Experiment v2.
Changes from v1:
  - SDPA attention (Flash Attention) instead of eager -> enables 32K-128K contexts
  - SOI Si platform MRR parameters (R=8um, 4-5 bit, reference paper values)
  - Extended contexts: 4K, 8K, 16K, 32K, 64K, 128K
  - No monkey-patching (combined mask + SDPA compatible)
  - 2D NIAH heatmap output format (context x needle position)
  - Better progress logging
  - Chunked prefill for 16K+ (avoids huge logits tensor)
  - v2b FIX: Pruned KV cache instead of second forward pass (fixes RoPE discontinuity)
  - Sparse eval now prunes full prefill cache to selected blocks, matching Quest/InfLLM
"""
import json, os, sys, time, gc, math

# Fix cuDNN DLL path for conda env
_cudnn_dir = r'C:\Users\admin\miniconda3\envs\gpu\Library\bin'
if os.path.isdir(_cudnn_dir):
    os.environ['PATH'] = _cudnn_dir + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(_cudnn_dir)
    except Exception:
        pass

# Note: sys.stdout.reconfigure() removed -- causes silent crash with numpy
# on Windows SSH. Use PYTHONIOENCODING=utf-8 env var instead if needed.

import numpy as np
import torch

# Auto-detect PRISM root: try D:\PRISM (lab-pc) then script's own directory
_prism_root = r"D:\PRISM" if os.path.isdir(r"D:\PRISM\prism") else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _prism_root)

LOG = r"C:\Users\admin\PRISM\mrr_niah_v2b_log.txt"
RESULTS_DIR = r"C:\Users\admin\PRISM\results\downstream"
RESULTS_FILE = "mrr_niah_v2b.json"  # v2b = pruned-cache fix
TEXT_FILE = r"C:\Users\admin\PRISM\paper_text.txt"
RH_FILE = r"C:\Users\admin\PRISM\results\retrieval_heads\qwen25_7b_bf16_128k.json"

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def vram_gb():
    return torch.cuda.memory_allocated() / 1e9

def chunked_prefill(model, input_ids, chunk_size=4096):
    """Chunked prefill with DynamicCache on GPU.
    Avoids materializing full logits tensor (e.g. 128K x 152K x 2 = 37.7GB).
    Each chunk only produces chunk_size x vocab logits (~1.2GB for 4096).
    KV cache grows on GPU but fits: 128K KV ~ 14.7GB + model 15GB = 30GB.
    """
    seq_len = input_ids.shape[1]
    n_chunks = math.ceil(seq_len / chunk_size)
    log(f"  Chunked prefill: {seq_len} tokens, {n_chunks} chunks x {chunk_size}")
    cache = None  # let model create its own cache type
    last_logits = None
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]
        pos_ids = torch.arange(start, end, device=chunk_ids.device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(chunk_ids, position_ids=pos_ids,
                            past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        last_logits = outputs.logits[:, -1:, :].clone()  # clone to free chunk logits
        del outputs
        torch.cuda.empty_cache()
        if (i + 1) % 4 == 0 or (i + 1) == n_chunks:
            log(f"    chunk {i+1}/{n_chunks} ({end}/{seq_len}) VRAM={vram_gb():.1f}GB")
    return cache, last_logits


def generate_from_cache(model, tokenizer, cache, last_logits, max_new_tokens=30):
    """Autoregressive generation using pre-built KV cache."""
    generated_ids = []
    next_token = torch.argmax(last_logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids.append(next_token.item())
    del last_logits
    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(next_token, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        tok_id = next_token.item()
        generated_ids.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
        del outputs
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def chunked_sparse_eval(model, tokenizer, sparse_ids, position_ids,
                        chunk_size=4096, max_new_tokens=30):
    """Chunked forward pass for sparse evaluation (non-contiguous position_ids).
    Same principle as chunked_prefill: process in chunks to avoid huge logits tensor.
    Works with any position_ids pattern (contiguous or sparse).
    """
    seq_len = sparse_ids.shape[1]
    if seq_len <= chunk_size:
        # Small enough for single pass
        with torch.no_grad():
            out = model(sparse_ids, position_ids=position_ids, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1:, :].clone()
        del out
        torch.cuda.empty_cache()
    else:
        n_chunks = math.ceil(seq_len / chunk_size)
        cache = None
        last_logits = None
        for i in range(n_chunks):
            s = i * chunk_size
            e = min((i + 1) * chunk_size, seq_len)
            chunk_ids = sparse_ids[:, s:e]
            chunk_pos = position_ids[:, s:e]
            with torch.no_grad():
                out = model(chunk_ids, position_ids=chunk_pos,
                           past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last_logits = out.logits[:, -1:, :].clone()
            del out
            torch.cuda.empty_cache()

    # Generate from cache
    generated_ids = []
    next_token = torch.argmax(last_logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids.append(next_token.item())
    del last_logits
    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        tok_id = next_token.item()
        generated_ids.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
        del out

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del cache
    torch.cuda.empty_cache()
    return answer


def generate_from_pruned_cache(model, tokenizer, full_cache, selected_positions,
                                ids_1d, actual_len, nL, max_new_tokens=30):
    """Generate using pruned KV cache from full prefill.

    Instead of a second forward pass with non-contiguous position_ids
    (which breaks RoPE continuity), reuse the full prefill's KV cache
    pruned to selected blocks + recent window.  The cached keys already
    have correct RoPE applied during full-context prefill.
    This matches real Quest/InfLLM decode-phase sparse attention.
    """
    from transformers.cache_utils import DynamicCache

    # Exclude last position -- we re-process it against pruned cache
    keep_positions = [p for p in selected_positions if p < actual_len - 1]
    pos_idx = torch.tensor(keep_positions, device="cuda", dtype=torch.long)

    # Build pruned cache by indexing full cache's sequence dimension
    pruned_cache = DynamicCache()
    cache_list = list(full_cache)
    for li in range(nL):
        k_full = cache_list[li][0]  # (batch, n_kv_heads, seq_len, head_dim)
        v_full = cache_list[li][1]
        pruned_cache.update(k_full[:, :, pos_idx, :], v_full[:, :, pos_idx, :], li)
    del cache_list

    # Forward pass for last token only, attending to pruned cache
    last_tok = ids_1d[actual_len - 1].view(1, 1)
    last_pos = torch.tensor([[actual_len - 1]], device="cuda", dtype=torch.long)
    with torch.no_grad():
        out = model(last_tok, position_ids=last_pos,
                    past_key_values=pruned_cache, use_cache=True)

    gen_cache = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids = [next_token.item()]
    del out
    torch.cuda.empty_cache()

    # Generate autoregressively with explicit position_ids
    for step in range(max_new_tokens - 1):
        gen_pos = torch.tensor([[actual_len + step]], device="cuda", dtype=torch.long)
        with torch.no_grad():
            out = model(next_token, position_ids=gen_pos,
                        past_key_values=gen_cache, use_cache=True)
        gen_cache = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        tok_id = next_token.item()
        generated_ids.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
        del out

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del gen_cache, pruned_cache
    torch.cuda.empty_cache()
    return answer


def extract_signatures_from_dynacache(cache, nL, nKV, D_SIG, BLOCK_SIZE, seq_len):
    """Extract mean-key block signatures from model's KV cache.
    Uses same iteration pattern as the <=8K path (for lkv in cache: lkv[0]).
    """
    n_blocks = seq_len // BLOCK_SIZE
    sigs_per_layer = []
    queries_per_layer = []
    # Iterate cache -> list of (key_tensor, value_tensor) per layer
    cache_list = list(cache)
    for li in range(nL):
        keys = cache_list[li][0]  # (batch, n_kv_heads, seq_len, head_dim)
        k = keys[0, :, :, :D_SIG].cpu().float().numpy()
        trimmed = k[:, :n_blocks * BLOCK_SIZE, :]
        reshaped = trimmed.reshape(nKV, n_blocks, BLOCK_SIZE, D_SIG)
        sigs_per_layer.append(reshaped.mean(axis=2))
        queries_per_layer.append(k[:, -1, :])
    del cache_list
    return sigs_per_layer, queries_per_layer, n_blocks


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

    # Load text (repeat if needed for long contexts)
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()
    log(f"Text: {len(raw_text)} chars")

    # Load R_h retrieval head mask
    rh_matrix = None
    if os.path.exists(RH_FILE):
        with open(RH_FILE) as f:
            rh_data = json.load(f)
        for ctx_key in ["8192", "4096", "2048"]:
            if ctx_key in rh_data and "rh_matrix" in rh_data[ctx_key]:
                rh_matrix = np.array(rh_data[ctx_key]["rh_matrix"])
                log(f"Loaded R_h from ctx={ctx_key}")
                break
    if rh_matrix is None:
        rh_matrix = np.ones((28, 4)) * 0.5
        rh_matrix[:3, :] = 0.1
        log("WARNING: Using fallback R_h matrix")

    retrieval_mask = rh_matrix > RH_THRESH
    n_ret = retrieval_mask.sum()
    log(f"Retrieval heads: {n_ret}/{retrieval_mask.size} ({100*n_ret/retrieval_mask.size:.1f}%)")

    # Load model -- SDPA (Flash Attention) for memory efficiency
    log(f"Loading {MODEL} (SDPA mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, **{dk: torch.bfloat16},
        attn_implementation="sdpa",
        device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model.eval()

    cfg = model.config
    nL = cfg.num_hidden_layers
    nA = cfg.num_attention_heads
    nKV = cfg.num_key_value_heads
    G = nA // nKV
    hd = cfg.hidden_size // nA
    log(f"Arch: {nL}L, {nA}AH, {nKV}KVH, G={G}, head_dim={hd}")
    log(f"SDPA enabled -- O(n) memory, supports up to 128K context")

    # Tokenize full text
    all_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    log(f"Available tokens: {len(all_ids)}")

    # MRR impairment configs (simplified model)
    mrr_configs = {
        "ideal": None,  # No MRR, ideal inner product (baseline)
        "si_5bit_20pm": {"bits": 5, "drift_sigma": 0.01},   # Standard SOI Si
        "si_4bit_30pm": {"bits": 4, "drift_sigma": 0.02},   # Pessimistic
        "si_5bit_10pm": {"bits": 5, "drift_sigma": 0.005},  # Optimistic
    }

    # NIAH Experiment -- extended contexts
    log("\n" + "="*60)
    log("NIAH v2: MRR-integrated (SDPA, SOI Si, extended ctx)")
    log("="*60)

    NEEDLE = "The secret access code for the photonic laboratory is 'quantum-cascade-7391'."
    QUESTION = "\n\nQuestion: What is the secret access code for the photonic laboratory?\nAnswer:"
    EXPECTED = "quantum-cascade-7391"

    NIAH_CTXS = [4096, 8192, 16384, 32768, 65536, 131072]
    POSITIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Resume: load existing results and skip completed contexts
    prev_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    niah_results = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as f:
                prev = json.load(f)
            if "niah" in prev:
                niah_results = prev["niah"]
                completed = [int(k) for k in niah_results.keys()]
                log(f"RESUME: found results for contexts {[c//1024 for c in sorted(completed)]}K")
                NIAH_CTXS = [c for c in NIAH_CTXS if str(c) not in niah_results]
                log(f"REMAINING contexts: {[c//1024 for c in NIAH_CTXS]}K")
        except Exception as e:
            log(f"WARNING: could not load previous results: {e}")

    results = {
        "model": MODEL,
        "block_size": BLOCK_SIZE,
        "d_sig": D_SIG,
        "method": "mrr_integrated_niah_v2",
        "attn_implementation": "sdpa",
        "platform": "SOI_Si",
        "positions": POSITIONS,
        "contexts": [4096, 8192, 16384, 32768, 65536, 131072],
        "k_values": K_VALUES,
        "mrr_configs": {k: str(v) for k, v in mrr_configs.items()},
    }

    needle_toks = tokenizer.encode(NEEDLE, add_special_tokens=False)
    question_toks = tokenizer.encode(QUESTION, add_special_tokens=False)

    total_experiments = len(NIAH_CTXS) * len(POSITIONS)
    experiment_count = 0

    for ctx in NIAH_CTXS:
        log(f"\n{'='*60}")
        log(f"Context: {ctx} tokens ({ctx//1024}K)")
        log(f"{'='*60}")

        n_blocks = ctx // BLOCK_SIZE
        hay_len = ctx - len(needle_toks) - len(question_toks) - 5

        if hay_len < 100:
            log(f"  SKIP: hay_len={hay_len} too short")
            continue

        # Prepare haystack tokens (repeat if needed for long contexts)
        if len(all_ids) >= hay_len:
            hay_toks = all_ids[:hay_len]
        else:
            repeats = (hay_len // len(all_ids)) + 1
            hay_toks = (all_ids * repeats)[:hay_len]
            log(f"  Text repeated {repeats}x to fill {hay_len} tokens")

        # Track hits per config per k
        hits = {}
        for cfg_name in mrr_configs:
            hits[cfg_name] = {k: 0 for k in K_VALUES}
        hits["full"] = 0
        total = 0

        for pos_frac in POSITIONS:
            experiment_count += 1
            t0 = time.time()

            insert = int(len(hay_toks) * pos_frac)
            input_toks = hay_toks[:insert] + needle_toks + hay_toks[insert:]
            input_toks = input_toks[:hay_len] + question_toks
            ids = torch.tensor([input_toks[:ctx]], device="cuda")
            actual_len = ids.shape[1]
            actual_n_blocks = actual_len // BLOCK_SIZE

            # -- Full attention baseline + extract KV cache --
            USE_CHUNKED = ctx >= 16384  # 16K+ uses chunked prefill

            if USE_CHUNKED:
                log(f"  Chunked mode (ctx={actual_len}, {actual_n_blocks} blocks)")
                full_cache, last_logits = chunked_prefill(model, ids, chunk_size=4096)
                log(f"  Prefill done. VRAM={vram_gb():.1f}GB. Generating...")
                answer_f = generate_from_cache(model, tokenizer, full_cache, last_logits,
                                                max_new_tokens=30)
                hit_f = EXPECTED.lower() in answer_f.lower()
                if hit_f:
                    hits["full"] += 1
                log(f"  Generate done: {'HIT' if hit_f else 'MISS'}. VRAM={vram_gb():.1f}GB")

                # Extract signatures from GPU-stored cache
                log(f"  Extracting signatures...")
                sigs_per_layer, queries_per_layer, _ = extract_signatures_from_dynacache(
                    full_cache, nL, nKV, D_SIG, BLOCK_SIZE, actual_len
                )
                log(f"  Signatures done. VRAM={vram_gb():.1f}GB")
                del last_logits
                torch.cuda.empty_cache()
                log(f"  Full cache retained for sparse eval. VRAM={vram_gb():.1f}GB")
            else:
                # <=8K: single forward pass (fast, fits in VRAM)
                with torch.no_grad():
                    cache_out = model(ids, use_cache=True)
                keys_per_layer = []
                for lkv in cache_out.past_key_values:
                    keys_per_layer.append(lkv[0][0, :, :, :D_SIG].cpu().float().numpy())
                last_logits = cache_out.logits[:, -1:, :].clone()
                full_cache = cache_out.past_key_values
                del cache_out
                torch.cuda.empty_cache()

                answer_f = generate_from_cache(model, tokenizer, full_cache, last_logits,
                                                max_new_tokens=30)
                hit_f = EXPECTED.lower() in answer_f.lower()
                if hit_f:
                    hits["full"] += 1
                del last_logits
                torch.cuda.empty_cache()

                # Block signatures from GPU-extracted keys
                sigs_per_layer = []
                queries_per_layer = []
                for li in range(nL):
                    keys = keys_per_layer[li]
                    trimmed = keys[:, :actual_n_blocks * BLOCK_SIZE, :]
                    reshaped = trimmed.reshape(nKV, actual_n_blocks, BLOCK_SIZE, D_SIG)
                    sigs_per_layer.append(reshaped.mean(axis=2))
                    queries_per_layer.append(keys[:, -1, :])
                del keys_per_layer
                gc.collect()

            # -- Test each MRR config (pruned KV cache from full prefill) --
            ids_1d = ids.squeeze(0)  # (seq,) for indexing

            for cfg_name, mrr_params in mrr_configs.items():
                for k in K_VALUES:
                    if k >= actual_n_blocks:
                        if hit_f:
                            hits[cfg_name][k] += 1
                        continue

                    # Block selection via MRR (union across retrieval heads)
                    selected_blocks = set()

                    for li in range(nL):
                        for hi in range(nKV):
                            if not retrieval_mask[li, hi]:
                                continue

                            sigs = sigs_per_layer[li][hi]  # (n_blocks, D_SIG)
                            query = queries_per_layer[li][hi]  # (D_SIG,)

                            if mrr_params is None:
                                scores = sigs @ query
                            else:
                                # Simplified MRR: normalize [0,1] -> quantize -> drift -> denormalize -> matmul
                                w_min, w_max = sigs.min(), sigs.max()
                                W = (sigs - w_min) / (w_max - w_min + 1e-30)
                                levels = 2 ** mrr_params["bits"]
                                W_q = np.round(W * (levels - 1)) / (levels - 1)
                                if mrr_params["drift_sigma"] > 0:
                                    W_d = W_q + np.random.normal(0, mrr_params["drift_sigma"], W_q.shape)
                                    W_d = np.clip(W_d, 0, 1)
                                else:
                                    W_d = W_q
                                S_recon = W_d * (w_max - w_min) + w_min
                                scores = S_recon @ query

                            top_idx = np.argsort(scores)[-k:]
                            selected_blocks.update(top_idx.tolist())

                    # If union covers >= 95% of blocks, just use full attention result
                    coverage = len(selected_blocks) / actual_n_blocks
                    if coverage >= 0.95:
                        if hit_f:
                            hits[cfg_name][k] += 1
                        log(f"    {cfg_name} k={k}: union={len(selected_blocks)}/{actual_n_blocks} "
                            f"({coverage:.0%}) -> use full result")
                        continue

                    # Collect selected token positions (blocks + recent window)
                    selected_positions = []
                    for b in sorted(selected_blocks):
                        s = b * BLOCK_SIZE
                        e = min((b + 1) * BLOCK_SIZE, actual_len)
                        selected_positions.extend(range(s, e))
                    recent_start = max(0, actual_len - RECENT_WINDOW)
                    selected_positions.extend(range(recent_start, actual_len))
                    selected_positions = sorted(set(selected_positions))

                    n_sel_tokens = len(selected_positions)
                    log(f"    {cfg_name} k={k}: union={len(selected_blocks)}/{actual_n_blocks} "
                        f"({coverage:.0%}), {n_sel_tokens} tokens")

                    # Generate from pruned KV cache (reuse full prefill cache)
                    answer_s = generate_from_pruned_cache(
                        model, tokenizer, full_cache, selected_positions,
                        ids_1d, actual_len, nL, max_new_tokens=30
                    )
                    if EXPECTED.lower() in answer_s.lower():
                        hits[cfg_name][k] += 1
                    torch.cuda.empty_cache()

            total += 1
            dt = time.time() - t0
            status = "HIT" if hit_f else "MISS"
            log(f"  [{experiment_count}/{total_experiments}] ctx={ctx//1024}K pos={pos_frac:.0%}: "
                f"full={status} ({dt:.1f}s) VRAM={vram_gb():.1f}GB")

            del ids, ids_1d, full_cache, sigs_per_layer, queries_per_layer
            torch.cuda.empty_cache()
            gc.collect()

        # Aggregate results for this context
        if total > 0:
            ctx_res = {
                "full": round(100 * hits["full"] / total, 1),
                "total": total,
            }
            for cfg_name in mrr_configs:
                for k in K_VALUES:
                    key = f"{cfg_name}_k{k}"
                    ctx_res[key] = round(100 * hits[cfg_name][k] / total, 1)
            niah_results[str(ctx)] = ctx_res

            log(f"\n  >> ctx={ctx//1024}K: full={ctx_res['full']}%")
            for cfg_name in mrr_configs:
                for k in K_VALUES:
                    key = f"{cfg_name}_k{k}"
                    log(f"     {cfg_name} k={k}: {ctx_res[key]}%")

        # Save intermediate results after each context
        results["niah"] = niah_results
        path = os.path.join(RESULTS_DIR, RESULTS_FILE)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  (intermediate save: {path})")

    # Final save and summary
    results["niah"] = niah_results
    path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "="*60)
    log("SUMMARY -- NIAH v2 (SDPA, SOI Si)")
    log("="*60)
    for ctx_str, r in niah_results.items():
        ctx = int(ctx_str)
        log(f"\nctx={ctx//1024}K: full={r['full']}%")
        for cfg_name in mrr_configs:
            for k in K_VALUES:
                key = f"{cfg_name}_k{k}"
                if key in r:
                    log(f"  {cfg_name} k={k}: {r[key]}%")

    log(f"\nSaved: {path}")
    log("DONE!")

if __name__ == "__main__":
    main()
