#!/usr/bin/env python3
"""
R_h analysis: Qwen2.5-7B bf16, 2K -> 128K context
Fix: SDPA prefill + monkey-patch to eager for last-token attention extraction
"""
import json, os, sys, time, gc, math, types
import torch

LOG = r"C:\Users\admin\PRISM\rh_128k_v2_log.txt"
RESULTS_DIR = r"C:\Users\admin\PRISM\results\retrieval_heads"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG, "w") as f:
        f.write("")

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dk = "dtype" if tv >= (4, 57) else "torch_dtype"
    log(f"transformers {transformers.__version__}, kwarg='{dk}'")

    MODEL = "Qwen/Qwen2.5-7B"
    CTXS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    N_TEXTS = 3
    THRESH = 0.3

    log(f"Loading {MODEL} with SDPA (default)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, **{dk: torch.bfloat16},
        device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model.eval()

    cfg = model.config
    nL = cfg.num_hidden_layers
    nA = cfg.num_attention_heads
    nKV = cfg.num_key_value_heads
    G = nA // nKV
    log(f"Arch: {nL}L, {nA}AH, {nKV}KVH, G={G}")
    log(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # === Determine which approach extracts attention weights ===
    log("Probing attention extraction approaches...")
    test_ids = torch.randint(100, 30000, (1, 32), device="cuda")
    approach = None

    # Approach A: SDPA auto-fallback when output_attentions=True
    with torch.no_grad():
        t_out = model(test_ids, output_attentions=True)
    if t_out.attentions is not None and t_out.attentions[0] is not None:
        approach = "sdpa_fallback"
        log("  [A] SDPA auto-fallback works")
    del t_out; torch.cuda.empty_cache()

    # Approach B: config._attn_implementation = "eager"
    if approach is None:
        old_impl = getattr(cfg, '_attn_implementation', 'sdpa')
        cfg._attn_implementation = "eager"
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'config'):
                layer.self_attn.config._attn_implementation = "eager"
        with torch.no_grad():
            t_out = model(test_ids, output_attentions=True)
        if t_out.attentions is not None and t_out.attentions[0] is not None:
            approach = "config_patch"
            log("  [B] config._attn_implementation patch works")
            # restore for now
            cfg._attn_implementation = old_impl
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'config'):
                    layer.self_attn.config._attn_implementation = old_impl
        else:
            cfg._attn_implementation = old_impl
        del t_out; torch.cuda.empty_cache()

    # Approach C: Monkey-patch forward from parent (eager) class
    if approach is None:
        try:
            attn_type = type(model.model.layers[0].self_attn)
            parent = attn_type.__bases__[0]
            if hasattr(parent, 'forward') and parent is not torch.nn.Module:
                eager_fwd = parent.forward
                originals = []
                for layer in model.model.layers:
                    originals.append(layer.self_attn.forward)
                    layer.self_attn.forward = types.MethodType(eager_fwd, layer.self_attn)
                with torch.no_grad():
                    t_out = model(test_ids, output_attentions=True)
                if t_out.attentions is not None and t_out.attentions[0] is not None:
                    approach = "monkey_patch"
                    log(f"  [C] Monkey-patch from {parent.__name__} works")
                # restore
                for i, layer in enumerate(model.model.layers):
                    layer.self_attn.forward = originals[i]
                del t_out; torch.cuda.empty_cache()
        except Exception as e:
            log(f"  [C] Failed: {e}")

    # Approach D: Manual Q*K hook (fallback)
    if approach is None:
        log("  [D] Using manual hook-based attention computation")
        approach = "manual_hook"

    del test_ids; torch.cuda.empty_cache()
    log(f"Selected approach: {approach}")

    # === Switching helpers ===
    _saved = []

    def to_eager():
        _saved.clear()
        if approach == "sdpa_fallback":
            return  # nothing to do
        elif approach == "config_patch":
            cfg._attn_implementation = "eager"
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'config'):
                    layer.self_attn.config._attn_implementation = "eager"
        elif approach == "monkey_patch":
            attn_type = type(model.model.layers[0].self_attn)
            parent = attn_type.__bases__[0]
            eager_fwd = parent.forward
            for layer in model.model.layers:
                _saved.append(layer.self_attn.forward)
                layer.self_attn.forward = types.MethodType(eager_fwd, layer.self_attn)

    def to_sdpa():
        if approach == "sdpa_fallback":
            return
        elif approach == "config_patch":
            cfg._attn_implementation = "sdpa"
            for layer in model.model.layers:
                if hasattr(layer.self_attn, 'config'):
                    layer.self_attn.config._attn_implementation = "sdpa"
        elif approach == "monkey_patch":
            for i, layer in enumerate(model.model.layers):
                layer.self_attn.forward = _saved[i]
            _saved.clear()

    # === Manual hook approach (if needed) ===
    hook_rh = {}
    hooks_handles = []

    def install_manual_hooks(distant_end):
        hook_rh.clear()
        for h in hooks_handles:
            h.remove()
        hooks_handles.clear()

        for li, layer in enumerate(model.model.layers):
            attn_mod = layer.self_attn

            def make_hook(layer_idx, mod):
                def hook_fn(module, args, kwargs, output):
                    # hidden_states -> Q via q_proj, apply rotary, then Q*K
                    try:
                        hidden = args[0] if args else kwargs.get('hidden_states')
                        if hidden is None or hidden.shape[1] != 1:
                            return
                        position_ids = kwargs.get('position_ids', None)
                        past_kv = kwargs.get('past_key_values', None)
                        if past_kv is None:
                            return

                        bsz, q_len, _ = hidden.shape
                        q = mod.q_proj(hidden)
                        k_new = mod.k_proj(hidden)

                        q = q.view(bsz, q_len, nA, cfg.hidden_size // nA).transpose(1, 2)
                        k_new = k_new.view(bsz, q_len, nKV, cfg.hidden_size // nA).transpose(1, 2)

                        # Apply rotary (approximate: use cos/sin from position)
                        if hasattr(mod, 'rotary_emb'):
                            if position_ids is not None:
                                cos, sin = mod.rotary_emb(k_new, position_ids)
                            else:
                                seq_len = past_kv[layer_idx][0].shape[-2] + 1
                                pos = torch.arange(seq_len-1, seq_len, device=hidden.device).unsqueeze(0)
                                cos, sin = mod.rotary_emb(k_new, pos)
                            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

                        # Get full K from cache
                        if hasattr(past_kv, 'key_cache'):
                            k_full = past_kv.key_cache[layer_idx]
                        elif isinstance(past_kv, (list, tuple)):
                            k_full = past_kv[layer_idx][0]
                        else:
                            return

                        # Expand KV heads for GQA
                        k_exp = k_full.repeat_interleave(G, dim=1)  # (bsz, nA, seq, dim)

                        # Compute attention: Q(1,nA,1,d) @ K(1,nA,seq,d)^T
                        scale = math.sqrt(cfg.hidden_size // nA)
                        attn = torch.matmul(q, k_exp.transpose(-2, -1)) / scale
                        attn = torch.softmax(attn, dim=-1)  # (1, nA, 1, seq)

                        last_a = attn[0, :, 0, :]  # (nA, seq)
                        dm = last_a[:, :distant_end].sum(dim=-1)
                        grouped = dm.view(nKV, G).mean(dim=1)
                        hook_rh[layer_idx] = grouped.detach().cpu().tolist()

                    except Exception as e:
                        log(f"    Hook L{layer_idx} error: {e}")
                return hook_fn

            h = attn_mod.register_forward_hook(make_hook(li, attn_mod), with_kwargs=True)
            hooks_handles.append(h)

    def remove_manual_hooks():
        for h in hooks_handles:
            h.remove()
        hooks_handles.clear()

    # === Main experiment loop ===
    results = {
        "model": MODEL, "precision": "bf16",
        "method": f"2step_sdpa_eager_{approach}",
        "arch": {"layers": nL, "attn_heads": nA, "kv_heads": nKV},
        "contexts": {}
    }

    for ctx in CTXS:
        log(f"\n{'='*60}")
        log(f"Context: {ctx}")
        log(f"{'='*60}")

        ctx_rh = []
        ctx_meta = []
        oom = False

        for ti in range(N_TEXTS):
            try:
                torch.cuda.reset_peak_memory_stats()
                ids = torch.randint(100, tokenizer.vocab_size - 100, (1, ctx), device="cuda")
                pre_ids = ids[:, :-1]
                last_id = ids[:, -1:]
                distant_end = max(1, ctx - 256)

                # Step 1: SDPA prefill
                t0 = time.time()
                with torch.no_grad():
                    o1 = model(pre_ids, use_cache=True)
                past_kv = o1.past_key_values
                del o1; torch.cuda.empty_cache()
                t_pre = time.time() - t0

                # Step 2: eager last-token
                if approach == "manual_hook":
                    install_manual_hooks(distant_end)
                    t1 = time.time()
                    with torch.no_grad():
                        o2 = model(last_id, past_key_values=past_kv)
                    t_s2 = time.time() - t1
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    remove_manual_hooks()

                    if not hook_rh:
                        log(f"  Text {ti+1}: manual hooks got no data, skip")
                        del past_kv, o2; torch.cuda.empty_cache(); gc.collect()
                        continue

                    layer_rh = [hook_rh[i] for i in range(nL)]
                    del o2
                else:
                    to_eager()
                    t1 = time.time()
                    with torch.no_grad():
                        o2 = model(last_id, past_key_values=past_kv, output_attentions=True)
                    t_s2 = time.time() - t1
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    to_sdpa()

                    attns = o2.attentions
                    if attns is None or attns[0] is None:
                        log(f"  Text {ti+1}: attentions=None, skip")
                        del past_kv, o2; torch.cuda.empty_cache(); gc.collect()
                        continue

                    layer_rh = []
                    for li, a in enumerate(attns):
                        la = a[0, :, 0, :]
                        dm = la[:, :distant_end].sum(dim=-1)
                        grouped = dm.view(nKV, G).mean(dim=1)
                        layer_rh.append(grouped.cpu().tolist())
                    del o2, attns

                ctx_rh.append(layer_rh)
                ctx_meta.append({"pre_s": round(t_pre,1), "s2_s": round(t_s2,1), "gb": round(peak,2)})
                log(f"  Text {ti+1}: pre={t_pre:.1f}s s2={t_s2:.1f}s VRAM={peak:.1f}GB")

                del past_kv, ids; torch.cuda.empty_cache(); gc.collect()

            except torch.cuda.OutOfMemoryError:
                log(f"  OOM at ctx={ctx}")
                to_sdpa()
                remove_manual_hooks() if approach == "manual_hook" else None
                torch.cuda.empty_cache(); gc.collect()
                torch.cuda.reset_peak_memory_stats()
                oom = True
                break
            except Exception as e:
                log(f"  ERROR: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                to_sdpa()
                if approach == "manual_hook":
                    remove_manual_hooks()
                torch.cuda.empty_cache(); gc.collect()
                break

        if ctx_rh:
            import numpy as np
            arr = np.array(ctx_rh)
            mean = arr.mean(axis=0)
            n_ret = int((mean > THRESH).sum())
            total = nL * nKV
            pct = 100 * n_ret / total

            log(f"  >> R_h(t={THRESH}): {n_ret}/{total} ({pct:.1f}%)")
            log(f"  >> Mean R_h: {mean.mean():.4f}, Max: {mean.max():.4f}")

            results["contexts"][str(ctx)] = {
                "rh_pct": round(pct, 1), "n_ret": n_ret, "total": total,
                "mean_rh": round(float(mean.mean()), 4),
                "max_rh": round(float(mean.max()), 4),
                "rh_matrix": mean.tolist(),
                "meta": ctx_meta,
            }

        if oom:
            log(f"OOM at ctx={ctx}, stopping.")
            break

    path = os.path.join(RESULTS_DIR, "qwen25_7b_bf16_128k_v2.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved: {path}")
    log("DONE!")

if __name__ == "__main__":
    main()
