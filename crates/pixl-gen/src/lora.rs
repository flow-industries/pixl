//! Runtime merge of kohya/sgm-format SDXL LoRAs into the base UNet.
//!
//! Pixel-art SDXL LoRAs (e.g. nerijs/pixel-art-xl) ship in sgm/A1111 naming
//! (`lora_unet_input_blocks_4_1_...`), while candle's UNet loads diffusers names
//! (`down_blocks.1.attentions.0...`). These LoRAs only touch attention blocks,
//! so we translate just the attention-block envelope and pass the inner payload
//! through, then add `scale * up @ down` into the base weight.
//!
//! The merge runs on CPU and writes a complete merged UNet to a content-addressed
//! cache, which is then loaded through candle's normal mmap path — so generation
//! keeps the base model's fast, flat-memory residency instead of a ~5 GB eager
//! in-RAM copy.

use std::collections::{hash_map::DefaultHasher, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

/// sgm attention-block envelope -> diffusers, for SDXL. Only attention sub-blocks
/// (the `.1` position) carry the modules these LoRAs touch.
fn envelope(section: &str, n: usize, sub: usize) -> Option<&'static str> {
    if sub != 1 {
        return None;
    }
    Some(match (section, n) {
        ("input_blocks", 4) => "down_blocks.1.attentions.0",
        ("input_blocks", 5) => "down_blocks.1.attentions.1",
        ("input_blocks", 7) => "down_blocks.2.attentions.0",
        ("input_blocks", 8) => "down_blocks.2.attentions.1",
        ("output_blocks", 0) => "up_blocks.0.attentions.0",
        ("output_blocks", 1) => "up_blocks.0.attentions.1",
        ("output_blocks", 2) => "up_blocks.0.attentions.2",
        ("output_blocks", 3) => "up_blocks.1.attentions.0",
        ("output_blocks", 4) => "up_blocks.1.attentions.1",
        ("output_blocks", 5) => "up_blocks.1.attentions.2",
        _ => return None,
    })
}

/// Inner attention payload, sgm underscore form -> diffusers dotted form.
fn convert_inner(inner: &str) -> Option<String> {
    match inner {
        "proj_in" => return Some("proj_in".into()),
        "proj_out" => return Some("proj_out".into()),
        _ => {}
    }
    let rest = inner.strip_prefix("transformer_blocks_")?;
    let (i, leaf) = rest.split_once('_')?;
    i.parse::<usize>().ok()?;
    let leaf = match leaf {
        "attn1_to_q" => "attn1.to_q",
        "attn1_to_k" => "attn1.to_k",
        "attn1_to_v" => "attn1.to_v",
        "attn1_to_out_0" => "attn1.to_out.0",
        "attn2_to_q" => "attn2.to_q",
        "attn2_to_k" => "attn2.to_k",
        "attn2_to_v" => "attn2.to_v",
        "attn2_to_out_0" => "attn2.to_out.0",
        "ff_net_0_proj" => "ff.net.0.proj",
        "ff_net_2" => "ff.net.2",
        "norm1" => "norm1",
        "norm2" => "norm2",
        "norm3" => "norm3",
        _ => return None,
    };
    Some(format!("transformer_blocks.{i}.{leaf}"))
}

/// Map a kohya LoRA module name to the diffusers weight key candle's UNet uses.
pub fn map_module(kohya: &str) -> Option<String> {
    let m = kohya.strip_prefix("lora_unet_")?;
    let (env, inner) = if let Some(r) = m.strip_prefix("input_blocks_") {
        let (n, r) = r.split_once('_')?;
        let (sub, inner) = r.split_once('_')?;
        (envelope("input_blocks", n.parse().ok()?, sub.parse().ok()?)?, inner)
    } else if let Some(r) = m.strip_prefix("output_blocks_") {
        let (n, r) = r.split_once('_')?;
        let (sub, inner) = r.split_once('_')?;
        (envelope("output_blocks", n.parse().ok()?, sub.parse().ok()?)?, inner)
    } else if let Some(r) = m.strip_prefix("middle_block_") {
        let (sub, inner) = r.split_once('_')?;
        if sub != "1" {
            return None;
        }
        ("mid_block.attentions.0", inner)
    } else {
        return None;
    };
    Some(format!("{env}.{}.weight", convert_inner(inner)?))
}

/// Add `scale * up @ down` for every mappable module of `lora_path` into `base`.
/// Returns (applied, skipped). Runs on whatever device `base` tensors live on.
fn merge_into(base: &mut HashMap<String, Tensor>, lora_path: &Path, user_scale: f32) -> Result<(usize, usize)> {
    let lora = candle_core::safetensors::load(lora_path, &Device::Cpu)
        .with_context(|| format!("load lora {}", lora_path.display()))?;
    let mut modules: HashSet<String> = HashSet::new();
    for k in lora.keys() {
        for suf in [".lora_down.weight", ".lora_up.weight", ".alpha"] {
            if let Some(m) = k.strip_suffix(suf) {
                modules.insert(m.to_string());
            }
        }
    }
    let (mut applied, mut skipped) = (0usize, 0usize);
    for m in &modules {
        let (down, up) = match (
            lora.get(&format!("{m}.lora_down.weight")),
            lora.get(&format!("{m}.lora_up.weight")),
        ) {
            (Some(d), Some(u)) => (d, u),
            _ => {
                skipped += 1;
                continue;
            }
        };
        let key = match map_module(m) {
            Some(k) => k,
            None => {
                skipped += 1;
                continue;
            }
        };
        let base_w = match base.get(&key) {
            Some(w) => w.clone(),
            None => {
                skipped += 1;
                continue;
            }
        };
        let rank = down.dim(0)? as f32;
        let alpha = match lora.get(&format!("{m}.alpha")) {
            Some(a) => a.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            None => rank,
        };
        let eff = (user_scale * alpha / rank) as f64;
        let d = down.to_dtype(DType::F32)?;
        let u = up.to_dtype(DType::F32)?;
        let delta = u.matmul(&d)?; // [out, rank] @ [rank, in] = [out, in]
        let merged = (base_w.to_dtype(DType::F32)? + (delta * eff)?)?.to_dtype(base_w.dtype())?;
        base.insert(key, merged);
        applied += 1;
    }
    Ok((applied, skipped))
}

fn cache_key(base_path: &Path, loras: &[(PathBuf, f32)]) -> String {
    let mut h = DefaultHasher::new();
    base_path.to_string_lossy().hash(&mut h);
    for (p, s) in loras {
        p.to_string_lossy().hash(&mut h);
        s.to_bits().hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

/// Path to a complete UNet safetensors with `loras` merged into `base_path`.
/// Content-addressed; the merge (on CPU) runs only on the first request.
pub fn merged_unet_path(base_path: &Path, loras: &[(PathBuf, f32)], cache_dir: &Path) -> Result<PathBuf> {
    let out = cache_dir.join(format!("unet-{}.safetensors", cache_key(base_path, loras)));
    if out.exists() {
        eprintln!("  using cached merged UNet ({})", out.display());
        return Ok(out);
    }
    std::fs::create_dir_all(cache_dir).with_context(|| format!("create {}", cache_dir.display()))?;

    let mut base = candle_core::safetensors::load(base_path, &Device::Cpu).context("load base unet")?;
    for (path, scale) in loras {
        let (a, s) = merge_into(&mut base, path, *scale)?;
        anyhow::ensure!(a > 0, "lora {} mapped 0 modules — key convention mismatch", path.display());
        eprintln!(
            "  lora {}: merged {a} modules ({s} skipped)",
            path.file_name().and_then(|s| s.to_str()).unwrap_or("?")
        );
    }
    let tmp = out.with_extension("tmp");
    candle_core::safetensors::save(&base, &tmp).context("write merged unet")?;
    std::fs::rename(&tmp, &out).context("commit merged unet cache")?;
    eprintln!("  cached merged UNet -> {}", out.display());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_known_modules() {
        assert_eq!(
            map_module("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q").unwrap(),
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight"
        );
        assert_eq!(
            map_module("lora_unet_input_blocks_4_1_proj_in").unwrap(),
            "down_blocks.1.attentions.0.proj_in.weight"
        );
        assert_eq!(
            map_module("lora_unet_output_blocks_5_1_transformer_blocks_0_attn2_to_out_0").unwrap(),
            "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.weight"
        );
        assert_eq!(
            map_module("lora_unet_middle_block_1_transformer_blocks_0_ff_net_0_proj").unwrap(),
            "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight"
        );
        assert!(map_module("lora_te1_text_model_encoder_layers_0_self_attn_k_proj").is_none());
        assert!(map_module("lora_unet_input_blocks_4_0_in_layers_2").is_none());
    }
}
