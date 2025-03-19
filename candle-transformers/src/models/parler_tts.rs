use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl Config {
    pub fn mini() -> Self {
        Self {
            vocab_size: 32000,
            max_position_embeddings: 2048,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv: Linear,
    o: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let kv_sz = cfg.num_key_value_heads * cfg.head_dim;
        let qkv_sz = hidden_sz + 2 * kv_sz;
        let qkv = candle_nn::linear(hidden_sz, qkv_sz, vb.pp("qkv"))?;
        let o = candle_nn::linear(hidden_sz, hidden_sz, vb.pp("o"))?;
        Ok(Self {
            qkv,
            o,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden_size: cfg.hidden_size,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        pos: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_sz) = xs.dims3()?;
        let qkv = self.qkv.forward(xs)?;
        let kv_sz = self.num_key_value_heads * self.head_dim;

        let q = qkv.narrow(D::Minus1, 0, hidden_sz)?;
        let k = qkv.narrow(D::Minus1, hidden_sz, kv_sz)?;
        let v = qkv.narrow(D::Minus1, hidden_sz + kv_sz, kv_sz)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = apply_rotary_emb(&q, pos, self.head_dim)?;
        let k = apply_rotary_emb(&k, pos, self.head_dim)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let y = {
            let attn_weights = (q.matmul(&k.t()?)? * scale)?;
            let attn_weights = match attention_mask {
                Some(mask) => {
                    let mask = mask.broadcast_as(attn_weights.dims())?;
                    attn_weights.broadcast_add(&mask)?
                }
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
            let y = attn_weights.matmul(&v)?;
            y.transpose(1, 2)?
                .reshape((b_sz, seq_len, self.hidden_size))?
        };
        self.o.forward(&y)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

fn apply_rotary_emb(xs: &Tensor, pos: &Tensor, head_dim: usize) -> Result<Tensor> {
    let (_b_sz, _num_heads, seq_len, _head_dim) = xs.dims4()?;
    let cos = pos.i((.., .., ..head_dim / 2))?;
    let sin = pos.i((.., .., head_dim / 2..))?;
    let (x1, x2) = (
        xs.narrow(D::Minus1, 0, head_dim / 2)?,
        xs.narrow(D::Minus1, head_dim / 2, head_dim / 2)?,
    );
    let x1p = x1.broadcast_mul(&cos)? + x2.broadcast_mul(&sin)?.neg()?;
    let x2p = x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?;
    Tensor::cat(&[x1p, x2p], D::Minus1)
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
    let xs = xs
        .unsqueeze(2)?
        .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
    Ok(xs)
}

struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate = candle_nn::linear(hidden_sz, intermediate_sz, vb.pp("gate"))?;
        let up = candle_nn::linear(hidden_sz, intermediate_sz, vb.pp("up"))?;
        let down = candle_nn::linear(intermediate_sz, hidden_sz, vb.pp("down"))?;
        Ok(Self { gate, up, down })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_gate = self.gate.forward(xs)?;
        let xs_up = self.up.forward(xs)?;
        let xs = (candle_nn::ops::silu(&xs_gate)? * xs_up)?;
        self.down.forward(&xs)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: FeedForward,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attn = Attention::new(vb.pp("self_attn"), cfg)?;
        let mlp = FeedForward::new(vb.pp("mlp"), cfg)?;
        let input_layernorm = candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        pos: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, pos, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Clone)]
pub struct ParlerTts {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: Linear,
    pos_emb: Tensor,
    device: Device,
}

impl ParlerTts {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(vb_m.pp(&format!("layers.{layer_idx}")), cfg)?;
            layers.push(layer);
        }
        let norm = candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb_m.pp("lm_head"))?;
        let pos_emb = build_rotary_emb(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            pos_emb,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = candle::utils::causal_mask(seq_len, input_ids.device())?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, pos, attention_mask.as_ref())?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    pub fn generate(
        &mut self,
        input_ids: Tensor,
        config: &GenerationConfig,
        stopping_criteria: &mut dyn StoppingCriteriaList,
    ) -> Result<Tensor> {
        let pos = self.pos_emb.clone();
        let mut input_ids = input_ids;
        let mut out = Vec::new();
        let mut logits_processor = config.get_logits_processor();
        let mut repetition_penalty = config.get_repetition_penalty();
        for idx in 0..config.max_new_tokens {
            let logits = self.forward(&input_ids, &pos)?;
            let logit = logits.i((.., -1, ..))?.flatten_all()?;
            let logit = logit.to_dtype(DType::F32)?;
            let mut logit = logit.to_vec1::<f32>()?;
            if let Some(lp) = &mut logits_processor {
                let token = lp.sample(&logit)?;
                out.push(token);
                let new_input = Tensor::new(&[token as i64], input_ids.device())?;
                input_ids = Tensor::cat(&[&input_ids, &new_input], 1)?;
            }
            if let Some(rp) = &mut repetition_penalty {
                rp.push_token(out.last().copied().unwrap_or(0));
                rp.warp(&mut logit);
            }
            if stopping_criteria.should_stop(&out) {
                break;
            }
        }
        let out = Tensor::new(out, input_ids.device())?.unsqueeze(0)?;
        Ok(out)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

fn build_rotary_emb(max_seq_len: usize, head_dim: usize, theta: f32) -> Result<Tensor> {
    let inv_freq = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    let t = Tensor::arange(0u32, max_seq_len as u32, &Device::Cpu)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?;
    let freqs = t.matmul(&Tensor::new(inv_freq, &Device::Cpu)?.reshape((1, head_dim / 2))?)?
        .to_device(&Device::Cpu)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let cos = cos.expand((1, 1, max_seq_len, head_dim / 2))?;
    let sin = sin.expand((1, 1, max_seq_len, head_dim / 2))?;
    Tensor::cat(&[cos, sin], D::Minus1)
}
