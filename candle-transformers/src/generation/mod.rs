use candle::{Error, Result};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct LogitsProcessor {
    rng: StdRng,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>, top_k: Option<usize>) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self {
            rng,
            temperature,
            top_p,
            top_k,
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        let logits = if let Some(temperature) = self.temperature {
            logits.iter().map(|l| l / temperature as f32).collect::<Vec<_>>()
        } else {
            logits.to_vec()
        };

        let prs = if let Some(top_k) = self.top_k {
            let mut logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, l)| (i, *l)).collect();
            logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            logits.iter().take(top_k).map(|(i, l)| (*i, *l)).collect::<Vec<_>>()
        } else {
            logits.iter().enumerate().map(|(i, l)| (i, *l)).collect::<Vec<_>>()
        };

        let (indices, logits): (Vec<usize>, Vec<f32>) = if let Some(top_p) = self.top_p {
            let mut logits: Vec<(usize, f32)> = prs.into_iter().collect();
            logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cumsum = 0f32;
            let mut prev_i = 0;
            for (i, (_, l)) in logits.iter().enumerate() {
                let l = l.exp();
                cumsum += l;
                if cumsum > top_p as f32 {
                    break;
                }
                prev_i = i + 1;
            }
            logits.iter().take(prev_i).map(|(i, l)| (*i, *l)).unzip()
        } else {
            prs.into_iter().unzip()
        };

        let prs = logits.iter().map(|v| v.exp()).collect::<Vec<f32>>();
        let sum = prs.iter().sum::<f32>();
        if sum <= 0.0 {
            return Err(Error::Msg("logits all zero or negative".to_string()));
        }
        let prs: Vec<f32> = prs.iter().map(|v| v / sum).collect();

        let distr = WeightedIndex::new(&prs).map_err(Error::wrap)?;
        let idx = distr.sample(&mut self.rng);
        let token = indices[idx];
        Ok(token)
    }
}

#[derive(Clone, Debug)]
pub struct RepetitionPenaltyLogitsWarper {
    penalty: f32,
    prev_tokens: Vec<usize>,
}

impl RepetitionPenaltyLogitsWarper {
    pub fn new(penalty: f32) -> Self {
        Self {
            penalty,
            prev_tokens: Vec::new(),
        }
    }

    pub fn warp(&self, logits: &mut [f32]) {
        for (i, l) in logits.iter_mut().enumerate() {
            if self.prev_tokens.contains(&i) {
                if *l >= 0f32 {
                    *l /= self.penalty;
                } else {
                    *l *= self.penalty;
                }
            }
        }
    }

    pub fn push_token(&mut self, token: usize) {
        self.prev_tokens.push(token);
    }
}

#[derive(Clone, Debug)]
pub struct StoppingCriteria {
    max_length: usize,
    eos_token_id: Option<usize>,
}

impl StoppingCriteria {
    pub fn new(max_length: usize, eos_token_id: Option<usize>) -> Self {
        Self {
            max_length,
            eos_token_id,
        }
    }

    pub fn should_stop(&self, tokens: &[usize]) -> bool {
        if tokens.len() >= self.max_length {
            return true;
        }
        if let Some(eos) = self.eos_token_id {
            if let Some(last) = tokens.last() {
                if *last == eos {
                    return true;
                }
            }
        }
        false
    }
}

pub fn generate_greedy<'a>(
    model: &mut dyn Module,
    input_ids: Tensor,
    config: &GenerationConfig,
    stopping_criteria: &mut dyn StoppingCriteriaList,
) -> Result<Tensor> {
    let mut input_ids = input_ids;
    let mut out = Vec::new();
    let mut logits_processor = config.get_logits_processor();
    let mut repetition_penalty = config.get_repetition_penalty();
    for _ in 0..config.max_new_tokens {
        let logits = model.forward(&input_ids)?;
        let logits = logits
            .to_dtype(DType::F32)?
            .i((.., -1, ..))?
            .flatten_all()?
            .to_vec0()?;
        if let Some(lp) = &mut logits_processor {
            let token = lp.sample(&logits)?;
            out.push(token);
            input_ids = Tensor::new(&[token as i64], input_ids.device())?;
        }
        if let Some(rp) = &mut repetition_penalty {
            rp.push_token(out.last().copied().unwrap_or(0));
            rp.warp(&mut logits);
        }
        if stopping_criteria.should_stop(&out) {
            break;
        }
    }
    let out = Tensor::new(out, input_ids.device())?;
    Ok(out)
}

pub trait StoppingCriteriaList {
    fn should_stop(&mut self, tokens: &[usize]) -> bool;
}

pub trait LogitsProcessorList {
    fn sample(&mut self, logits: &[f32]) -> Result<usize>;
}

pub struct GenerationConfig {
    max_new_tokens: usize,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    seed: u64,
}

impl GenerationConfig {
    pub fn new(
        max_new_tokens: usize,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repetition_penalty: Option<f32>,
        seed: u64,
    ) -> Self {
        Self {
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
        }
    }

    pub fn get_logits_processor(&self) -> Option<LogitsProcessor> {
        if self.temperature.is_some() || self.top_p.is_some() || self.top_k.is_some() {
            Some(LogitsProcessor::new(
                self.seed,
                self.temperature,
                self.top_p,
                self.top_k,
            ))
        } else {
            None
        }
    }

    pub fn get_repetition_penalty(&self) -> Option<RepetitionPenaltyLogitsWarper> {
        self.repetition_penalty
            .map(|penalty| RepetitionPenaltyLogitsWarper::new(penalty))
    }
}
