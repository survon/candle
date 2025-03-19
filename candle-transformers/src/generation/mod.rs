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

        let distr = WeightedIndex::new(prs).map_err(Error::wrap)?;
        let idx = distr.sample(&mut self.rng);
        let token = indices[idx];
        Ok(token)
    }
}
