use std::result::Result;
use rand::prelude::*;

struct Neuron {
    weights: Vec<f64>,
    activation: usize,
    bias: f64
}

pub trait New {
    fn new(num_weights: usize, activation: usize, bias: Option<f64>) -> Self;
    fn set_weights(&mut self, weights: Vec<f64>) -> Result<bool, String>;
}

pub trait ForwardPass {
    fn forward_pass(self, inputs: Vec<f64>) -> f64;
}

impl New for Neuron {
    fn new(num_weights: usize, activation: usize, bias: Option<f64>) -> Self {
        let mut initial_weights: Vec<f64> = Vec::with_capacity(num_weights);
        let mut rng = rand::thread_rng();
        for _ in 0..num_weights {
            initial_weights.push(rng.gen());
        }
        Neuron {
            weights: initial_weights,
            activation: activation,
            bias: bias.unwrap_or(1.)
        }
    }

    fn set_weights(&mut self, weights: Vec<f64>) -> Result<bool, String>{
        if weights.len() != self.weights.len() {
            return Err(format!("Expected {} weights, but got {} weights!", self.weights.len(), weights.len()));
        }
        let num_weights = self.weights.len();
        for i in 0..num_weights {
            self.weights[i] = weights[i];
        }
        Ok(true)
    }
}

impl ForwardPass for Neuron {
    fn forward_pass(self, inputs: Vec<f64>) -> f64 {
        let num_weights: usize = self.weights.len();
        let mut result: f64 = 0.;
        for i in 0..num_weights {
            result += self.weights[i] * inputs[i];
        }
        result += self.bias;
        result
    }
}

fn main() {
    let inputs: [f64; 3] = [1.2, 5.1, 2.1];
    let weights: [f64; 3] = [3.1, 2.1, 8.7];
    // let bias: f64 = 3.;

    let mut n: Neuron = Neuron::new(3, 1, Some(3.));
    n.set_weights(weights.to_vec()).unwrap();
    println!("{}", n.forward_pass(inputs.to_vec()));
}
