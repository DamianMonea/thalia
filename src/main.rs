use std::result::Result;
mod neuron;
mod activations;
use crate::neuron::*;
use crate::activations::*;

fn main() {
    let inputs: [f64; 3] = [1.2, 5.1, 2.1];
    let weights: [f64; 3] = [3.1, 2.1, 8.7];

    let mut n: Neuron<f64> = Neuron::new(3, sigmoid::<f64>, Some(3.));
    n.set_weights(weights.to_vec()).unwrap();
    println!("{}", n.forward_pass(inputs.to_vec()));
}
