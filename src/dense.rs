mod neuron;
use crate::neuron::*;

pub struct Dense {
    neurons: Vec<Neuron>,
    activation: usize,
}