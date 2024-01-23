use crate::utils::*;
use crate::neuron::*;

#[allow(dead_code)]
pub struct Dense<T: Float> {
    neurons: Vec<Neuron<T>>,
    activation: usize,
}