use std::ops::{Mul, Add};
use std::iter::Sum;
use rand::prelude::*;
use crate::activations::*;


pub struct Neuron<T>
where
    T: PartialOrd + Copy + From<u8>,
{
    weights: Vec<T>,
    activation: fn(T) -> T,
    bias: T,
}

pub trait New<T> 
where
    T: PartialOrd + Copy + From<u8>,
{
    fn new(num_weights: usize, activation: fn(T)->T, bias: Option<T>) -> Self;
    fn set_weights(&mut self, weights: Vec<T>) -> Result<bool, String>;
}

pub trait ForwardPass<T> 
where
    T: PartialOrd + Copy + From<u8>,
{
    fn forward_pass(self, inputs: Vec<T>) -> T;
}

impl<T> New<T> for Neuron<T> 
where
    T: PartialOrd + Copy + From<u8>,
{
    fn new(num_weights: usize, activation: fn(T)->T, bias: Option<T>) -> Self {
        Neuron {
            weights: Vec::with_capacity(num_weights),
            activation: activation,
            bias: bias.unwrap_or(T::from(1))
        }
    }

    fn set_weights(&mut self, weights: Vec<T>) -> Result<bool, String>{
        let num_weights = weights.len();
        if self.weights.len() == 0 {
            for i in 0..num_weights {
                self.weights.push(weights[i]);
            }
        } else {
            for i in 0..num_weights {
                self.weights[i] = weights[i];
            }
        }
        Ok(true)
    }
}

impl<T> ForwardPass<T> for Neuron<T>
where
    T: PartialOrd + Copy + From<u8> + Mul<Output = T> + Add<Output = T> + From<f64> + Into<f64>, for<'a> &'a T: Mul<&'a T>
{
    fn forward_pass(self, inputs: Vec<T>) -> T {
        (self.activation)(T::from(inputs.iter().zip(self.weights.iter()).map(|(&x, &y)| x * y).map(Into::into).sum::<f64>() + self.bias.into()))
    }
}