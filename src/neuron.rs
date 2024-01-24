use std::ops::{Mul, Add};
use crate::utils::*;

pub struct Neuron<T: Float>
{
    weights: Vec<T>,
    activation: fn(T) -> T,
    bias: T,
}

pub trait New<T: Float> 
where
    T: PartialOrd + Copy + From<u8>,
{
    fn new(num_weights: usize, activation: fn(T)->T, bias: Option<T>) -> Self;
    fn set_weights(&mut self, weights: Vec<T>) -> Result<bool, String>;
}

impl<T: Float> New<T> for Neuron<T> 
where
    T: PartialOrd + Copy + From<u8>,
{
    /// Create a new Neuron.
    /// 
    /// # Examples
    /// ```
    /// let n1: Neuron<f32> = Neuron::new(10, relu::<f64>);
    /// let n2: Neuron<f64> = Neuron::new(20, sigmoid::<f64>, Some(2.));
    /// ```
    fn new(num_weights: usize, activation: fn(T)->T, bias: Option<T>) -> Self {
        Neuron {
            weights: Vec::with_capacity(num_weights),
            activation: activation,
            bias: bias.unwrap_or(T::from(1))
        }
    }

    /// Set the weights for a Neuron
    /// 
    /// # Examples
    /// ```
    /// let n1: Neuron<f32> = Neuron::new(3, relu::<f64>);
    /// n1.set_weights([1.2, -0.5, 0.42].to_vec());
    /// ```
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

pub trait ForwardPass<T: Float>
{
    fn forward_pass(self, inputs: &Vec<Vec<T>>) -> Vec<T>;
}

impl<T: Float> ForwardPass<T> for Neuron<T>
where
    T: PartialOrd + Copy + From<u8> + Mul<Output = T> + Add<Output = T> + From<f64> + Into<f64>, for<'a> &'a T: Mul<&'a T>
{
    fn forward_pass(self, inputs: &Vec<Vec<T>>) -> Vec<T> {
        let mut res: Vec<T> = Vec::with_capacity(inputs.len());
        for input in inputs {
            res.push((self.activation)(T::from(input.iter().zip(self.weights.iter()).map(|(&x, &y)| x * y).map(Into::into).sum::<f64>() + self.bias.into())));
        }
        res   
    }
}