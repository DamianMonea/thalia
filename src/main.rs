use rand::Rng;
use thalia::neuron::*;
use thalia::activations::*;

fn main() {

    // Build test inputs
    let mut rng = rand::thread_rng();
    let shape: (usize, usize) = (10, 10);
    let mut input_batch: Vec<Vec<f64>> = Vec::with_capacity(shape.0);
    for i in 0..shape.0 {
        input_batch.push(Vec::with_capacity(shape.1));
        for _ in 0..shape.1 {
            input_batch[i].push(rng.gen_range(-1.0..1.0));
        }
    }
    
    // Build test weights
    let mut weights: Vec<f64> = Vec::with_capacity(shape.1);
    for _ in 0..shape.1 {
        weights.push(rng.gen_range(-1.0..1.0));
    }

    println!("Inputs:");
    for i in 0..shape.0 {
        println!("{:?}", input_batch[i]);
    }
    println!("Weights:");
    println!("{:?}", weights);
    
    let mut n1: Neuron<f64> = Neuron::new(3, relu::<f64>, Some(3.));
    n1.set_weights(weights.to_vec()).unwrap();
    println!("Output:");
    println!("{:?}", n1.forward_pass(&input_batch));

    let mut n2: Neuron<f64> = Neuron::new(3, badrelu::<f64>, Some(3.));
    n2.set_weights(weights.to_vec()).unwrap();
    println!("Bad Output:");
    println!("{:?}", n2.forward_pass(&input_batch));
}
