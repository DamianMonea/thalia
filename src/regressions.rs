extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray::concatenate;
use ndarray_linalg::Solve; // For solving linear equations

#[derive(Debug)]
struct LinearRegression {
    coefficients: Option<Array1<f64>>,
}

impl LinearRegression {
    // Constructor for LinearRegression
    fn new() -> Self {
        LinearRegression { coefficients: None }
    }

    // Method to train the model
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) {
        let ones = Array::ones((x.nrows(), 1));
        let X = concatenate![Axis(1), ones, x];

        // Calculate (X^T * X)
        let xtx = X.t().dot(&X);

        // Calculate (X^T * y)
        let xty = X.t().dot(y);

        // Solve for coefficients
        match xtx.solve_into(xty) {
            Ok(coeffs) => self.coefficients = Some(coeffs),
            Err(_) => panic!("Could not solve for coefficients"),
        }
    }

    // Method to make predictions
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        if let Some(ref coeffs) = self.coefficients {
            let ones = Array::ones((x.nrows(), 1));
            let X = concatenate![Axis(1), ones, x];
            X.dot(coeffs)
        } else {
            panic!("Model is not trained yet!");
        }
    }

    // Method to calculate mean squared error
    fn mean_squared_error(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let diff = y_true - y_pred;
        diff.mapv(|x| x.powi(2)).mean().unwrap()
    }
}