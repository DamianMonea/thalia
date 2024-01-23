use std::ops::{Mul, Add};

pub fn relu<T>(x: T) -> T
where
    T: PartialOrd + Copy + From<u8>,
{
    if x > T::from(0) {
        x
    } else {
        T::from(0)
    }
}

pub fn sigmoid<T>(x: T) -> T
where
    T: Into<f64>, T: From<f64>
{
    let x_f64: f64 = x.into();
    let sigmoid = 1.0 / (1.0 + (-x_f64).exp());
    sigmoid.into()
}