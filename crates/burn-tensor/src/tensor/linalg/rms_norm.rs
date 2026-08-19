use crate::tensor::Tensor;
use burn_std::DType;

/// Options for root-mean-square norm.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RmsNormOptions {
    /// Epsilon value for numerical stability.
    pub epsilon: f64,
}

impl Default for RmsNormOptions {
    fn default() -> Self {
        Self { epsilon: 1e-5 }
    }
}

impl RmsNormOptions {
    /// Sets epsilon value.
    pub fn with_eps(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Applies root-mean-square norm.
    pub fn rms_norm<const R: usize>(&self, x: Tensor<R>) -> Tensor<R> {
        rms_norm(x, self)
    }
}

/// Applies root-mean-square norm.
pub fn rms_norm<const R: usize>(x: Tensor<R>, options: &RmsNormOptions) -> Tensor<R> {
    let rms = x
        .clone()
        .cast(DType::F32)
        .square()
        .mean_dim(-1)
        .add_scalar(options.epsilon)
        .sqrt()
        .cast(x.dtype());

    x / rms
}

#[cfg(test)]
mod tests {
    use crate::tensor::Distribution;

    use super::*;

    #[test]
    fn test_rms_norm() {
        let device = Default::default();

        let x: Tensor<3> = Tensor::random([2, 3, 4], Distribution::Default, &device);
        let options = RmsNormOptions::default();

        let y = rms_norm(x.clone(), &options);

        let x_rms = x
            .clone()
            .square()
            .mean_dim(-1)
            .add_scalar(options.epsilon)
            .sqrt();
        let expected = x / x_rms;

        y.to_data().assert_eq(&expected.to_data(), true);
    }
}
