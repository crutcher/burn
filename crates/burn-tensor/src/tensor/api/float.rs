use crate::FloatDType;
use crate::Tensor;
use crate::quantization::{QuantScheme, QuantizationParameters};
use crate::tensor::backend::Backend;
use crate::tensor::stats;
use crate::tensor::{Distribution, TensorData};
use crate::{Int, TensorPrimitive};

impl<const D: usize, B> Tensor<B, D>
where
    B: Backend,
{
    /// Applies element wise exponential operation.
    ///
    /// $y_i = e^{x_i}$
    pub fn exp(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_exp(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise natural log operation *ln*.
    ///
    /// $y_i = \log_e\(x_i\)$
    pub fn log(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_log(
            self.primitive.tensor(),
        )))
    }

    /// Applies the natural logarithm of one plus the input tensor, element-wise.
    ///
    /// $y_i = \log_e\(x_i + 1\)$
    pub fn log1p(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_log1p(
            self.primitive.tensor(),
        )))
    }

    /// Applies the [error function](https://en.wikipedia.org/wiki/Error_function) element wise.
    ///
    /// $y_i = \text{erf}\(x_i\)$
    ///
    /// The error function is defined as:
    ///
    /// $$\text{erf}\(x\) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$$
    pub fn erf(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_erf(
            self.primitive.tensor(),
        )))
    }

    /// Applies [reciprocal operation](https://en.wikipedia.org/wiki/Multiplicative_inverse)
    /// (or multiplicative inverse) element wise.
    ///
    /// $y_i = \frac{1}{x_i}$
    pub fn recip(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_recip(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise root square operation.
    ///
    /// $y_i = \sqrt{x_i}$
    pub fn sqrt(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_sqrt(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise cosine operation.
    ///
    /// $y_i = \cos\(x_i\)$
    pub fn cos(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_cos(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise sine operation.
    ///
    /// $y_i = \sin\(x_i\)$
    pub fn sin(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_sin(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise tangent operation.
    ///
    /// $y_i = \tan\(x_i\)$
    pub fn tan(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_tan(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise hyperbolic cosine operation.
    ///
    /// $y_i = \cosh\(x_i\)$
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 3>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.cosh()); // [1.0, 1.5430, 3.7621]
    /// }
    /// ```
    pub fn cosh(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_cosh(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise hyperbolic sine operation.
    ///
    /// $y_i = \sinh\(x_i\)$
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 3>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.sinh()); // [0.0, -1.1752, 3.6269]
    /// }
    /// ```
    pub fn sinh(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_sinh(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise hyperbolic tangent operation.
    ///
    /// $y_i = \tanh\(x_i\)$
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///
    ///     let tensor = Tensor::<B, 3>::from_data([0.0, -1.0, 2.0], &device);
    ///     println!("{}", tensor.sinh()); // [0.0, -0.7616, 0.9640]
    /// }
    /// ```
    pub fn tanh(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_tanh(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise round operation.
    ///
    /// This function implements the [round half to even](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even)
    /// strategy, with halfway cases rounded to the nearest even integer value.
    pub fn round(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_round(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise floor operation.
    pub fn floor(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_floor(
            self.primitive.tensor(),
        )))
    }

    /// Applies element wise ceil operation.
    pub fn ceil(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_ceil(
            self.primitive.tensor(),
        )))
    }

    /// Create a tensor from floats (f32) on a given device.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let _ = Tensor::<B, 1>::from_floats([1.0, 2.0], &device);
    ///     let _ = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    /// }
    /// ```
    pub fn from_floats<A: Into<TensorData>>(floats: A, device: &B::Device) -> Self {
        Self::from_data(floats.into().convert::<f32>(), device)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data
    /// cast to Integer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let float_tensor = Tensor::<B, 1>::from_floats([1.0, 2.0], &device);
    ///     let int_tensor = float_tensor.int();
    /// }
    /// ```
    pub fn int(self) -> Tensor<B, D, Int> {
        Tensor::new(B::float_into_int(self.primitive.tensor()))
    }

    /// Returns a new tensor with the same shape and device as the current tensor filled random
    /// values sampled from the given distribution.
    pub fn random_like(&self, distribution: Distribution) -> Self {
        Tensor::new(TensorPrimitive::Float(B::float_random(
            self.shape(),
            distribution,
            &self.device(),
        )))
    }

    /// Calculate the variance along the given dimension.
    pub fn var(self, dim: usize) -> Self {
        stats::var(self, dim)
    }

    /// Calculate the variance along the given dimension without applying the Bessel’s correction.
    pub fn var_bias(self, dim: usize) -> Self {
        stats::var_bias(self, dim)
    }

    /// Calculate the variance along the given dimension and also returns the mean.
    pub fn var_mean(self, dim: usize) -> (Self, Self) {
        let mean = self.clone().mean_dim(dim);
        let var = stats::var_with_mean(self, mean.clone(), dim);
        (var, mean)
    }

    /// Calculate the variance along the given dimension without applying the Bessel’s correction and also returns the mean.
    pub fn var_mean_bias(self, dim: usize) -> (Self, Self) {
        let mean = self.clone().mean_dim(dim);
        let var = stats::var_with_mean_bias(self, mean.clone(), dim);
        (var, mean)
    }

    /// Converts a tensor to the specified floating point data type.
    ///
    /// # Warning
    /// Most backends don't have automatic type promotion at this time, so make sure that all tensors
    /// have the same floating point precision data type for operations multiple input tensors (e.g., binary ops).
    pub fn cast<F: Into<FloatDType>>(self, dtype: F) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::Float(B::float_cast(
            self.primitive.tensor(),
            dtype.into(),
        )))
    }

    /// Detach the current tensor from the autodiff graph.
    ///
    /// This function does nothing when autodiff is not enabled.
    /// This can be used in batchers or elsewhere to ensure that previous operations are not
    /// considered in the autodiff graph.
    pub fn detach(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_detach(
            self.primitive.tensor(),
        )))
    }

    /// Mark the tensor to keep gradients during the backward pass.
    ///
    /// This function does nothing when autodiff is not enabled.
    pub fn require_grad(self) -> Self {
        self.set_require_grad(true)
    }

    /// Returns true if the tensor requires gradients during the backward pass.
    pub fn is_require_grad(&self) -> bool {
        match &self.primitive {
            TensorPrimitive::Float(tensor) => B::float_is_require_grad(tensor),
            TensorPrimitive::QFloat(tensor) => B::q_is_require_grad(tensor),
        }
    }

    /// Mark the tensor as tracked or untracked depending on the require_grad argument.
    /// When tracked, the gradients will be available after the backward pass.
    ///
    /// This function does nothing when autodiff is not enabled.
    pub fn set_require_grad(self, require_grad: bool) -> Self {
        let primitive = match self.primitive {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_set_require_grad(tensor, require_grad))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_set_require_grad(tensor, require_grad))
            }
        };
        Self::new(primitive)
    }

    /// Applies the relu function to the tensor.
    pub(crate) fn relu(self) -> Self {
        Self::new(TensorPrimitive::Float(B::relu(self.primitive.tensor())))
    }

    /// Calculate covaraince matrix between different entries alongside a given dimension.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the square matrix.
    /// * `correction_factor` - Is usually 1 for samples and 0 for population.
    pub fn cov(self, dim: usize, correction_factor: usize) -> Tensor<B, D> {
        let n = self.dims()[dim];
        let centered = (self.clone() - self.mean_dim(dim)).swap_dims(dim, 0);
        centered
            .clone()
            .transpose()
            .matmul(centered)
            .div_scalar(n as f32 - correction_factor as f32)
    }

    /// Convert the tensor to a lower precision data type based on the quantization scheme.
    ///
    /// # Arguments
    ///
    /// * `scheme` - The quantization scheme.
    /// * `qparams` - The pre-computed quantization parameters.
    ///
    /// # Returns
    ///
    /// The quantized tensor.
    pub fn quantize(
        self,
        scheme: &QuantScheme,
        qparams: QuantizationParameters<B>,
    ) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::QFloat(B::quantize(
            self.primitive.tensor(),
            scheme,
            qparams.into(),
        )))
    }

    /// Dynamically convert the tensor to a lower precision data type based on the quantization scheme.
    ///
    /// # Arguments
    ///
    /// * `scheme` - The quantization scheme.
    ///
    /// # Returns
    ///
    /// The quantized tensor.
    ///
    /// # Notes
    /// This uses [min-max calibration](crate::quantization::Calibration::MinMax).
    pub fn quantize_dynamic(self, scheme: &QuantScheme) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::QFloat(B::quantize_dynamic(
            self.primitive.tensor(),
            scheme,
        )))
    }

    /// Convert the tensor back to a higher precision data type.
    ///
    /// If the tensor is not quantized, its value is simply returned.
    ///
    /// # Returns
    ///
    /// The dequantized tensor.
    pub fn dequantize(self) -> Tensor<B, D> {
        Tensor::new(TensorPrimitive::Float(self.primitive.tensor()))
    }
}
