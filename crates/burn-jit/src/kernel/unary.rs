use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use cubecl::{
    calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*,
    tensor_vectorization_factor, unexpanded,
};

#[cube]
pub(crate) trait UnaryOp<C: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    /// Execute a unary operation.
    fn execute(_input: Line<C>, _options: &Self::Options) -> Line<C> {
        unexpanded!();
    }
}

#[cube(launch)]
pub(crate) fn unary_kernel<C: CubePrimitive, O: UnaryOp<C>>(
    input: &Tensor<Line<C>>,
    output: &mut Tensor<Line<C>>,
    options: &O::Options,
    #[comptime] rank: Option<u32>,
    #[comptime] to_contiguous: bool,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    if to_contiguous {
        let offset_input = index_offset_with_layout::<C, C>(
            input,
            output,
            offset_output,
            0,
            rank.unwrap_or_else(|| output.rank()),
            rank.is_some(),
        );

        output[offset_output] = O::execute(input[offset_input], options);
    } else {
        output[offset_output] = O::execute(input[offset_output], options);
    }
}

pub(crate) fn launch_unary<R: JitRuntime, E: JitElement, O: UnaryOp<E>, F>(
    tensor: JitTensor<R>,
    options: F,
) -> JitTensor<R>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> F: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options, R>,
{
    let ndims = tensor.shape.num_dims();
    // Vectorization is only enabled when the last dimension is contiguous.
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &tensor.shape.dims, &tensor.strides, ndims - 1);

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / vectorization_factor as usize, cube_dim);
    let is_contiguous = tensor.is_contiguous();

    if tensor.can_mut() && tensor.is_contiguous_buffer() {
        unary_kernel::launch::<E, O, R>(
            &client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(vectorization_factor),
            TensorArg::alias(0),
            options(&()),
            None,
            false,
        );

        tensor
    } else {
        let output = empty_device::<R, E>(
            tensor.client.clone(),
            tensor.device.clone(),
            tensor.shape.clone(),
        );

        unary_kernel::launch::<E, O, R>(
            &client,
            cube_count,
            CubeDim::default(),
            tensor.as_tensor_arg::<E>(vectorization_factor),
            output.as_tensor_arg::<E>(vectorization_factor),
            options(&()),
            Some(ndims as u32),
            !is_contiguous,
        );
        output
    }
}

macro_rules! unary_op {
    ($name:ident, $elem:ident, $expand:expr) => {
        struct $name;

        impl<C: $elem> UnaryOp<C> for $name {
            type Options = ();

            #[allow(clippy::redundant_closure_call)]
            fn __expand_execute(
                context: &mut CubeContext,
                input: <Line<C> as CubeType>::ExpandType,
                _options: <Self::Options as CubeType>::ExpandType,
            ) -> <Line<C> as CubeType>::ExpandType {
                $expand(context, input)
            }
        }
    };
    (scalar $name:ident, $elem:ident, $expand:expr) => {
        struct $name;

        impl<C: $elem> UnaryOp<C> for $name {
            type Options = C;

            #[allow(clippy::redundant_closure_call)]
            fn __expand_execute(
                context: &mut CubeContext,
                input: <Line<C> as CubeType>::ExpandType,
                scalar: C::ExpandType,
            ) -> <Line<C> as CubeType>::ExpandType {
                $expand(context, input, scalar)
            }
        }
    };
    (float($tensor:expr) => $exp:expr) => {{
        unary_op!(Op, Float, $exp);
        launch_unary::<R, F, Op, _>($tensor, |_| ())
    }};
    (int($tensor:expr) => $exp:expr) => {{
        unary_op!(Op, Numeric, $exp);
        launch_unary::<R, I, Op, _>($tensor, |_| ())
    }};
    (numeric($tensor:expr) => $exp:expr) => {{
        unary_op!(Op, Numeric, $exp);
        launch_unary::<R, E, Op, _>($tensor, |_| ())
    }};
    (numeric($tensor:expr, $scalar:expr) => $exp:expr) => {{
        unary_op!(scalar Op, Numeric, $exp);
        launch_unary::<R, E, Op, _>($tensor, |_| ScalarArg::new($scalar))
    }};
    (float($tensor:expr, $scalar:expr) => $exp:expr) => {{
        unary_op!(scalar Op, Float, $exp);
        launch_unary::<R, F, Op, _>($tensor, |_| ScalarArg::new($scalar))
    }};
}

pub(crate) use unary_op;
