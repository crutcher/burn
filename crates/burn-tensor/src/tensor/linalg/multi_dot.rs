use std::iter::iter;
use crate::Tensor;
use crate::kind::Numeric;

/// Multiplies a list of tensors together using matrix multiplication.
pub fn multi_dot<const D: usize, K: Numeric>(tensors: Vec<Tensor<D, K>>) -> Tensor<D, K> {
    tensors.into_iter().reduce(|acc, tensor| acc.matmul(tensor)).unwrap()
}


/*
def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """
    Return a np.array that encodes the optimal order of multiplications.

    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.

    Also return the cost matrix if `return_costs` is `True`

    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.

        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])

    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = zeros((n, n), dtype=double)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = empty((n, n), dtype=intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s
 */

pub fn multi_dot_matrix_chain_order<const D: usize, K: Numeric>(tensors: &[Tensor<D, K>]) {
    let n = tensors.len();

    let p = {
        let mut p = tensors.iter().map(|tensor| tensor.shape()[D-2]).collect::<Vec<_>>();
        p.push(tensors.last().unwrap().dims()[D-1]);
        p
    };

}