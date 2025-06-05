import typing

import paddle
# noinspection PyPep8Naming
import paddle.nn.functional as F


__use_cpu: bool = False


def get_device(use_cpu: bool = __use_cpu) -> str:
    return 'cpu' if (use_cpu or not paddle.is_compiled_with_cuda()) else 'gpu'


__device: str = get_device()


"""
Masked discrete Poisson equation (with arbitray Dirchilet boundary condition): 
        (I - bc_mask) A x = (I - bc_mask) f
             bc_mask    x =        bc_value  
Note for simplicity, 
    we preprocess bc_value s.t. bc_value == bc_mask bc_value, 
    and we have the exterior band of x be zero (trivial boundary condition). 
 
Masked Jacobi update: Step matrix P = -4I
        x' = (I - bc_mask) ( (I - P^-1 A) x                        + P^-1 f ) + bc_value
           = (I - bc_mask) ( F.conv2d(x, jacobi_kernel, padding=1) - 0.25 f ) + bc_value
           = util.jacobi_step(x)
"""


jacobi_kernel = paddle.to_tensor([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]], dtype='float32'
                                 ).reshape([1, 1, 3, 3]) / 4.0
if __device == 'gpu':
    jacobi_kernel = jacobi_kernel.cuda()

laplace_kernel = paddle.to_tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype='float32'
                                  ).reshape([1, 1, 3, 3])
if __device == 'gpu':
    laplace_kernel = laplace_kernel.cuda()

# Half-weight restriction
restriction_kernel = paddle.to_tensor([[0, 1, 0],
                                       [1, 4, 1],
                                       [0, 1, 0]], dtype='float32'
                                      ).reshape([1, 1, 3, 3]) / 8.0
if __device == 'gpu':
    restriction_kernel = restriction_kernel.cuda()


def initial_guess(bc_value: paddle.Tensor, bc_mask: paddle.Tensor, initialization: str) -> paddle.Tensor:
    """
    Assemble the initial guess of solution.
    """
    if initialization == 'random':
        # Use smaller random values for better stability
        routine = lambda x: paddle.uniform(shape=x.shape, dtype=x.dtype, min=-0.1, max=0.1)
    elif initialization == 'zero':
        routine = paddle.zeros_like
    else:
        raise NotImplementedError

    return (1 - bc_mask) * routine(bc_value) + bc_value


def jacobi_step(x: paddle.Tensor, bc_value: paddle.Tensor, bc_mask: paddle.Tensor, f: typing.Optional[paddle.Tensor]):
    """
    One iteration step of masked Jacobi iterative solver.
    """
    # Apply gradient scaling to prevent explosion
    scale = paddle.max(paddle.abs(x))
    if scale > 1e5:
        x = x / scale
        if f is not None:
            f = f / scale
        bc_value = bc_value / scale

    y = F.conv2d(x, jacobi_kernel, padding=1)

    if f is not None:
        y = y - 0.25 * f

    result = (1 - bc_mask) * y + bc_value
    
    if paddle.any(paddle.isnan(result)) or paddle.any(paddle.isinf(result)):
        print(f"[Warning] Jacobi step result contains NaN/Inf! Range: {paddle.min(result)} ~ {paddle.max(result)}")
    
    return result


def downsample2x(x: paddle.Tensor) -> paddle.Tensor:
    """
    Bilinear 2x-downsampling of an image of size 2^N + 1 is essentially direct injection.
    E.g., 257 -> 129 -> 65 -> ...

    Note: paddle.nn.UpsamplingBilinear2D is deprecated in favor of interpolate.
    It is equivalent to paddle.nn.functional.interpolate(..., mode='bilinear', align_corners=True).
    """
    new_size = (x.shape[-1] - 1) // 2 + 1
    y = F.interpolate(x, size=[new_size, new_size], mode='bilinear', align_corners=True)
    return y


def upsample2x(x: paddle.Tensor) -> paddle.Tensor:
    """
    Bilinear 2x-upsampling of an image of size 2^N + 1.
    E.g., 65 -> 129 -> 257 -> ...

    Note: paddle.nn.UpsamplingBilinear2D is deprecated in favor of interpolate.
    It is equivalent to paddle.nn.functional.interpolate(..., mode='bilinear', align_corners=True).
    """
    new_size = x.shape[-1] * 2 - 1
    y = F.interpolate(x, size=[new_size, new_size], mode='bilinear', align_corners=True)
    return y


def norm(x: paddle.Tensor) -> paddle.Tensor:
    """
    Vector norm on each batch.
    Note: We only deal with cases where channel == 1!
    :param x: (batch_size, channel, image_size, image_size)
    :return: (batch_size,)
    """
    y = x.reshape([x.shape[0], -1])
    return paddle.sqrt((y * y).sum(axis=1))


def absolute_residue(x: paddle.Tensor,
                     bc_mask: paddle.Tensor,
                     f: typing.Optional[paddle.Tensor],
                     reduction: str = 'norm') -> paddle.Tensor:
    """
    For a linear system Ax = f,
    the absolute residue is r = f - Ax,
    the absolute residual (norm) error eps = ||f - Ax||.
    """
    # eps of size (batch_size, channel (1), image_size, image_size)
    eps = F.conv2d(x, laplace_kernel, padding=1)

    if f is not None:
        eps = eps - f

    eps = eps * (1 - bc_mask)
    eps = eps.reshape([eps.shape[0], -1])            # of size (batch_size, image_size ** 2)

    if reduction == 'norm':
        error = norm(eps)                            # of size (batch_size,)
    elif reduction == 'mean':
        error = paddle.abs(eps).mean(axis=1)         # of size (batch_size,)
    elif reduction == 'max':
        error = paddle.abs(eps).max(axis=1)          # of size (batch_size,)
    elif reduction == 'none':
        error = -eps                                 # of size (batch_size, image_size ** 2)
    else:
        raise NotImplementedError

    if paddle.any(paddle.isnan(error)) or paddle.any(paddle.isinf(error)):
        print(f"[Warning] Residue error contains NaN/Inf! Error range: {paddle.min(error)} ~ {paddle.max(error)}")
    
    return error


def relative_residue(x: paddle.Tensor,
                     bc_value: paddle.Tensor,
                     bc_mask: paddle.Tensor,
                     f: typing.Optional[paddle.Tensor]) -> typing.Tuple[paddle.Tensor, paddle.Tensor]:
    """
    For a linear system Ax = f, the relative residual error eps = ||f - Ax|| / ||f||.
    :return: abs_residual_error, relative_residual_error
    """
    numerator: paddle.Tensor = absolute_residue(x, bc_mask, f, reduction='norm')  # norm of size (batch_size,)

    denominator: paddle.Tensor = bc_value                                         # (batch_size, image_size, image_size)

    if f is not None:
        denominator = denominator + f                                            # (batch_size, image_size, image_size)

    denominator = norm(denominator)

    return numerator, numerator / denominator                                    # (batch_size,)
