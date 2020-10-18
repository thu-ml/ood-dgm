import tensorflow as tf
import numpy as np
from pixel_cnn_pp.nn import *


def logistic_score(x, mu, inv_s):
    """
    score fn = 1/s * (-1 + e^{-(x-mu)/s}) / (1 + e^{-(x-mu)/s})
    = 1/s * (-sigm((x-mu)/s) + sigm(-(x-mu)/s))
    """
    assert x.shape.ndims == mu.shape.ndims == inv_s.shape.ndims
    xn = (x-mu) * inv_s
    score = (tf.nn.sigmoid(-xn) - tf.nn.sigmoid(xn)) * inv_s
    return tf.where(
        x < -0.999, inv_s,
        tf.where(
            x > 0.999, -inv_s,
            score
        ))


def discretized_mix_logistic_loss_(x, l, fake_3channels=False, clip=False):
    """
    log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    :input x: Target, tensor of size [N, H, W, C=3]
    :input l: Parameterizes the model distribution, tensor of size [N, H, W, (1+3C)*nr_mix]
    :return: 
        log_probs_pixelwise: [N,H,W], log p(x_{h,w}|x_{<(h,w)}) (collapsed in the channel axis)
        ar_residual: [N,H,W,C], x_{h,w,c} - E(x_{h,w,c}|<...)
    """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    # |nr_mix: logit_probs|9nr_mix|
    # 9nr_mix part: for each channel, |means|log_scales|coeffs|
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    # m2,m3: [N,H,W,1,nr_mix]: parameterizes (as a mixture model) AR mean conditioned on previous channels
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)  # [N,H,W,C=3,nr_mix]
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
    
    log_probs_given_c_mi = log_probs  # [N,H,W,C,K] p(Red|Context,MI),p(Green|Red,Context,MI),p(Blue|R,G,C,MI)
    
    log_mixture_coef = log_prob_from_logits(logit_probs)
    log_probs = tf.reduce_sum(log_probs,3) + log_mixture_coef
    log_probs_pixelwise = log_sum_exp(log_probs)
    
    if clip:
        means1 = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)  # [N,H,W,C=3,nr_mix]
        centered_x = x - tf.clip_by_value(means1, -1., 1.)

    p_mi_given_c = tf.exp(log_mixture_coef)  # P(MixtureIndicator_{i,j} | Context_{i,j})
    ar_residual_r = tf.reduce_sum(centered_x[:,:,:,0] * p_mi_given_c, axis=-1, keepdims=True)
    p_mi_given_cr = tf.nn.softmax(log_mixture_coef + log_probs_given_c_mi[:,:,:,0], axis=-1)
    ar_residual_g = tf.reduce_sum(centered_x[:,:,:,1] * p_mi_given_cr, axis=-1, keepdims=True)
    p_mi_given_crg = tf.nn.softmax(log_mixture_coef + log_probs_given_c_mi[:,:,:,0] + log_probs_given_c_mi[:,:,:,1], axis=-1)
    ar_residual_b = tf.reduce_sum(centered_x[:,:,:,2] * p_mi_given_crg, axis=-1, keepdims=True)
    ar_residual = tf.concat([ar_residual_r, ar_residual_g, ar_residual_b], axis=-1)
    
    def mixture_score(logp_cwise, score_cwise, mixt_coef):
        assert logp_cwise.shape.ndims == score_cwise.shape.ndims == mixt_coef.shape.ndims
        p_cwise = tf.exp(logp_cwise + 1e-9)
        return tf.reduce_sum(mixt_coef * score_cwise, axis=-1) / tf.reduce_sum(mixt_coef * p_cwise, axis=-1)
        
    channelwise_score = [
        mixture_score(log_probs_given_c_mi[...,0,:], logistic_score(x[...,0,:],means[...,0,:],inv_stdv[...,0,:]), p_mi_given_c),
        mixture_score(log_probs_given_c_mi[...,1,:], logistic_score(x[...,1,:],means[...,1,:],inv_stdv[...,1,:]), p_mi_given_cr),
        mixture_score(log_probs_given_c_mi[...,2,:], logistic_score(x[...,2,:],means[...,2,:],inv_stdv[...,2,:]), p_mi_given_crg)
    ]
    channelwise_score = tf.concat([c[...,None] for c in channelwise_score], axis=-1)
    
    return log_probs_pixelwise, ar_residual, channelwise_score


def value_and_gradient(f,
                       xs,
                       output_gradients=None,
                       use_gradient_tape=False,
                       name=None):
  """Computes `f(*xs)` and its gradients wrt to `*xs`.
  Args:
    f: Python `callable` to be differentiated. If `f` returns a scalar, this
      scalar will be differentiated. If `f` returns a tensor or list of tensors,
      by default a scalar will be computed by adding all their values to produce
      a single scalar. If desired, the tensors can be elementwise multiplied by
      the tensors passed as the `dy` keyword argument to the returned gradient
      function.
    xs: Python list of parameters of `f` for which to differentiate. (Can also
      be single `Tensor`.)
    output_gradients: A `Tensor` or list of `Tensor`s the same size as the
      result `ys = f(*xs)` and holding the gradients computed for each `y` in
      `ys`. This argument is forwarded to the underlying gradient implementation
      (i.e., either the `grad_ys` argument of `tf.gradients` or the
      `output_gradients` argument of `tf.GradientTape.gradient`).
    use_gradient_tape: Python `bool` indicating that `tf.GradientTape` should be
      used regardless of `tf.executing_eagerly()` status.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., `'value_and_gradient'`).
  Returns:
    y: `y = f(*xs)`.
    dydx: Gradient of `y` wrt each of `xs`.
  """
  with tf.name_scope(name or 'value_and_gradient'):
    xs, is_xs_list_like = _prepare_args(xs)
    y = f(*xs)
    dydx = tf.gradients(ys=y, xs=xs, grad_ys=output_gradients)
    if not is_xs_list_like:
      dydx = dydx[0]
    return y, dydx


def diag_jacobian(xs,
                  ys=None,
                  sample_shape=None,
                  fn=None,
                  parallel_iterations=10,
                  name=None):
  """Computes diagonal of the Jacobian matrix of `ys=fn(xs)` wrt `xs`.
    If `ys` is a tensor or a list of tensors of the form `(ys_1, .., ys_n)` and
    `xs` is of the form `(xs_1, .., xs_n)`, the function `jacobians_diag`
    computes the diagonal of the Jacobian matrix, i.e., the partial derivatives
    `(dys_1/dxs_1,.., dys_n/dxs_n`). For definition details, see
    https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
  #### Example
  ##### Diagonal Hessian of the log-density of a 3D Gaussian distribution
  In this example we sample from a standard univariate normal
  distribution using MALA with `step_size` equal to 0.75.
  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import numpy as np
  tfd = tfp.distributions
  dtype = np.float32
  with tf.Session(graph=tf.Graph()) as sess:
    true_mean = dtype([0, 0, 0])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 2, 0.25], [0.25, 0.25, 3]])
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)
    # Assume that the state is passed as a list of tensors `x` and `y`.
    # Then the target function is defined as follows:
    def target_fn(x, y):
      # Stack the input tensors together
      z = tf.concat([x, y], axis=-1) - true_mean
      return target.log_prob(z)
    sample_shape = [3, 5]
    state = [tf.ones(sample_shape + [2], dtype=dtype),
             tf.ones(sample_shape + [1], dtype=dtype)]
    fn_val, grads = tfp.math.value_and_gradient(target_fn, state)
    # We can either pass the `sample_shape` of the `state` or not, which impacts
    # computational speed of `diag_jacobian`
    _, diag_jacobian_shape_passed = diag_jacobian(
        xs=state, ys=grads, sample_shape=tf.shape(fn_val))
    _, diag_jacobian_shape_none = diag_jacobian(
        xs=state, ys=grads)
    diag_jacobian_shape_passed_ = sess.run(diag_jacobian_shape_passed)
    diag_jacobian_shape_none_ = sess.run(diag_jacobian_shape_none)
  print('hessian computed through `diag_jacobian`, sample_shape passed: ',
        np.concatenate(diag_jacobian_shape_passed_, -1))
  print('hessian computed through `diag_jacobian`, sample_shape skipped',
        np.concatenate(diag_jacobian_shape_none_, -1))
  ```
  Args:
    xs: `Tensor` or a python `list` of `Tensors` of real-like dtypes and shapes
      `sample_shape` + `event_shape_i`, where `event_shape_i` can be different
      for different tensors.
    ys: `Tensor` or a python `list` of `Tensors` of the same dtype as `xs`. Must
        broadcast with the shape of `xs`. Can be omitted if `fn` is provided.
    sample_shape: A common `sample_shape` of the input tensors of `xs`. If not,
      provided, assumed to be `[1]`, which may result in a slow performance of
      `jacobians_diag`.
    fn: Python callable that takes `xs` as an argument (or `*xs`, if it is a
      list) and returns `ys`. Might be skipped if `ys` is provided and
      `tf.enable_eager_execution()` is disabled.
    parallel_iterations: `int` that specifies the allowed number of coordinates
      of the input tensor `xs`, for which the partial derivatives `dys_i/dxs_i`
      can be computed in parallel.
    name: Python `str` name prefixed to `Ops` created by this function.
      Default value: `None` (i.e., "diag_jacobian").
  Returns:
    ys: a list, which coincides with the input `ys`, when provided.
      If the input `ys` is None, `fn(*xs)` gets computed and returned as a list.
    jacobians_diag_res: a `Tensor` or a Python list of `Tensor`s of the same
      dtypes and shapes as the input `xs`. This is the diagonal of the Jacobian
      of ys wrt xs.
  Raises:
    ValueError: if lists `xs` and `ys` have different length or both `ys` and
      `fn` are `None`, or `fn` is None in the eager execution mode.
  """
  with tf.name_scope(name or 'jacobians_diag'):
    if sample_shape is None:
      sample_shape = [1]
    # Output Jacobian diagonal
    jacobians_diag_res = []
    # Convert input `xs` to a list
    xs = list(xs) if _is_list_like(xs) else [xs]
    xs = [tf.convert_to_tensor(x) for x in xs]
    if not tf.executing_eagerly():
      if ys is None:
        if fn is None:
          raise ValueError('Both `ys` and `fn` can not be `None`')
        else:
          ys = fn(*xs)
      # Convert ys to a list
      ys = list(ys) if _is_list_like(ys) else [ys]
      if len(xs) != len(ys):
        raise ValueError('`xs` and `ys` should have the same length')
      for y, x in zip(ys, xs):
        # Broadcast `y` to the shape of `x`.
        y_ = y + tf.zeros_like(x)
        # Change `event_shape` to one-dimension
        y_ = tf.reshape(y, tf.concat([sample_shape, [-1]], -1))

        # Declare an iterator and tensor array loop variables for the gradients.
        n = tf.size(x) / tf.cast(tf.reduce_prod(sample_shape), dtype=tf.int32)
        n = tf.cast(n, dtype=tf.int32)
        loop_vars = [
            0,
            tf.TensorArray(x.dtype, n)
        ]

        def loop_body(j):
          """Loop function to compute gradients of the each direction."""
          # Gradient along direction `j`.
          res = tf.gradients(ys=y_[..., j], xs=x)[0]  # pylint: disable=cell-var-from-loop
          if res is None:
            # Return zero, if the gradient is `None`.
            res = tf.zeros(tf.concat([sample_shape, [1]], -1),
                           dtype=x.dtype)  # pylint: disable=cell-var-from-loop
          else:
            # Reshape `event_shape` to 1D
            res = tf.reshape(res, tf.concat([sample_shape, [-1]], -1))
            # Add artificial dimension for the case of zero shape input tensor
            res = tf.expand_dims(res, 0)
            res = res[..., j]
          return res  # pylint: disable=cell-var-from-loop

        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        _, jacobian_diag_res = tf.while_loop(
            cond=lambda j, _: j < n,  # pylint: disable=cell-var-from-loop
            body=lambda j, result: (j + 1, result.write(j, loop_body(j))),
            loop_vars=loop_vars,
            parallel_iterations=parallel_iterations)

        shape_x = tf.shape(x)
        # Stack gradients together and move flattened `event_shape` to the
        # zero position
        reshaped_jacobian_diag = tf.transpose(a=jacobian_diag_res.stack())
        # Reshape to the original tensor
        reshaped_jacobian_diag = tf.reshape(reshaped_jacobian_diag, shape_x)
        jacobians_diag_res.append(reshaped_jacobian_diag)

    else:
      if fn is None:
        raise ValueError('`fn` can not be `None` when eager execution is '
                         'enabled')
      if ys is None:
        ys = fn(*xs)

      def fn_slice(i, j):
        """Broadcast y[i], flatten event shape of y[i], return y[i][..., j]."""
        def fn_broadcast(*state):
          res = fn(*state)
          res = list(res) if _is_list_like(res) else [res]
          if len(res) != len(state):
            res *= len(state)
          res = [tf.reshape(r + tf.zeros_like(s),
                            tf.concat([sample_shape, [-1]], -1))
                 for r, s in zip(res, state)]
          return res
        # Expand dimensions before returning in order to support 0D input `xs`
        return lambda *state: tf.expand_dims(fn_broadcast(*state)[i], 0)[..., j]

      def make_loop_body(i, x):
        """Loop function to compute gradients of the each direction."""
        def _fn(j, result):
          res = value_and_gradient(fn_slice(i, j), xs)[1][i]
          if res is None:
            res = tf.zeros(tf.concat([sample_shape, [1]], -1), dtype=x.dtype)
          else:
            res = tf.reshape(res, tf.concat([sample_shape, [-1]], -1))
            res = res[..., j]
          return j + 1, result.write(j, res)
        return _fn

      for i, x in enumerate(xs):
        # Declare an iterator and tensor array loop variables for the gradients.
        n = tf.size(x) / tf.cast(tf.reduce_prod(sample_shape), dtype=tf.int32)
        n = tf.cast(n, dtype=tf.int32)
        loop_vars = [
            0,
            tf.TensorArray(x.dtype, n)
        ]

        # Iterate over all elements of the gradient and compute second order
        # derivatives.
        _, jacobian_diag_res = tf.while_loop(
            cond=lambda j, _: j < n,
            body=make_loop_body(i, x),
            loop_vars=loop_vars,
            parallel_iterations=parallel_iterations)

        shape_x = tf.shape(x)
        # Stack gradients together and move flattened `event_shape` to the
        # zero position
        reshaped_jacobian_diag = tf.transpose(a=jacobian_diag_res.stack())
        # Reshape to the original tensor
        reshaped_jacobian_diag = tf.reshape(reshaped_jacobian_diag, shape_x)
        jacobians_diag_res.append(reshaped_jacobian_diag)

  return ys, jacobians_diag_res


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))


def tile_images(imgs):
    z = int(imgs.shape[0] ** 0.5)
    if z*z < imgs.shape[0]:
        imgs = np.concatenate([imgs, np.zeros_like(imgs[:(z+1)*(z+1) - imgs.shape[0]])], axis=0)
        z = z+1
    imgs = imgs.reshape([z,z]+list(imgs.shape[1:]))
    return np.concatenate(np.concatenate(imgs, axis=1), axis=1)