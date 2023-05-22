import json
import math
from functools import wraps

import numpy as np
import torch
from sklearn.model_selection import LeaveOneOut

from src.emicore.utils import SingularGramError


KERNELS = {}


def register_kernel(name):
    def wrapped(kernel):
        kernel.__serialized_name__ = name
        KERNELS[name] = kernel
        return kernel
    return wrapped


def atomize(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().item()
    return obj


def cdiff(x1, x2):
    return x1[..., :, None, :] - x2[..., None, :, :]


def tdiff(x1, x2, p=2*math.pi, lscale=1):
    ''' Function to compute torus distance

    Parameters
    ----------
    x1 : obj:`numpy.ndarray`
        First set of points with shape (number of samples x dimensions).
    x2 : obj:`numpy.ndarray`
        Second set of points with shape (number of samples x dimensions).
    p : obj:`float`
        Periodicity of the manifold (torus).
    lscale : obj:`float`
        Lenghtscale parameter.
    Returns
    -------
    obj:`torch.tensor`
        Torus distance for all points in the 2 sets.
    '''
    torus_dist = (p / math.pi / lscale * torch.sin(math.pi / p * cdiff(x1, x2))) ** 2
    return torus_dist.sum(dim=-1) ** .5


class MultivariateNormal:
    '''MultivariateNormal, possibly with independent variables.

    Parameters
    ----------
    mean : obj:`torch.Tensor`
        One-dimensional Tensor with means. May have arbitrary batch dimension matching `covar`.
    covar : obj:`torch.Tensor`
        Two-dimensional Tensor of covariance matrix, or one-dimensional Tensor with the diagonal elements of the
        covariance matrix. If two-dimensional, must be square (indicates covariance) or have one dimension equal to 1
        (indicates diagonal). May have an arbiratry batch dimension, in which case two dimensions must be used to
        describe the covariance diagonal or full covariance.
    '''
    def __init__(self, mean, covar):
        if covar.ndim > 1 and not covar.shape[-2] == covar.shape[-1] and 1 not in covar.shape[-2:]:
            raise TypeError('Shape of covariance is invalid!')
        if (
            not ((covar.ndim in (1, 2) and mean.ndim == 1) or (covar.shape[:-2] == mean.shape[:-1]))
            or (max(covar.shape[-2:]) != mean.shape[-1])
        ):
            raise TypeError('Shapes of covariance and mean do not match!')
        self._mean = mean
        self._covar = covar
        self._covrank = None

    def __getitem__(self, key):
        return MultivariateNormal(self._mean[key], self._covar[key])

    @property
    def mean(self):
        '''Return the means.'''
        return self._mean

    @property
    def covar(self):
        '''Return the covariance matrix.'''
        if self._covar.ndim == 1:
            return torch.diagflat(self._covar)
        if 1 in self._covar.shape[-2:]:
            return torch.diag_embed(self._covar.flatten(start_dim=-2))
        return self._covar

    @property
    def var(self):
        '''Return the diagonal of the covariance matrix.'''
        if self._covar.ndim == 1:
            return self._covar
        if 1 in self._covar.shape[-2:]:
            return self._covar.flatten(start_dim=-2)
        return torch.diagonal(self._covar, dim1=-2, dim2=-1)

    @property
    def std(self):
        '''Return the square root of the diagonal of the covariance matrix.'''
        return self.var ** .5

    def sample(self, size=1):
        '''Return samples from the distribution.'''
        if isinstance(size, int):
            size = (size,)
        samples = torch.randn((*size, *self._mean.shape))
        if self._covar.ndim == 1 or self._covar.ndim == 2 and 1 in self._covar.shape:
            samples = samples * self.std + self._mean
        else:
            cholesky = torch.cholesky(self.covar)
            samples = (cholesky @ samples[..., None]).squeeze(-1) + self._mean
        return samples

    @property
    def stats(self):
        '''Return the means and the covariance matrix.'''
        return self.mean, self.covar

    @property
    def diag_stats(self):
        '''Return the means and the diagonal of the covariance matrix.'''
        return self.mean, self.var

    @property
    def covrank(self):
        '''Return the rank of the covariance matrix.'''
        if self._covrank is None:
            self._covrank = torch.linalg.matrix_rank(self.covar, atol=1e-10)
        return self._covrank

    def pdf(self, x):
        '''Probability density function, assuming independent variables.

        Parameters
        ----------
        x : float or obj:`torch.Tensor`
            One-dimensional tensor of inputs.

        Returns
        -------
        obj:`torch.Tensor`
            One-dimensional tensor with probabilities.
        '''
        mean, var = self.diag_stats
        var = var.clip(min=1e-30)
        return (-(x - mean) ** 2. / var / 2.).exp() / (var * 2 * math.pi) ** .5

    def cdf(self, x):
        '''Cumulative distribution function, assuming independent variables.

        Parameters
        ----------
        x : float or obj:`torch.Tensor`
            One-dimensional tensor of inputs.

        Returns
        -------
        obj:`torch.Tensor`
            One-dimensional tensor with cumulative probabilities.
        '''
        mean, var = self.diag_stats
        var = var.clip(min=1e-30)
        return (1. + torch.erf((x - mean) / (var * 2) ** .5)) / 2.


def flatargs(func):
    '''Create a wrapper which flattens the arguments before passing them to the true function.'''
    @wraps(func)
    def wrapped(self, x1, x2):
        return func(self, self.flat(x1), self.flat(x2))
    return wrapped


class Kernel:
    '''Base class for kernels.'''
    __kernel_params__ = tuple()

    def __init__(self, n_feature_dim=2):
        self.n_feature_dim = 2

    def flat(self, x):
        return x.flatten(start_dim=-self.n_feature_dim)

    def __call__(self, x1, x2):
        '''Compute kernel values.

        Parameters
        ----------
        x1 : obj:`numpy.ndarray`
            First set of points with shape (number of samples x dimensions).
        x2 : obj:`numpy.ndarray`
            Second set of points with shape (number of samples x dimensions).

        Returns
        -------
        obj:`torch.tensor`
            Gram matrix of kernel with shape (number of samples x number of samples).

        '''

    def __repr__(self):
        params = ', '.join(f'{key}={atomize(getattr(self, key))}' for key in self.__kernel_params__)
        return f'{self.__class__.__name__}({params})'

    def diag(self, x1):
        '''Compute diagonal kernel values.

        Parameters
        ----------
        x1 : obj:`numpy.ndarray`
            Set of points with shape (number of samples x dimensions).

        Returns
        -------
        obj:`torch.tensor`
            Diagonal Gram matrix of kernel of (x1, x1) with shape (number of samples).

        '''

    def param_dict(self):
        return {key: atomize(getattr(self, key)) for key in self.__kernel_params__}

    def serialize(self):
        return json.dumps((self.__serialized_name__, self.param_dict()))

    @staticmethod
    def deserialize(string):
        name, kwargs = json.loads(string)
        return KERNELS[name](**kwargs)

    def parameters(self):
        return (
            getattr(self, key) for key in self.__kernel_params__ if isinstance(getattr(self, key), torch.Tensor)
        )


@register_kernel('vqe')
class ShinKernel(Kernel):
    __kernel_params__ = ('gamma',)

    def __init__(self, sigma_0=1.0, gamma=1.0, n_feature_dim=2):
        super().__init__(n_feature_dim=n_feature_dim)
        self.sigma_0 = sigma_0
        self.gamma = torch.tensor(gamma)
        self.sigma_0_sq = sigma_0 ** 2

    @flatargs
    def __call__(self, x1, x2):
        gram_matrix = (self.gamma ** 2 + torch.cos(cdiff(x1, x2))).log() - (1 + self.gamma ** 2).log()
        kern = self.sigma_0_sq * gram_matrix.sum(axis=-1).exp()
        return kern

    def diag(self, x1):
        return torch.full(x1.shape[:-self.n_feature_dim], self.sigma_0_sq, dtype=x1.dtype)


@register_kernel('sharedvqe')
class SharedShinKernel(Kernel):
    __kernel_params__ = ('gamma', 'eta')

    def __init__(self, sigma_0=1.0, gamma=1.0, eta=1.0, share=1, n_feature_dim=2):
        super().__init__(n_feature_dim=n_feature_dim)
        self.sigma_0 = sigma_0
        self.gamma = torch.tensor(gamma)
        self.sigma_0_sq = sigma_0 ** 2
        self.eta = torch.tensor(eta)
        self.share = share

    @flatargs
    def __call__(self, x1, x2):
        s = torch.arange(1, self.share + 1)[..., :, None, None, None]
        numerator = self.gamma ** 2 + (self.eta ** (2 * s) * torch.cos(s * cdiff(x1, x2)[..., None, :, :, :])).sum(-3)
        denominator = self.gamma ** 2 + (self.eta ** (2 * s)).sum(-3)

        gram_matrix = numerator / denominator
        kern = self.sigma_0_sq * gram_matrix.prod(axis=-1)
        return kern

    def diag(self, x1):
        return torch.full(x1.shape[:-self.n_feature_dim], self.sigma_0_sq, dtype=x1.dtype)


@register_kernel('rbf')
class RBFKernel(Kernel):
    __kernel_params__ = ('gamma',)

    def __init__(self, sigma_0=1.0, gamma=1.0, n_feature_dim=2):
        super().__init__(n_feature_dim=n_feature_dim)
        self.sigma_0 = sigma_0
        self.gamma = torch.tensor(gamma)
        self.sigma_0_sq = sigma_0 ** 2

    @flatargs
    def __call__(self, x1, x2):
        exp_term = (cdiff(x1, x2) / self.gamma) ** 2
        kern = self.sigma_0_sq * torch.exp(-0.5 * exp_term.sum(-1))
        return kern

    def diag(self, x1):
        return torch.full(x1.shape[:-self.n_feature_dim], self.sigma_0_sq, dtype=x1.dtype)


@register_kernel('periodic')
class PeriodicKernel(Kernel):
    __kernel_params__ = ('gamma',)

    def __init__(self, sigma_0=1.0, gamma=1.0, n_feature_dim=2):
        super().__init__(n_feature_dim=n_feature_dim)
        self.sigma_0 = sigma_0
        self.gamma = torch.tensor(gamma)
        self.sigma_0_sq = sigma_0 ** 2

    @flatargs
    def __call__(self, x1, x2):
        dists = cdiff(x1, x2)
        kern = (torch.cos(dists) - 1.) / (2 * self.gamma) ** 2
        return self.sigma_0_sq * kern.sum(axis=-1).exp()

    def diag(self, x1):
        return torch.full(x1.shape[:-self.n_feature_dim], self.sigma_0_sq, dtype=x1.dtype)


@register_kernel('matern')
class MaternKernel(Kernel):
    __kernel_params__ = ('nu',)

    def __init__(self, nu=2.5, n_feature_dim=2):
        super().__init__(n_feature_dim=n_feature_dim)
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError('Only values 0.5, 1.5 and 2.5 are allowed for nu.')
        self.nu = nu

    @flatargs
    def __call__(self, x1, x2):
        dists = torch.cdist(x1, x2)
        if self.nu == 0.5:
            kern = torch.exp(-dists)
        elif self.nu == 1.5:
            kern = dists * math.sqrt(3)
            kern = (1. + kern) * torch.exp(-kern)
        elif self.nu == 2.5:
            kern = dists * math.sqrt(5)
            kern = (1. + kern + kern ** 2 / 3.0) * torch.exp(-kern)
        else:
            raise ValueError('Only values 0.5, 1.5 and 2.5 are allowed for nu.')

        return kern

    def diag(self, x1):
        return torch.full(x1.shape[:-self.n_feature_dim], self.sigma_0, dtype=x1.dtype)


@register_kernel('torusrbf')
class TorusRBFKernel(RBFKernel):
    @flatargs
    def __call__(self, x1, x2):
        exp_term = (tdiff(x1, x2) / self.gamma) ** 2
        kern = self.sigma_0 * torch.exp(-0.5 * exp_term)
        return kern


@register_kernel('torusmatern')
class TorusMaternKernel(MaternKernel):

    @flatargs
    def __call__(self, x1, x2):
        dists = tdiff(x1, x2)
        if self.nu == 0.5:
            kern = torch.exp(-dists)
        elif self.nu == 1.5:
            kern = dists * math.sqrt(3)
            kern = (1. + kern) * torch.exp(-kern)
        elif self.nu == 2.5:
            kern = dists * math.sqrt(5)
            kern = (1. + kern + kern ** 2 / 3.0) * torch.exp(-kern)
        else:
            raise ValueError('Only values 0.5, 1.5 and 2.5 are allowed for nu.')

        return kern


class GaussianProcess:
    '''A gaussian process with noise using cholesky decomposition.
    The mean function is constant zero.

    Parameters
    ----------
    x_train : obj:`numpy.ndarray`
        Initial training sample inputs.
    y_train : obj:`numpy.ndarray`
        Initial training sample function values.
    kernel : obj:`numpy.ndarray`
        The covariance function.
    reg : obj:`numpy.ndarray`
        Regularization parameter for training data.

    Attributes
    ----------
    x_train : obj:`numpy.ndarray`
        Current training sample inputs.
    y_train : obj:`numpy.ndarray`
        Current training sample function values.
    kernel : obj:`numpy.ndarray`
        The covariance function.
    mean : obj:`numpy.ndarray`
        Current mean values.
    covar : obj:`numpy.ndarray`
        Current prior covariance matrix.
    cholesky : obj:`numpy.ndarray`
        Current lower triangular cholesky decomposition of `covar`.
    cov_inv_y : obj:`numpy.ndarray`
        Current solution x for Kx = y, where K is the covariance matrix and y are the training sample function values.
    reg : float, optional
        Regularization parameter for training data. Default is 0.1.

    '''
    _state_attributes = (
        'x_train',
        'y_train',
        'kernel',
        'mean',
        'covar',
        'cholesky',
        'cov_inv_y',
        'reg',
    )

    def __init__(self, x_train, y_train, kernel, reg=0.1, mean=0.0):
        self.initialize(x_train, y_train, kernel, reg, mean=0.0)

    def __repr__(self):
        return f'{self.__class__.__name__}(size={len(self.x_train):d}, kernel={self.kernel}, reg={self.reg:.2e})'

    def state_dict(self):
        def detach(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().numpy()
            return obj
        state_dict = {key: detach(getattr(self, key)) for key in self._state_attributes}
        state_dict['kernel'] = state_dict['kernel'].serialize()
        return state_dict

    def load_state_dict(self, state_dict):
        def attach(obj):
            if isinstance(obj, np.ndarray):
                return torch.from_numpy(obj).clone()
            return obj
        state_dict = state_dict.copy()
        state_dict['kernel'] = Kernel.deserialize(state_dict['kernel'])
        for key in self._state_attributes:
            setattr(self, key, attach(state_dict[key]))

    @classmethod
    def from_state_dict(cls, state_dict):
        instance = object.__new__(cls)
        instance.load_state_dict(state_dict)
        return instance

    def initialize(self, x_train, y_train, kernel, reg=0.1, mean=0.0):
        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel
        self.reg = reg
        self._mean = mean[-1] if isinstance(mean, (torch.Tensor, np.ndarray)) else float(mean)
        self.mean = torch.full((len(x_train),), self._mean, dtype=x_train.dtype) if isinstance(mean, float) else mean
        self.covar = kernel(x_train, x_train)
        # self.covar.diagonal()[()] += reg
        self.covar = self.covar + torch.diag_embed(torch.tensor((self.reg,) * len(self.covar)))
        self.cholesky = torch.linalg.cholesky(self.covar, upper=False)
        self.cov_inv_y = torch.cholesky_solve(self.y_train[:, None], self.cholesky, upper=False)

    def posterior(self, x_test, noise_level=None, diag=False):
        '''Compute the predictive posterior distribution.

        Parameters
        ----------
        x_test : obj:`numpy.ndarray`
            Sample values for which to compute the predictive posterior.
        noise_level : float, optional
            Independent noise as gaussian variance to be added to the distribution.
        diag : bool, optional
            Wether only the diagonal of the covariance should be computed. Default is False.

        Returns
        -------
        distribution : obj:`MultivariateNormal`
            The predictive posterior as a multivariate normal distribution.

        '''
        kernel_test = self.kernel(self.x_train, x_test)

        post_mean = (kernel_test.transpose(-1, -2) @ self.cov_inv_y)[..., 0]
        cov_inv_test = torch.cholesky_solve(kernel_test, self.cholesky, upper=False)

        if diag:
            prior_covar = self.kernel.diag(x_test)
            post_covar = (prior_covar - (kernel_test * cov_inv_test).sum(-2))[..., None, :]
            if noise_level is not None:
                post_covar = post_covar + noise_level
        else:
            prior_covar = self.kernel(x_test, x_test)
            post_covar = prior_covar - kernel_test.transpose(-1, -2) @ cov_inv_test
            if noise_level is not None:
                # post_covar.diagonal()[()] += noise_level
                post_covar = post_covar + torch.diag_embed(torch.tensor((noise_level,) * len(x_test)))

        return MultivariateNormal(post_mean, post_covar)

    def peek_posterior(self, x_peek, x_test, y_peek=None, noise_level=None, diag=False):
        '''Compute the predictive posterior distribution.

        Parameters
        ----------
        x_peek : obj:`numpy.ndarray`
            Sample values assumed to be known.
        x_test : obj:`numpy.ndarray`
            Sample values for which to compute the predictive posterior.
        noise_level : float, optional
            Independent noise as gaussian variance to be added to the distribution.
        diag : bool, optional
            Wether only the diagonal of the covariance should be computed. Default is False.

        Returns
        -------
        distribution : obj:`MultivariateNormal`
            The predictive posterior as a multivariate normal distribution.

        '''
        # const wrt. peek
        kernel_peek = self.kernel(self.x_train, x_peek)
        peek_covar = self.kernel(x_peek, x_peek)
        peek_covar = peek_covar + torch.diag_embed(torch.full((peek_covar.shape[-1],), self.reg))
        L21, L22 = self.cholesky_append(kernel_peek, peek_covar)

        # const wrt. test
        kernel_train_test = self.kernel(self.x_train, x_test)
        kernel_peek_test = self.kernel(x_peek, x_test)

        # const wrt. test
        z_train = torch.linalg.solve_triangular(self.cholesky, kernel_train_test, upper=False)

        z_peek = torch.linalg.solve_triangular(L22, kernel_peek_test - L21 @ z_train, upper=False)
        cit_peek = torch.linalg.solve_triangular(L22.transpose(-1, -2), z_peek, upper=True)

        cit_train = torch.linalg.solve_triangular(
            self.cholesky.transpose(-1, -2),
            z_train - L21.transpose(-1, -2) @ cit_peek,
            upper=True
        )

        if diag:
            # const wrt. test
            prior_covar = self.kernel.diag(x_test)
            post_covar = (
                prior_covar
                - (kernel_train_test * cit_train).sum(-2)
                - (kernel_peek_test * cit_peek).sum(-2)
            )[..., None, :]
            if noise_level is not None:
                post_covar = post_covar + noise_level
        else:
            # const wrt. test
            prior_covar = self.kernel(x_test, x_test)
            post_covar = (
                prior_covar
                - kernel_train_test.transpose(-1, -2) @ cit_train
                - kernel_peek_test.transpose(-1, -2) @ cit_peek
            )
            if noise_level is not None:
                post_covar = post_covar + torch.diag_embed(torch.tensor((noise_level,) * len(x_test)))

        if y_peek is not None:
            c_train = torch.linalg.solve_triangular(self.cholesky, self.y_train[:, None], upper=False)

            c_peek = torch.linalg.solve_triangular(L22, y_peek[:, None] - L21 @ c_train, upper=False)
            ciy_peek = torch.linalg.solve_triangular(L22.transpose(-1, -2), c_peek, upper=True)

            ciy_train = torch.linalg.solve_triangular(
                self.cholesky.transpose(-1, -2),
                c_train - L21.transpose(-1, -2) @ ciy_peek,
                upper=True
            )
            post_mean = (
                kernel_train_test.transpose(-1, -2) @ ciy_train
                + kernel_peek_test.transpose(-1, -2) @ ciy_peek
            ).squeeze(-1)
        else:
            post_mean = torch.zeros(
                (*post_covar.shape[:-2], post_covar.shape[-1]),
                dtype=post_covar.dtype,
                device=post_covar.device
            )

        return MultivariateNormal(post_mean, post_covar)

    def prior(self):
        '''Return the prior as a multivariate normal distribution.

        Returns
        -------
        distribution : obj:`MultivariateNormal`
            The prior as a multivariate normal distribution.
        '''
        return MultivariateNormal(self.mean, self.covar)

    def update(self, x_cand, y_cand):
        """ Update the training points with new candidate point(s).

        Parameters
        ----------
        x_cand : obj:`numpy.ndarray`
            New (batch of) candidate points to be added.
        y_cand : obj:`numpy.ndarray`, optional
            Function values of candidate points to be added.
        """
        dim = len(self.covar)
        dnew = len(x_cand)
        kernel_both = self.kernel(self.x_train, x_cand)
        kernel_cand = self.kernel(x_cand, x_cand)
        # kernel_cand.diagonal()[()] += self.reg
        kernel_cand = kernel_cand + torch.diag_embed(torch.tensor((self.reg,) * dnew))

        covar = torch.empty((dim + dnew,) * 2, dtype=self.covar.dtype)
        covar[:dim, :dim] = self.covar
        covar[:dim, dim:] = kernel_both
        covar[dim:, :dim] = kernel_both.t()
        covar[dim:, dim:] = kernel_cand

        L21, L22 = self.cholesky_append(kernel_both, kernel_cand)

        chol = torch.empty((dim + dnew,) * 2, dtype=self.cholesky.dtype)
        chol[:dim, :dim] = self.cholesky
        chol[:dim, dim:] = 0.
        chol[dim:, :dim] = L21
        chol[dim:, dim:] = L22

        y_train = torch.cat((self.y_train, y_cand), dim=0)
        cov_inv_y = torch.cholesky_solve(y_train[:, None], chol, upper=False)

        self.x_train = torch.cat((self.x_train, x_cand), dim=0)
        self.y_train = y_train

        self.mean = torch.cat(
            (self.mean, torch.full((dnew,), self._mean, dtype=self.mean.dtype)), dim=0
        )

        self.covar = covar
        self.cholesky = chol
        self.cov_inv_y = cov_inv_y

    def replace_last(self, x_cand, y_cand=None, dirty=False):
        """Replace the most recent training points with new candidate point(s).

        Parameters
        ----------
        x_cand : obj:`numpy.ndarray`
            New (batch of) candidate points to be replaced.
        y_cand : obj:`numpy.ndarray`, optional
            Function values of candidate points to be replaced.
        """
        dnew = len(x_cand)
        dim = len(self.covar) - dnew

        kernel_both = self.kernel(self.x_train[:dim], x_cand)
        kernel_cand = self.kernel(x_cand, x_cand)
        kernel_cand = kernel_cand + torch.diag_embed(torch.tensor((self.reg,) * dnew))

        self.covar[:dim, dim:] = kernel_both
        self.covar[dim:, :dim] = kernel_both.t()
        self.covar[dim:, dim:] = kernel_cand

        L21 = torch.linalg.solve_triangular(self.cholesky, kernel_both, upper=False).t()
        pred_cov = kernel_cand - L21 @ L21.t()
        if (torch.linalg.matrix_rank(pred_cov, atol=1e-10) < pred_cov.shape[-1]).any():
            raise SingularGramError('Updated Gram matrix is singular!')
        try:
            L22 = torch.linalg.cholesky(pred_cov, upper=False)
        except RuntimeError as error:
            raise SingularGramError('Updated Gram matrix is singular!') from error

        self.cholesky[:dim, dim:] = 0.
        self.cholesky[dim:, :dim] = L21
        self.cholesky[dim:, dim:] = L22

        if y_cand is not None:
            self.y_train[dim:] = y_cand

        if not dirty:
            cov_inv_y = torch.cholesky_solve(self.y_train[:, None], self.cholesky, upper=False)
            self.cov_inv_y = cov_inv_y

        self.x_train[dim:] = x_cand

    def remove_last(self, size):
        """Remove the most recent training points."""
        self.x_train = self.x_train[:-size]
        self.y_train = self.y_train[:-size]
        self.mean = self.mean[:-size]
        self.covar = self.covar[:-size, :-size]
        self.cholesky = self.cholesky[:-size, :-size]
        self.cov_inv_y = torch.cholesky_solve(self.y_train[:, None], self.cholesky, upper=False)

    def reinit(self):
        self.initialize(self.x_train, self.y_train, self.kernel, reg=self.reg, mean=self.mean)

    def log_likelihood(self):
        term_1 = 0.5 * self.y_train.t() @ self.cov_inv_y
        term_2 = self.cholesky.diagonal().log().sum()
        term_3 = 0.5 * len(self.y_train) * math.log(2 * math.pi)
        return -(term_1 + term_2 + term_3)

    def grad_log_likelihood(self, kernel_grad):
        term_1 = self.cov_inv_y.t() @ self.cov_inv_y @ kernel_grad
        term_2 = torch.cholesky_solve(kernel_grad, self.cholesky, upper=False)
        return 0.5 * (term_1.trace() - term_2.trace())

    def loocv_mll_closed(self):
        kernel_inv_diag = torch.diag(torch.cholesky_inverse(self.cholesky, upper=False))
        loo_mu = self.y_train - self.cov_inv_y.squeeze(1) / kernel_inv_diag
        loo_var = 1 / kernel_inv_diag
        # log2pi = 1.8378770664093453
        return (-0.5 * (loo_var.log() + (self.y_train - loo_mu) ** 2 / loo_var)).sum()

    def loocv_mll(self):
        loo = LeaveOneOut()
        result = 0
        loo.get_n_splits(self.x_train)

        for train_index, val_index in loo.split(self.x_train):
            x_train, x_val = self.x_train[train_index], self.x_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            sigma = torch.diag_embed(torch.tensor((self.reg,) * len(x_train)))
            kernel_inv = torch.linalg.inv(self.kernel(x_train, x_train) + sigma)
            kernel_inv_y = torch.matmul(kernel_inv, y_train)
            test_kernel = self.kernel(x_train, x_val)
            pred = torch.dot(test_kernel.squeeze(1), kernel_inv_y)

            result += ((y_val - pred) ** 2).sum()
        return result

    def cholesky_append(self, kernel_peek, peek_covar):
        L21 = torch.linalg.solve_triangular(self.cholesky, kernel_peek, upper=False).transpose(-2, -1)
        pred_cov = peek_covar - L21 @ L21.transpose(-2, -1)
        if (torch.linalg.matrix_rank(pred_cov, atol=1e-10) < pred_cov.shape[-1]).any():
            raise SingularGramError('Updated Gram matrix is singular!')
        try:
            L22 = torch.linalg.cholesky(pred_cov, upper=False)
        except RuntimeError as error:
            raise SingularGramError('Updated Gram matrix is singular!') from error

        return L21, L22
