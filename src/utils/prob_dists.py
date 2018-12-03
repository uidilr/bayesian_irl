import scipy.stats
import numpy as np


class UniformDist:
    def __init__(self, xmax=1., xmin=None):
        self.xmax = xmax
        self.xmin = - xmax if xmin is None else xmin
        self.prob = 1 / (self.xmax - self.xmin)

    def __call__(self, *args, **kwargs):
        return self.prob

    def __str__(self):
        return 'UniformDist(max={}, min={})'.format(self.xmax, self.xmin)


class DistBase:
    def __init__(self, dist, params):
        self.dist = dist
        self.params = params

    def __call__(self, x):
        """
        :x: input
        :return: P(x)
        """
        return np.exp(np.sum(self.dist.logpdf(x, **self.params)))

    def sample(self, size=10):
        return self.dist.rvs(size=size, **self.params)

    def __str__(self):
        return self.__class__.__name__ + '(' + ', '.join(['{}={}'.format(key, value)
                                                        for key, value in self.params.items()]) + ')'


class GaussianDist(DistBase):
    def __init__(self, loc=0, scale=0.1):
        """
        :param loc: location of gaussian distribution
        :param scale: var == scale ** 2
        """
        params = dict(loc=loc, scale=scale)
        dist = scipy.stats.norm
        super().__init__(dist=dist, params=params)



class BetaDist(DistBase):
    def __init__(self, a=0.5, b=0.5, loc=0, scale=1):
        params = dict(a=a, b=b, loc=loc, scale=scale)
        dist = scipy.stats.beta
        super().__init__(dist=dist, params=params)


class GammaDist(DistBase):
    def __init__(self, a=2, loc=0, scale=1):
        params = dict(a=a, loc=loc, scale=scale)
        dist = scipy.stats.gamma
        super().__init__(dist=dist, params=params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    dists = (GaussianDist, BetaDist, GammaDist)
    for dist in dists:
        distribution = dist()
        samples = distribution.sample(size=100)
        plt.hist(samples)
        plt.title(distribution)
        path = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-3], 'results',
                                  '{}.png'.format(dist.__name__))
        plt.savefig(path)
        plt.cla()
