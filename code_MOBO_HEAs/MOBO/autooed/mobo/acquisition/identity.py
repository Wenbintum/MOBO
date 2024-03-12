'''
Identity acquisition function.
'''

from autooed.mobo.acquisition.base import Acquisition


class Identity(Acquisition):
    '''
    Identity function.
    '''
    def _fit(self, X, Y):
        pass

    def _evaluate(self, X, gradient, hessian):
        val = self.surrogate_model.evaluate(X, dtype='continuous', std=False, gradient=gradient, hessian=hessian)
        F, dF, hF, y_true = val['F'], val['dF'], val['hF'], val['F']
        return F, dF, hF, y_true
