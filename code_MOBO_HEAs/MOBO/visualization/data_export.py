import os
import pandas as pd
import numpy as np
#from mobo.utils import find_pareto_front, calc_hypervolume



from autooed.utils.pareto import find_pareto_front, calc_hypervolume

#from utils import get_result_dir
from .utils_DGEMO import get_result_dir


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from pymoo.factory import get_performance_indicator
from autooed.utils.pareto import convert_minimization


def parallel_transform(Y):
    '''
    Transform performance values from cartesian to parallel coordinates
    '''
    Y = np.array(Y)
    return np.dstack([np.vstack([np.arange(Y.shape[1])] * len(Y)), Y])


'''
Export csv files for external visualization.
'''

class DataExport:

    def __init__(self, optimizer, X, Y, args):
        '''
        Initialize data exporter from initial data (X, Y).
        '''
#        print(optimizer.__dict__)
        self.optimizer = optimizer
#        self.problem = optimizer.real_problem
        self.problem = optimizer
        self.n_var, self.n_obj = self.problem.n_var, self.problem.n_obj
        self.batch_size = args.batch_size
#        self.batch_size = self.optimizer.selection.batch_size
        self.iter = 0
        self.transformation = optimizer.transformation

        # saving path related
        self.result_dir = get_result_dir(args)
        
        n_samples = X.shape[0]

        # compute initial hypervolume
        pfront, pidx = find_pareto_front(Y, return_index=True)
        pset = X[pidx]
        if args.ref_point is None:
            args.ref_point = np.max(Y, axis=0)
        hv_value = calc_hypervolume(pfront, ref_point=args.ref_point)
        
        # init data frame
        column_names = ['iterID']
        d1 = {'iterID': np.zeros(n_samples, dtype=int)}
        d2 = {'iterID': np.zeros(len(pset), dtype=int)}

        # design variables
        for i in range(self.n_var):
            var_name = f'x{i + 1}'
            d1[var_name] = X[:, i]
            d2[var_name] = pset[:, i]
            column_names.append(var_name)

        # performance
        for i in range(self.n_obj):
            obj_name = f'f{i + 1}'
            d1[obj_name] = Y[:, i]
            obj_name = f'Pareto_f{i + 1}'
            d2[obj_name] = pfront[:, i]

        # predicted performance
        for i in range(self.n_obj):
            obj_pred_name = f'Expected_f{i + 1}'
            d1[obj_pred_name] = np.zeros(n_samples)
            obj_pred_name = f'Uncertainty_f{i + 1}'
            d1[obj_pred_name] = np.zeros(n_samples)
            obj_pred_name = f'Acquisition_f{i + 1}'
            d1[obj_pred_name] = np.zeros(n_samples)

        d1['Hypervolume_indicator'] = np.full(n_samples, hv_value)

        self.export_data = pd.DataFrame(data=d1) # export all data
        self.export_pareto = pd.DataFrame(data=d2) # export pareto data
        column_names.append('ParetoFamily')
        self.export_approx_pareto = pd.DataFrame(columns=column_names) # export pareto approximation data

        self.has_family = hasattr(self.optimizer.selection, 'has_family') and self.optimizer.selection.has_family

#WX status for exporting data
        self.status = {
            'pset': None,
            'pfront': None,
            'hv': None,
            'ref_point': args.ref_point,
        }

#WX update when first initialization
        self._update_status(X, Y)
       
    def _update_status(self, X, Y):
        '''
        Update the status of algorithm from data
        '''
        if self.iter == 0:
            self.X = X
            self.Y = Y
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])

        self.status['pfront'], pfront_idx = find_pareto_front(self.Y, return_index=True)
        self.status['pset'] = self.X[pfront_idx]
        #self.status['hv'] = calc_hypervolume(self.status['pfront'], self.ref_point)
        self.status['hv'] = calc_hypervolume(self.status['pfront'], self.status['ref_point'])

    def update(self, X_next, Y_next):
        '''
        For each algorithm iteration adds data for visualization.
        Input:
            X_next: proposed sample values in design space
            Y_next: proposed sample values in performance space
        '''
        self.iter += 1

        # evaluate prediction of X_next on surrogate model
        val = self.optimizer.surrogate_model.evaluate(self.transformation.do(X_next), std=True)
        Y_next_pred_mean = self.transformation.undo(val['F'])
        Y_next_pred_std = val['S']
#WX update status when 
        self._update_status(X_next, Y_next)
#        import pdb; pdb.set_trace()

#WX evaluate for identitiy acq. only, you may change this for other acq. functions
        acquisition, _, _ = val['F'], val['dF'], val['hF']
        #acquisition, _, _ = self.optimizer.acquisition.evaluate(val)
 
        pset = self.status['pset']
        pfront = self.status['pfront']
        hv_value = self.status['hv']

        d1 = {'iterID': np.full(self.batch_size, self.iter, dtype=int)} # export all data
        d2 = {'iterID': np.full(pfront.shape[0], self.iter, dtype=int)} # export pareto data

        # design variables
        for i in range(self.n_var):
            var_name = f'x{i + 1}'
            d1[var_name] = X_next[:, i]
            d2[var_name] = pset[:, i]

        # performance and predicted performance
        for i in range(self.n_obj):
            col_name = f'f{i + 1}'
            d1[col_name] = Y_next[:, i]
            d2['Pareto_'+col_name] = pfront[:, i]

            col_name = f'Expected_f{i + 1}'
            d1[col_name] = Y_next_pred_mean[:, i]
            col_name = f'Uncertainty_f{i + 1}'
            d1[col_name] = Y_next_pred_std[:, i]
            col_name = f'Acquisition_f{i + 1}'
            d1[col_name] = acquisition[:, i]

        d1['Hypervolume_indicator'] = np.full(self.batch_size, hv_value)

       #WX
       # if self.has_family:
       #     info = self.optimizer.info
       #     family_lbls, approx_pset, approx_pfront = info['family_lbls'], info['approx_pset'], info['approx_pfront']
       #     approx_front_samples = approx_pfront.shape[0]
       #     
       #     d3 = {'iterID': np.full(approx_front_samples, self.iter, dtype=int)} # export pareto approximation data

       #     for i in range(self.n_var):
       #         var_name = f'x{i + 1}'
       #         d3[var_name] = approx_pset[:, i]

       #     for i in range(self.n_obj):
       #         d3[f'Pareto_f{i + 1}'] = approx_pfront[:, i]

       #     d3['ParetoFamily'] = family_lbls
       # 
       # else:
       #     import pdb; pdb.set_trace()
       #     approx_pset = self.optimizer.solver.solution['x']
       #     val = self.optimizer.surrogate_model.evaluate(approx_pset)
       #     approx_pfront = val['F']
       #     approx_pset, approx_pfront = self.transformation.undo(approx_pset, approx_pfront)

       #     # find undominated
       #     approx_pfront, pidx = find_pareto_front(approx_pfront, return_index=True)
       #     approx_pset = approx_pset[pidx]
       #     approx_front_samples = approx_pfront.shape[0]

       #     d3 = {'iterID': np.full(approx_front_samples, self.iter, dtype=int)}

       #     for i in range(self.n_var):
       #         var_name = f'x{i + 1}'
       #         d3[var_name] = approx_pset[:, i]

       #     for i in range(self.n_obj):
       #         d3[f'Pareto_f{i + 1}'] = approx_pfront[:, i]

       #     d3['ParetoFamily'] = np.zeros(approx_front_samples)

        df1 = pd.DataFrame(data=d1)
        df2 = pd.DataFrame(data=d2)
       # df3 = pd.DataFrame(data=d3)

        self.export_data = pd.concat([self.export_data, df1], ignore_index=True) 
        self.export_pareto = pd.concat([self.export_pareto, df2], ignore_index=True)
       #wx ignore d3 dataframe
       # self.export_approx_pareto = pd.concat([self.export_approx_pareto, df3],ignore_index=True)

       #wx
       # self.export_data = self.export_data.append(df1, ignore_index=True)
       # self.export_pareto = self.export_pareto.append(df2, ignore_index=True)
       # self.export_approx_pareto = self.export_approx_pareto.append(df3, ignore_index=True)

    def write_csvs(self):
        '''
        Export data to csv files.
        '''
        dataframes = [self.export_data, self.export_pareto, self.export_approx_pareto]
        filenames = ['EvaluatedSamples', 'ParetoFrontEvaluated','ParetoFrontApproximation']

        for dataframe, filename in zip(dataframes, filenames):
            filepath = os.path.join(self.result_dir, filename + '.csv')
            dataframe.to_csv(filepath, index=False)

    def write_truefront_csv(self, truefront):
        '''
        Export true pareto front to csv files.
        '''
        problem_dir = os.path.join(self.result_dir, '..', '..') # result/problem/subfolder/
        filepath = os.path.join(problem_dir, 'TrueParetoFront.csv')

        if os.path.exists(filepath): return

        d = {}
        for i in range(truefront.shape[1]):
            col_name = f'f{i + 1}'
            d[col_name] = truefront[:, i]

        export_tf = pd.DataFrame(data=d)
        export_tf.to_csv(filepath, index=False)


    def plot_performance_space(self, Y):
        '''
        '''
        Y = np.array(Y)
        assert Y.ndim == 2, f'Invalid shape {Y.shape} of objectives to plot'
        if Y.shape[1] == 1:
            plt.scatter(Y, [0] * len(Y), marker='x')
        elif Y.shape[1] == 2:
            plt.scatter(*Y.T)
        elif Y.shape[1] == 3:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(*Y.T)
        elif Y.shape[1] > 3:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            segments = parallel_transform(Y)
            ax.add_collection(LineCollection(segments))
            ax.set_xlim(0, Y.shape[1] - 1)
            ax.set_ylim(np.min(Y), np.max(Y))
        else:
            raise Exception(f'Objectives with dimension {Y.shape[1]} is not supported')
        #import pdb; pdb.set_trace()
        xlocs, xlabels = plt.xticks()
        plt.xticks(ticks=xlocs, labels=[str(-int(i)) for i in xlocs])
        #plt.xticks(-xlables)
        plt.ylim([-200, 0])
        plt.xlim([-350, 0])
        plt.xlabel('Activity rel. to Pt(111) (%)')
        plt.ylabel('Price rel. to Pt (%)')
        plt.title('Performance Space')

        filename = 'performance_space'
        filepath = os.path.join(self.result_dir, filename + '.png')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        #plt.show()
    
    
    
    

    def plot_performance_metric(self, Y, obj_type):

        '''
        '''
        if Y.shape[1] == 1:
            opt_list = []
            if obj_type == ['min']:
                opt_func = np.min
            elif obj_type == ['max']:
                opt_func == np.max
            else:
                raise Exception(f'Invalid objective type {obj_type}')
            for i in range(1, len(Y)):
                opt_list.append(opt_func(Y[:i]))
            plt.plot(np.arange(1, len(Y)), opt_list)
            plt.title('Optimum')
        elif Y.shape[1] > 1:
            Y = convert_minimization(Y, obj_type)
            ref_point = np.max(Y, axis=0)
            #WX using same ref_point
            if len(ref_point) == 2:
               ref_point = [0, 300]; print("Plotting ref_point", ref_point)
            if len(ref_point) == 3:
               ref_point = [0, 300, 0]; print("Plotting ref_point", ref_point)
            #ref_point = np.max(Y, axis=0)
            indicator = get_performance_indicator('hv', ref_point=ref_point)
            hv_list = []
            for i in range(1, len(Y)):
                hv = indicator.calc(Y[:i])
                hv_list.append(hv)
            plt.plot(np.arange(1, len(Y)), hv_list)
            plt.title('Hypervolume')
        else:
            raise Exception(f'Invalid objective dimension {Y.shape[1]}')
        plt.xlabel('Nums. Evaluations')
        plt.ylabel('Hypervolume')

        filename = 'performance_metric'
        filepath = os.path.join(self.result_dir, filename + '.png')

        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        #plt.show()
        return max(hv_list)
    
