import os
from argparse import ArgumentParser, Namespace
import yaml
from multiprocessing import cpu_count

'''
Get argument values from command line
Here we speficy different argument parsers to avoid argument conflict between initializing each components
'''

def get_general_args(args=None):
    '''
    General arguments: problem and algorithm description, experiment settings
    '''
    parser = ArgumentParser()

    parser.add_argument('--problem', type=str, default='DTLZ1', 
        help='optimization problem')
    parser.add_argument('--algo', type=str, default='tsemo',
        help='type of algorithm to use with some predefined arguments, or custom arguments')
    parser.add_argument('--seed', type=int, default=0,
        help='the specific seed to run')
    parser.add_argument('--batch-size', type=int, default=10, 
        help='size of the batch in optimization')
    parser.add_argument('--n-init-sample', type=int, default=20, 
        help='number of initial design samples')
    parser.add_argument('--n-total-sample', type=int, default=100, 
        help='number of total design samples (budget)')

#wenbin for output
    parser.add_argument('--ref-point', type=float, nargs='+', default=None,
        help='reference point for calculating hypervolume')
    parser.add_argument('--subfolder', type=str, default='default',
        help='subfolder name for storing results, directly store under result/ as default')
    parser.add_argument('--exp-name', type=str, default=None,
        help='custom experiment name to distinguish between experiments on same problem and same algorithm')
    parser.add_argument('--log-to-file', default=False, action='store_true',
        help='log output to file rather than print by stdout')
    parser.add_argument('--n-process', type=int, default=cpu_count(),
        help='number of processes to be used for parallelization')
#wenbin for runing multi-seed
    parser.add_argument('--n_seed', type=int, default=1,
        help='number of seeds for ensemble')


    args, _ = parser.parse_known_args(args)
    return args


def get_surroagte_args(args=None):
    '''
    Arguments for fitting the surrogate model
    '''
    parser = ArgumentParser()

    parser.add_argument('--surrogate', type=str, 
        choices=['gp', 'nn', 'bnn'], default='gp', 
        help='type of the surrogate model')

    args, _ = parser.parse_known_args(args)
    return args

def get_acquisition_args(args=None):
    '''
    Arguments for acquisition function
    '''
    parser = ArgumentParser()

    parser.add_argument('--acquisition', type=str,  
        choices=['identity', 'pi', 'ei', 'ucb', 'ts'], default='identity', 
        help='type of the acquisition function')

    args, _ = parser.parse_known_args(args)
    return args

def get_solver_args(args=None):
    '''
    Arguments for multi-objective solver
    '''
    parser = ArgumentParser()

    # general solver
    parser.add_argument('--solver', type=str, 
        choices=['nsga2', 'moead', 'parego', 'discovery', 'ga', 'cmaes'], default='nsga2', 
        help='type of the multiobjective solver')
    parser.add_argument('--n-process', type=int, default=cpu_count(),
        help='number of processes to be used for parallelization')

#WX adding for discovery solver
    parser.add_argument('--delta_p', type=int, default=10,
        help='factor of perturbation in stochastic sampling')
    parser.add_argument('--label_cost', type=int, default=10,
        help='reducing number of unique labels in sparse approximation')
    parser.add_argument('--delta_s', type=float, default=0.3,
        help='scaling factor for local minimization')
    parser.add_argument('--delta_b', type=float, default=2,
        help='unary energy normalization constant')
    parser.add_argument('--n_grid_sample', type=int, default=100,
        help='number of samples on local manifold')
    parser.add_argument('--n_cell', type=int, default=100,
        help='number of buffer cells')

    args, _ = parser.parse_known_args(args)
    return args


def get_selection_args(args=None):
    '''
    Arguments for sample selection
    '''
    parser = ArgumentParser()

    parser.add_argument('--selection', type=str,
        choices=['direct', 'hvi', 'random', 'uncertainty'], default='hvi', 
        help='type of selection method for new batch')

    args, _ = parser.parse_known_args(args)
    return args


def get_args():
    '''
    Get arguments from all components
    You can specify args-path argument to directly load arguments from specified yaml file
    '''
    parser = ArgumentParser()
    parser.add_argument('--args-path', type=str, default=None,
        help='used for directly loading arguments from path of argument file')
    args, _ = parser.parse_known_args()

    if args.args_path is None:

        general_args = get_general_args()
        surroagte_args = get_surroagte_args()
        acquisition_args = get_acquisition_args()
        solver_args = get_solver_args()
        selection_args = get_selection_args()

        module_cfg = {
            'surrogate': vars(surroagte_args),
            'acquisition': vars(acquisition_args),
            'solver': vars(solver_args),
            'selection': vars(selection_args),
        }

    else:
        
        with open(args.args_path, 'r') as f:
            all_args = yaml.load(f)
        
        general_args = Namespace(**all_args['general'])
        module_cfg = all_args.copy()
        module_cfg.pop('general')

    return general_args, module_cfg
