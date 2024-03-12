import numpy as np

from autooed.problem import build_problem
from autooed.mobo import build_algorithm
from autooed.utils.seed import set_seed
from autooed.utils.initialization import generate_random_initial_samples
#from autooed.utils.plot import plot_performance_space, plot_performance_metric

from arguments import get_args

import os
os.environ['OMP_NUM_THREADS'] = '1' # speed up #WX
from visualization.data_export import DataExport #WX
from autooed.mobo.algorithms import get_algorithm

if __name__ == '__main__':
    #import pdb; pdb.set_trace()

    # load arguments
    args, module_cfg = get_args()
     
    # set random seed
    set_seed(args.seed)
    
   # import pdb; pdb.set_trace()
    # build problem
    problem = build_problem(args.problem)
    print(problem)

    # build algorithm
    algorithm = build_algorithm(args.algo, problem, module_cfg)
    print(algorithm)
    #import pdb; pdb.set_trace()

    # generate initial random samples
    X = generate_random_initial_samples(problem, args.n_init_sample)
    Y = np.array([problem.evaluate_objective(x) for x in X])

    #WX: initialize data explorter
    exporter = DataExport(algorithm, X, Y, args)

    # optimization
    while len(X) < args.n_total_sample:

    #    import pdb; pdb.set_trace()
        # propose design samples
        X_next = algorithm.optimize(X, Y, None, args.batch_size)

        # evaluate proposed samples
        Y_next = np.array([problem.evaluate_objective(x) for x in X_next])
        
        #WX update & export current status to csv
        exporter.update(X_next, Y_next)
        exporter.write_csvs()

        # combine into dataset
        X = np.vstack([X, X_next])
        Y = np.vstack([Y, Y_next])

        print(f'{len(X)}/{args.n_total_sample} complete')

    # plot
    hv_max = exporter.plot_performance_metric(Y, problem.obj_type)
    exporter.plot_performance_space(Y)
    #print("min activity set:", Y[np.argmin(Y[:,0])], "max hv", hv_max) 
    #plot_performance_space(Y)
    #plot_performance_metric(Y, problem.obj_type)
