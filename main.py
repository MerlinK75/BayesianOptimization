from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement

def sphere(x):
    """1D sphere function with slope and center variable
    """
    return -(x-12)**2 *1



if __name__=='__main__':
    # Bounded region of parameter space
    pbounds = {'x': (4, 40)}

    optimizer = BayesianOptimization(
        f=sphere,
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        population=False,
        )
    
    optimizer.maximize(
    init_points=0,
    n_iter=25)

    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))
    #     #print(f"GP: {optimizer._gp.get_params()}")k