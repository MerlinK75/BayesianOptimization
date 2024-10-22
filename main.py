from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement

def sphere(x, y):
    """sphere function with n amount of inputs
    """

    return -(x**2 + y**2)

if __name__=='__main__':
    # Bounded region of parameter space
    pbounds = {'x': (-3, 3), 'y': (-3, 3)}

    optimizer = BayesianOptimization(
        f=sphere,
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        population=False)
    
    optimizer.maximize(
    init_points=0,
    n_iter=25)

    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))
    #     #print(f"GP: {optimizer._gp.get_params()}")k