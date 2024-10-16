from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

if __name__=='__main__':
    # Bounded region of parameter space
    pbounds = {'x': (2, 4), 'y': (-3, 3)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        population=False)
    
    optimizer.maximize(
    init_points=10,
    n_iter=15)

    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))
    #     #print(f"GP: {optimizer._gp.get_params()}")k