from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement
import numpy as np

def sphere(x, y):
    """sphere function with n amount of inputs
    """

    return -(x**2 + y**2)

def branin(x,y):
    """Branin function with 2 inputs"""
    return -((y-((5.1/(4*np.pi**2))*x**2)+((5/np.pi)*x)-6)**2+10*(1-(1/(8*np.pi)))*np.cos(x)+10)

def hartmann_3(x,y,z):
    """Hartmann 3 function with 3 inputs"""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = np.array([[0.3689, 0.1170, 0.2673],
                  [0.4699, 0.4387, 0.7470],
                  [0.1091, 0.8732, 0.5547],
                  [0.03815, 0.5743, 0.8828]])
    outer = 0
    for i in range(4):
        inner = 0
        for j in range(3):
            inner += A[i,j]*(x[j]-P[i,j])**2
        outer += alpha[i]*np.exp(-inner)
    return -outer
                  


if __name__=='__main__':
    # Bounded region of parameter space
    pbounds = {'x': (-3, 3), 'y': (-3, 3)}

    weights = [1.0, 0.0]

    optimizer = BayesianOptimization(
        f=[sphere, branin],
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(weights=weights, xi=0.01),
        population=True)
    
    optimizer.maximize(
    init_points=10,
    n_iter=15)

    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))
    #     #print(f"GP: {optimizer._gp.get_params()}")k