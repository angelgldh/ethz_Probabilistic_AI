import numpy as np
from scipy.optimize import fmin_l_bfgs_b


### To add in the requirements!
import os
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k
from math import *
from scipy.stats import norm

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        
        # We need to intialize the GPs for both f and v
        var_f = 0.5
        noise_f = 0.15
        self.f_model = GaussianProcessRegressor( 
            kernel = var_f * k.Matern(length_scale = 0.5, length_scale_bounds = "fixed", nu = 2.5),
            optimizer = None,
            normalize_y = True,
            alpha = noise_f
            )
        var_v = sqrt(2)
        noise_v = 0.0001
        self.v_model = GaussianProcessRegressor( 
            kernel = k.ConstantKernel(1.5) + (var_v * k.Matern(length_scale = 0.5, length_scale_bounds = "fixed", nu = 2.5)),
            optimizer = None,
            normalize_y = True,
            alpha = noise_v
        )
        
        # var_v = sqrt(2)
        # noise_v = 0.0001
        # self.v_model = GaussianProcessRegressor( 
        #     kernel =  (var_v * k.Matern(length_scale = 0.5, length_scale_bounds = "fixed", nu = 2.5)),
        #     optimizer = None,
        #     normalize_y = True,
        #     alpha = noise_v
        # )
        
        # The array storing points is initalized in add_data_points
        # self.set_of_points = np.array([])
        self.set_of_points = []



    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        next_x = self.optimize_acquisition_function()

        return next_x



    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        # for _ in range(20):
        for _ in range(30):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        thr_v = SAFETY_THRESHOLD
        mu_v, std_v = self.v_model.predict(x.reshape(1,-1), return_std = True)
        # pi_v = norm.cdf( (mu_v - thr_v) / std_v )
        pi_v = float(norm.cdf( (mu_v - thr_v) / std_v ))

        # 2nd. evaluate the EXPECTED improvement in f          

        # Let's try to define f_star optimally
        x_grid = (np.arange(domain[0,0], domain[0,1], 0.1)).reshape(-1,1)
        f_star = np.max(self.f_model.predict(x_grid))

        mu_f, std_f = self.f_model.predict(x.reshape(1,-1), return_std = True)
        Z = ( (mu_f - f_star) / std_f )
        ei_f = float(std_f*(Z*norm.cdf(Z) + norm.pdf(Z)))

        af_value = float(ei_f * pi_v)
        return af_value



    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.set_of_points.append([float(x), float(f), float(v)])

        aux = np.array(self.set_of_points)

        x_vals = aux[:,0].reshape(-1,1)
        f_vals = aux[:,1].reshape(-1,1)
        v_vals = aux[:,2].reshape(-1,1)
        # Every time we add a point, we refit the model
        self.f_model.fit(x_vals, f_vals)
        self.v_model.fit(x_vals, v_vals)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        
        aux = np.array(self.set_of_points)

        x_vals = aux[:,0].reshape(-1,1)
        f_vals = aux[:,1].reshape(-1,1)
        v_vals = aux[:,2].reshape(-1,1)

        if (v_vals >= SAFETY_THRESHOLD).any():
          f_filtered = f_vals[v_vals >= SAFETY_THRESHOLD]
          x_filtered = x_vals[v_vals >= SAFETY_THRESHOLD]
          ind = np.argmax(f_filtered)

          x_opt = x_filtered[ind]
        else:
          x_opt = get_initial_safe_point()

        
        return x_opt


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()