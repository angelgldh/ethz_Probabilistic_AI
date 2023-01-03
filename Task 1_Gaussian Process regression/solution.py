import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary

    def determine_regions(self,train_x, train_y):
        """
        This function is used to determine the regions to which every point belongs
        """

        regions_vec = np.array(range(train_x.shape[0])) # this vector will store the region each point belongs to

        regions_vec[(train_x[:,0] <= 0.25) & (train_x[:,1] <= 0.25)] = 1
        regions_vec[((train_x[:,0] > 0.25 ) & (train_x[:,0] <= 0.5)) & (train_x[:,1] <= 0.25)]  = 2
        regions_vec[((train_x[:,0] > 0.5 ) & (train_x[:,0] <= 0.75)) & (train_x[:,1] <= 0.25)]  = 3
        regions_vec[(train_x[:,0] > 0.75 )  & (train_x[:,1] <= 0.25)]  = 4

        regions_vec[(train_x[:,0] <= 0.25) & ((train_x[:,1] > 0.25) & (train_x[:,1] <= 0.5) )] = 5
        regions_vec[((train_x[:,0] > 0.25 ) & (train_x[:,0] <= 0.5)) & ((train_x[:,1] > 0.25) & (train_x[:,1] <= 0.5) )] = 6
        regions_vec[((train_x[:,0] > 0.5 ) & (train_x[:,0] <= 0.75)) & ((train_x[:,1] > 0.25) & (train_x[:,1] <= 0.5) )] = 7
        regions_vec[(train_x[:,0] > 0.75 ) & ((train_x[:,1] > 0.25) & (train_x[:,1] <= 0.5) )] = 8

        regions_vec[(train_x[:,0] <= 0.25) & ((train_x[:,1] > 0.5) & (train_x[:,1] <= 0.75) )] = 9
        regions_vec[((train_x[:,0] > 0.25 ) & (train_x[:,0] <= 0.5)) & ((train_x[:,1] > 0.5) & (train_x[:,1] <= 0.75) )] = 10
        regions_vec[((train_x[:,0] > 0.5 ) & (train_x[:,0] <= 0.75)) & ((train_x[:,1] > 0.5) & (train_x[:,1] <= 0.75) )] = 11
        regions_vec[(train_x[:,0] > 0.75 ) & ((train_x[:,1] > 0.5) & (train_x[:,1] <= 0.75) )] = 12

        regions_vec[(train_x[:,0] <= 0.25) & (train_x[:,1] > 0.75)] = 13
        regions_vec[((train_x[:,0] > 0.25 ) & (train_x[:,0] <= 0.5)) & (train_x[:,1] > 0.75)] = 14
        regions_vec[((train_x[:,0] > 0.5 ) & (train_x[:,0] <= 0.75)) & (train_x[:,1] > 0.75)] = 15
        regions_vec[(train_x[:,0] > 0.75 ) & (train_x[:,1] > 0.75)] = 16

        
        train_x = np.insert(train_x,2, np.array( regions_vec ), axis = 1)

        train_y = train_y[:, np.newaxis] 
        train_y = np.insert(train_y,1, np.array( regions_vec ), axis = 1)

        return train_x, train_y
    
    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean = np.zeros(test_features.shape[0], dtype=float)
        gp_std = np.zeros(test_features.shape[0], dtype=float)

        # TODO: Use the GP posterior to form your predictions here
        predictions = gp_mean

        y_test = predictions
        X_test, y_test = self.determine_regions(test_features, y_test)

        # pred_hat = np.zeros(y_test.shape[0])
        
        for ii in range(1,17):
          # print(gprcs[0][ii - 1])
          gp_pred = self.model[0][ii - 1]
          
          predictions[ y_test[:,1] == ii ] = gp_pred.predict(X = X_test[ X_test[:,2] == ii][:,[0,1]] )



        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray,train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # We split the training data according to their region
        X_train, y_train = self.determine_regions(train_features, train_GT) 

        # Now, we build the kernel regressors, 1 per region
        k= RationalQuadratic(length_scale=1.0, alpha=1.5)
        gprcs = [ [GaussianProcessRegressor(kernel = k), GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k), 
                   GaussianProcessRegressor(kernel = k), GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k), 
                   GaussianProcessRegressor(kernel = k), GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k), 
                   GaussianProcessRegressor(kernel = k), GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k) , GaussianProcessRegressor(kernel = k)], 
                 np.array(range(1,17))] 

        for i in range(1,17):
            print("Fitting region %d" % i)
            gp_aux = GaussianProcessRegressor(kernel=k, random_state=42)
            gp_aux.fit(X =  X_train[X_train[:,2] == i][:,[0,1]], y = y_train[y_train[:,1] == i][:,0])
            gprcs[0][i - 1] = gp_aux

        
        self.model = gprcs

        # TODO: Fit your model here
        pass


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)



def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT,train_features)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_features)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
