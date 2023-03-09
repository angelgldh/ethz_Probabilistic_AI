<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
  * [Task 1: Gaussian Process regression to predict air pollution](#task-1-Gaussian-Process-regression-to-predict-air-pollution)
  * [Task 2: Bayesian neural Network (BNN) for multi-class classification](#task-2-Bayesian-neural-Network-BNN-for-multi-class-classification)
  * [Task 3: Bayesian optimization](#task-3-Bayesian-optimization)
  * [Task 4: Lunar lander game following RL Actor-critic approach](#task-4-Lunar-lander-game-following-RL-Actor-critic-approach)



<!-- About the Project -->
## About the Project

This Project contains 4 tasks carried out as part of the [Probabilistic Artificial Intelligence (PAI)](https://las.inf.ethz.ch/teaching/pai-f22) subject at ETH-Zurich, Autumn 2022 semester.

### Task 1: Gaussian Process regression to predict air pollution

![Air_pollution_map](https://user-images.githubusercontent.com/102548683/211349765-1ef7a120-0bb2-4546-a9b3-37dc6f321383.png)
 
 
Image source: Koussoulakou, Alexandra & Soulakellis, Nikolaos & Sarafidis, Dimitrios. (2018). INTERACTIVE VISUALIZATIONS OF SPATIAL AND TEMPORAL AIR POLLUTION ASPECTS FOR MONITORING AND CONTROL. 

#### Goal

In this task, we aim to help a city predict and audit the concentration of fine particulate matter (PM2.5) per cubic meter of air. In an initial phase, the city has collected preliminary measurements using mobile measurement stations. The goal is now to develop a pollution model that can predict the air pollution concentration in locations without measurements. This model will then be used to determine particularly polluted areas where permanent measurement stations should be deployed.

A pervasive class of models for weather and meteorology data are Gaussian Processes (GPs). We are then implementing  Gaussian Process regression in order to model air pollution and try to predict the concentration of PM2.5 at previously unmeasured locations.

(For a gentle introduction to GPs: [Intro to Gaussian process regression](https://medium.com/data-science-at-microsoft/introduction-to-gaussian-process-regression-part-1-the-basics-3cb79d9f155f#:~:text=Gaussian%20process%20(GP)%20is%20a,generalization%20of%20multivariate%20Gaussian%20distributions.))

#### Problem set-up and challenges

As features, we are given the coordinates (X,Y) of the city map. As target, we need to predict the pollution particles concentration at a given location.

The problem presents various challenges:
- Model selection. Determination of right kernel and hyperparameters are key for GPs performance.
- Large scale learning. As the number of observations increase, the computational cost of GPs grows exponentially: posterior complexity is of O(n^3). To tackle this problem, there are several approaches (undersampling, Kernel low-rank approximations, Local GPs). In this implementation, we made use of Local GPs, i.e., the fitting of various individual GPs along the map, instead of a global regressor for the whole of the set.
- Asymmetric cost. Cost-sensitive learning is implemented as we implement a loss function that weights different kinds of errors differently: 
![loss_function_task 1](https://user-images.githubusercontent.com/102548683/211354718-021ca464-f29b-4086-a04b-90fe00a274a5.png)


#### Approach and results

The main contribution is the division of the city map into various squares of same size, where an individual GP is fitted. This way, we implement a Local GP approach and make predictions more specific to every location.

This approach achieves a more specific prediction than implementing a global GP and produces a cumulative cost of 48.519.

![task_1_results](https://user-images.githubusercontent.com/102548683/211350293-8b55d009-fbf6-4bfe-ba32-b23f47931e4c.png)


### Task 2: Bayesian neural Network (BNN) for multi-class classification
![image](https://user-images.githubusercontent.com/102548683/211356167-e1271fe4-5923-429d-8e97-cf3fb1effbdb.png)
Source: Ahamed, S. (2019). Estimating uncertainty of earthquake rupture using Bayesian neural network. arXiv preprint arXiv:1911.09660.

#### Goal
Implement a Bayesian Neural Network for multi-class classification using Monte Carlo drop-out.

(For a gentle introduction to Bayesian Neural Networks: [Bayesian NNets @ Towards Data Science](https://towardsdatascience.com/bayesian-neural-network-7041dd09f2cc). For Monte Carlo drop-out: [MC drop-out@ Towards Data Science](https://towardsdatascience.com/monte-carlo-dropout-7fd52f8b6571))


#### Problem set-up and challenges

The training set of this problem is the famous MNIST image data set: 
![image](https://user-images.githubusercontent.com/102548683/211357150-0fd0a3a2-a630-477a-9c66-3f7b23ff2eb6.png)

However, the testing data is constituted by modified versions of the MNIST images:
![image](https://user-images.githubusercontent.com/102548683/211357356-90f39df9-f00f-44d5-9e58-f87b5825c650.png)

Them the main challenge of this task is to build a model robust to modifications in the input data. 

The performance metric governing the problem is the Expected Calibration error (ECE), composed by the accuracy and the empirical confidence of every prediction:
![image](https://user-images.githubusercontent.com/102548683/211357862-8db50b12-48fd-4652-8d27-b2a81717d1eb.png)



#### Approach and results

The implementation of Monte Carlo drop out results in a satisfactory prediction result, yielding an ECE of 0.162, averaged in 200 repeats of the setting. Below the most confident predictions are shown, togehter with the model prediction of more ambigiuos test images. As potential next steps, one could developd a model more robust to this variability in the input.

![image](https://user-images.githubusercontent.com/102548683/224070761-5531d03f-895a-4358-89f0-c9a817bdd3ae.png)
![image](https://user-images.githubusercontent.com/102548683/224070837-3304f315-acfe-42bf-88fe-e7290009e819.png)



### Task 3: Bayesian optimization


#### Goal

In many real-world scenarios, collecting data can be very expensive. In this context, Bayesian optimization can be a good method: Bayesian optimization is characterizing by the sampling of new data points while at the same time it optimizes a given function. In this task, a Bayesian optimization model is implemented.

![image](https://user-images.githubusercontent.com/102548683/224078044-49131cc1-7f7b-4564-bcda-c122e17bb1e5.png)

(For a gentle introduction to Bayesian Optimization: [Exploring Bayesian optimization]([https://towardsdatascience.com/bayesian-neural-network-7041dd09f2cc](https://distill.pub/2020/bayesian-optimization/)). Implementing Bayesian Optimization from scratch in Python: [Bayesian Optimization @ ML Mastery]([https://towardsdatascience.com/monte-carlo-dropout-7fd52f8b6571](https://machinelearningmastery.com/what-is-bayesian-optimization/)))


#### Problem settting and challenges

The task is to use Bayesian optimization to tune one hyperparameter of a machine learning model subject to a constraint on a property of the model. The hyperparameter of interest, denoted as T, is the number of layers in a deep neural network. The goal is to find a network that makes accurate and fast predictions, with the highest possible validation accuracy, such that a requirement on the average prediction speed is satisfied.

The accomplishment of the problem presentes the following main challenges: 

- The objective of this problem does not admit an analytical expression, is computationally expensive to evaluate and is only accessible through noisy evaluations.

- The mapping from the space of hyperparameters to the corresponding validation accuracy and prediction speed can be effectively modeled with a Matérn kernel with variance, length scale, and smootheness parameter.

- The minimum tolerated speed is a constraint on the problem, which may require trying hyperparameters for which the speed constraint is violated.

- The training of the neural network is simulated, and the time required for this step is platform-independent.

- The noise perturbing the observations is Gaussian with standard deviation, and the unit of measurement is not relevant.

- The domain is T = [0, 5], and the hyperparameter may take continuous values.


#### Approach and results

Implementation of both the probability of improvement and the expected improvement leads to satisfactory results. These two approaches can be revisited in the following papers:

[1] [Practical Bayesian Optimization of Machine
Learning Algorithms](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf)
[2] [Bayesian Optimization with Unknown Constraints](https://www.cs.princeton.edu/~rpa/pubs/gelbart2014constraints.pdf)



#### Task 4: Lunar lander game following RL Actor-critic approach

