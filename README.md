<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
  * [Task 1: Gaussian Process regression to predict air pollution](#task-1-Gaussian-Process-regression-to-predict-air-pollution)
  * [Task 2: Bayesian neural Network (BNN) for multi-class classification](#task-2-Bayesian-neural-Network-(BNN)-for-multi-class-classification)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Run Locally](#run-locally)
- [Roadmap](#roadmap)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  

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


<!-- Getting Started -->
## Getting Started

<!-- Prerequisites -->
### Prerequisites

This project uses Yarn as package manager

```bash
 npm install --global yarn
```

<!-- Installation -->
### Installation

Install my-project with npm

```bash
  yarn install my-project
  cd my-project
```


<!-- Run Locally -->
### Run Locally

Clone the project

```bash
  git clone https://github.com/Louis3797/awesome-readme-template.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  yarn install
```

Start the server

```bash
  yarn start
```


<!-- Roadmap -->
## Roadmap

* [x] Todo 1
* [ ] Todo 2

<!-- Contributing -->
## Contributing

Contributions are always welcome!


<!-- Contact -->
## Contact

Angel Garcia Lopez de Haro
[@Linkedin](https://www.linkedin.com/in/angel-garcia-lopez-de-haro/)
ETHZ email: angarcia@ethz.ch
Work email: agarcialopezdeharo@gmail.com


<!-- Acknowledgments -->
## Acknowledgements

