<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Run Locally](#run-locally)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
  * [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  

<!-- About the Project -->
## About the Project

<div align="center"> 
  <img src="https://placehold.co/600x400?text=Your+Screenshot+here" alt="screenshot" />
</div>

This Project contains 4 tasks carried out as part of the [Probabilistic Artificial Intelligence (PAI)](https://las.inf.ethz.ch/teaching/pai-f22) subject at ETH-Zurich, Autumn 2022 semester.

### Task 1: Gaussian Process regression to predict air pollution

<div align="center"> 
<![image](https://user-images.githubusercontent.com/102548683/211349765-1ef7a120-0bb2-4546-a9b3-37dc6f321383.png)>
</div>
Image source: Koussoulakou, Alexandra & Soulakellis, Nikolaos & Sarafidis, Dimitrios. (2018). INTERACTIVE VISUALIZATIONS OF SPATIAL AND TEMPORAL AIR POLLUTION ASPECTS FOR MONITORING AND CONTROL. 

#### Goal

In this task, we aim to help a city predict and audit the concentration of fine particulate matter (PM2.5) per cubic meter of air. In an initial phase, the city has collected preliminary measurements using mobile measurement stations. The goal is now to develop a pollution model that can predict the air pollution concentration in locations without measurements. This model will then be used to determine particularly polluted areas where permanent measurement stations should be deployed.

A pervasive class of models for weather and meteorology data are Gaussian Processes (GPs). We are then implementing  Gaussian Process regression in order to model air pollution and try to predict the concentration of PM2.5 at previously unmeasured locations.

(For a gentle introduction to Gaussian processes: [Intro to Gaussian process regression](https://medium.com/data-science-at-microsoft/introduction-to-gaussian-process-regression-part-1-the-basics-3cb79d9f155f#:~:text=Gaussian%20process%20(GP)%20is%20a,generalization%20of%20multivariate%20Gaussian%20distributions.))

#### Problem set-up and challenges

As features, we are given the coordinates (X,Y) of the city map. As target, we need to predict the pollution particles concentration at a given location.

The problem presents various challenges:
- Model selection. 

#### Approach and results



![image](https://user-images.githubusercontent.com/102548683/211350293-8b55d009-fbf6-4bfe-ba32-b23f47931e4c.png)


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


<!-- Usage -->
## Usage

Use this space to tell a little more about your project and how it can be used. Show additional screenshots, code samples, demos or link to other resources.


```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```

<!-- Roadmap -->
## Roadmap

* [x] Todo 1
* [ ] Todo 2

<!-- Contributing -->
## Contributing

<a href="https://github.com/Louis3797/awesome-readme-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Louis3797/awesome-readme-template" />
</a>


Contributions are always welcome!

See `contributing.md` for ways to get started.


<!-- Code of Conduct -->
### Code of Conduct

Please read the [Code of Conduct](https://github.com/Louis3797/awesome-readme-template/blob/master/CODE_OF_CONDUCT.md)


<!-- License -->
## License

Distributed under the no License. See LICENSE.txt for more information.


<!-- Contact -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/Louis3797/awesome-readme-template](https://github.com/Louis3797/awesome-readme-template)

<!-- Acknowledgments -->
## Acknowledgements

Use this section to mention useful resources and libraries that you have used in your projects.

 - [Shields.io](https://shields.io/)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [Emoji Cheat Sheet](https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md#travel--places)
 - [Readme Template](https://github.com/othneildrew/Best-README-Template)
