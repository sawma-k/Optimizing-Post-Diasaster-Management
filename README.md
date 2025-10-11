# Optimizing Post-Disaster Humanitarian Supply Chains Using Machine Learning and Multi-Criterion Decision-Making

This repository contains the code and supporting documents for a research project on designing an efficient post-disaster humanitarian supply chain (HSC). The project integrates Machine Learning (ML), Multi-Criterion Decision-Making (MCDM), and Bi-Objective Optimization to create a robust decision-support framework for resource allocation in disaster scenarios.

## Overview

In the chaotic aftermath of a natural disaster, effective and timely resource allocation is critical to minimizing human suffering. This project proposes a three-phase methodology to address this challenge:
1.  **Prioritizing affected regions** using MCDM techniques to determine which areas require the most urgent attention.
2.  **Classifying injured individuals** by severity using a machine learning model to enable efficient patient triage.
3.  **Optimizing resource allocation** through a bi-objective mathematical model that balances the competing goals of minimizing travel distance (time) and operational costs.

The framework is tested on a synthetic case study based on a hypothetical earthquake in Mizoram, India, a region located in the highest-risk seismic zone (Zone V).

## Methodology

The project's methodology is broken down into three core analytical components:

### 1. Regional Prioritization (Multi-Criterion Decision-Making)

To decide which districts to prioritize for rescue efforts, a combination of the **Best-Worst Method (BWM)** and the **Weighted Aggregated Sum Product Assessment (WASPAS)** was used.

* **BWM** was used to calculate the weights of four key criteria:
    * Population
    * Area Size (Cost Criterion)
    * Hazard Factor
    * Accessibility
* **WASPAS** then used these weights to rank the 11 districts of Mizoram, producing a priority score ($\alpha_m$) for each. This score is a direct input into the optimization model.

### 2. Patient Triage (Machine Learning)

A machine learning model was trained to predict the severity of an injured person's condition, enabling better triage and resource allocation.

* **Dataset**: A synthetic dataset of 1,000 patients was generated with features like Age, Gender, Consciousness Level, Severe Bleeding, Fractures, and Pulse Rate.
* **Models Compared**: Several classification algorithms were evaluated, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), and a Decision Tree.
* **Selected Model**: The **Decision Tree Classifier** was chosen for its high accuracy (95.5%) and F1-score (95.6%), outperforming the other models. This model predicts one of five severity levels (Very Low, Low, Medium, High, Very High) for each patient.

![Model Accuracy Comparison]([https://i.imgur.com/vHqQcSO.png](https://github.com/sawma-k/Optimizing-Post-Diasaster-Management/blob/df7df99340a8fe3ce7f3b01685e33bd1d475b879/Images/model_accuracy.png))
![Model F1 Score Comparison]([https://i.imgur.com/vHqQcSO.png](https://github.com/sawma-k/Optimizing-Post-Diasaster-Management/blob/df7df99340a8fe3ce7f3b01685e33bd1d475b879/Images/model_f1.png))


### 3. Resource Allocation (Bi-Objective Optimization)

A Mixed-Integer Linear Programming (MILP) model was developed to optimize the allocation of rescue teams, vehicles, and relief centers. The model has two primary objectives:

1.  **Minimize Total Rescue Distance ($Z_1$)**: Aims to reduce the total travel time for rescue forces, weighted by the number of rescuers required for each patient's severity level.
    $$\text{Min } Z_1 = \sum_{I}\sum_{J}\sum_{O}\sum_{M}\sum_{S}\sum_{L}X_{i,j,o,m,s,l} \cdot D_{ij}$$
2.  **Minimize Total Operational Cost ($Z_2$)**: Accounts for the costs of activating relief centers, training and deploying rescuers, and purchasing and using vehicles.
    $$\text{Min } Z_2 = \sum_{M}A_{i} \cdot Q_{i,m} + \sum_{I}\sum_{O}\sum_{L}C_{o} \cdot N_{i,o,l} + \sum_{I}Y_{i} \cdot CA_{i}$$

To find a balanced solution, the **LP-Metric Method ($Z_3$)** was used to create a single, normalized objective function that represents the trade-off between distance and cost.

## Results & Sensitivity Analysis

The optimization model was run under various scenarios to test its robustness. Key findings from the sensitivity analysis include:
* **Impact of Patient Severity ($N_s$)**: Increasing the number of rescuers required per patient ($N_s$) leads to a linear increase in both total travel distance ($Z_1$) and operational cost ($Z_2$).
* **Impact of Patient Volume ($F_m$)**: As the number of patients increases, both objectives ($Z_1$ and $Z_2$) and the required number of vehicles and rescuers rise predictably.
* **Impact of Budget**: A higher budget allows for the deployment of more vehicles and rescuers, improving service coverage.
* **Impact of Vehicle Capacity**: The required number of vehicles decreases as the capacity of each vehicle increases, demonstrating an inverse relationship.

<p float="left">
  <img src="[https://i.imgur.com/YvV0V8L.png](https://github.com/sawma-k/Optimizing-Post-Diasaster-Management/blob/df7df99340a8fe3ce7f3b01685e33bd1d475b879/Images/fig11.png)" width="48%" />
  <img src="[https://i.imgur.com/N742J7V.png](https://github.com/sawma-k/Optimizing-Post-Diasaster-Management/blob/df7df99340a8fe3ce7f3b01685e33bd1d475b879/Images/fig%207.png)" width="48%" /> 
   <img src="[https://i.imgur.com/N742J7V.png](https://github.com/sawma-k/Optimizing-Post-Diasaster-Management/blob/df7df99340a8fe3ce7f3b01685e33bd1d475b879/Images/fig%208.png)" width="48%" /> 
</p>

## Repository Structure

* `internship_project_final.ipynb`: A Jupyter Notebook containing the complete Python code for the MCDM analysis, machine learning model training, bi-objective optimization, and sensitivity analysis.
* `Designing a Post-Disaster Humanitarian Supply Chain (3).pdf`: The final project presentation summarizing the methodology, case study, and results.
* `Summer_research_report (5).pdf`: The detailed summer research report providing an in-depth explanation of the project's background, methodology, and findings.
* `README.md`: This file.

## Getting Started

To run this project, you will need Python 3 and the following libraries.

### Prerequisites

You can install the required Python libraries using pip:
```bash
pip install numpy pandas scikit-learn matplotlib pyomo
```

**Optimization Solver:**
This project uses **Gurobi** as the solver for the Pyomo optimization model. You must have Gurobi installed and have a valid license (a free academic license is available).
1.  Follow the instructions on the [Gurobi website](https://www.gurobi.com/) to install the solver.
2.  Ensure that your Gurobi license is activated.

### Running the Analysis

1.  Clone this repository to your local machine.
2.  Ensure you have all the prerequisites installed and your Gurobi license is active.
3.  Open and run the `internship_project_final.ipynb` notebook in a Jupyter environment. The notebook is structured with markdown cells explaining each step of the process, from data generation to the final sensitivity analysis plots.

## Acknowledgments

This project was completed under the supervision of **Prof. Ramesh Anbanandam** and **Siddharth Prajapati** at the Indian Institute of Technology, Roorkee. Their guidance and support were invaluable.

## References

* Behl, A. and Dutta, P. (2019). Humanitarian supply chain management: a thematic literature review and future directions of research. *Annals of Operations Research*.
* Choudhary, A. G. (2023). Designing a post-disaster humanitarian supply chain using machine learning and multi-criteria decision-making techniques. *Kybernetes*.
* Brintrup, A. P. (2020). Supply chain data analytics for predicting supplier disruptions. *International Journal of Production Research*.
* Budak, A. K. (2020). Real-time location systems selection by using a fuzzy MCDM approach. *Applied Soft Computing*.
