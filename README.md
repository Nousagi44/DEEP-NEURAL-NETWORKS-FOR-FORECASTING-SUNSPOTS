# Sunspot & CO2 Time Series Forecasting with Custom Deep Neural Networks

[![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=matlab&logoColor=white)](https://www.mathworks.com/products/matlab.html)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-FF2D00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Time Series](https://img.shields.io/badge/Time%20Series%20Analysis-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Introduction

This project delves into the challenging domain of **time series forecasting** using custom-built deep neural networks implemented entirely in MATLAB. The core objective is to rigorously compare different network architectures and advanced training techniques on two distinct real-world datasets: the **chaotic sunspot activity** and the **trending-seasonal atmospheric CO2 concentration**.

By implementing fundamental deep learning concepts from scratch, this work provides empirical insights into how network depth, optimization strategies (Stochastic Gradient Descent with momentum), and regularization methods (dropout and pruning) impact predictive performance and generalization across varied time series characteristics.

## 🎯 Problem Statement

Accurate forecasting of non-linear and complex time series, such as irregular solar cycles or fluctuating CO2 levels with underlying trends and seasonality, remains a significant challenge. These series are vital for scientific understanding (e.g., solar weather prediction, climate modeling) but pose difficulties for traditional linear models. This project aims to demonstrate and evaluate the capability of custom-built deep neural networks to effectively learn and predict patterns in such complex data.

## 🛠️ Methodology

The project implements and evaluates four distinct neural network configurations, applying them consistently to both datasets.

### 1. Data Acquisition & Preprocessing

*   **Sunspot Dataset:** Yearly sunspot numbers (1700-2000) sourced from SILSO via Carnegie Mellon University.
*   **CO2 Dataset:** Monthly Mauna Loa CO2 concentrations (1959-1997) from the University of Auckland's Statlib.
*   **Preprocessing:**
    *   Min-Max Normalization to the range [-1, 1].
    *   **5-Lag Input:** A sliding window creates input-output pairs where 5 consecutive prior values predict the subsequent value.
    *   **Data Split:** Chronological 80\% training / 20\% testing split to assess generalization.

### 2. Network Architectures

Two primary feedforward architectures were developed:

*   **Deep Feedforward Network (DFN) / "CNN-like":**
    *   **Structure:** 5 (Input) - 5 (Hidden Layer 1) - 5 (Hidden Layer 2) - 5 (Hidden Layer 3) - 1 (Output).
    *   All connections are **fully connected**.
    *   **Activation:** `tanh` for hidden layers, linear for output.
    *   **Bias:** Implicit bias units added to each layer's input computation.
    *   *(Note: While labelled "CNN" in some contexts for its depth, this network consists purely of fully connected layers and does not employ explicit convolutional or pooling operations.)*

*   **Shallow Multi-Layer Perceptron (MLP):**
    *   **Structure:** 5 (Input) - 5 (Hidden Layer 1) - 1 (Output).
    *   **Activation:** `tanh` for hidden, linear for output.
    *   **Bias:** Implicit bias units included.
3. Training Algorithm: Incremental SGD with Momentum
Online Stochastic Gradient Descent (SGD): Weights are updated after processing each individual training sample for efficiency and regularisation benefit from gradient noise.
Update Rule: w(t+1) = w(t) - η * ∇E_i(w(t))
Momentum: Applied to the DFNs to accelerate convergence and dampen oscillations. A velocity term v accumulates past gradients.
v(t+1) = α * v(t) + η * ∇E_i(w(t))
w(t+1) = w(t) - v(t+1)
α was 0.9 for Sunspot DFN, 0.5 for CO2 DFN.
Custom Implementation: All forward passes, backpropagation, and weight updates are implemented from scratch in MATLAB functions.

4. Regularisation Techniques
Dropout (DFN only):
Randomly sets 1% of neurons in Hidden Layer 2 to zero during training.
Aims to prevent co-adaptation and improve generalization.
Magnitude-based Pruning (MLP only):
After each training epoch, weights with an absolute value less than 0.001 are set to zero.
Aims to simplify the model and reduce overfitting.

5. Evaluation
Metrics: Mean Squared Error (MSE) and Relative Absolute Error (RAE) on training and test sets.
Visualizations: Plots showing MSE convergence, training fit, and one-step-ahead forecasts.
📊 Results & Key Findings
The project's experiments revealed that optimal model choice and regularization strategies are highly dependent on the characteristics of the time series data.
Network Configuration	MSE Train	RAE Train	MSE Test	RAE Test
Sunspot Dataset				
CNN+Dropout+Momentum	0.02249	0.38618	0.07816	0.43935
CNN NoDropout+Momentum	0.02290	0.39090	0.08103	0.43964
MLP NoPrune	0.04056	0.51022	0.10797	0.50166
MLP Prune	0.04104	0.51019	0.11095	0.50064
CO2 Dataset				
CNN+Dropout+Momentum	0.003355	0.12608	0.014600	0.85469
CNN NoDropout+Momentum	0.003863	0.13229	0.013894	0.83994
MLP NoPrune	0.007149	0.20023	0.009941	0.74856
MLP Prune	0.006028	0.18279	0.008587	0.68086
Key Observations:
Architecture Superiority is Context-Dependent:
For Sunspots (chaotic): The DFN ("CNN-like") models consistently outperformed MLPs, indicating depth's benefit for complex dynamics.
For CO2 (trending/seasonal): The MLP models (especially pruned) achieved better test performance than DFNs, suggesting simpler models can excel on more regular data.
Momentum: Proved highly effective in accelerating and stabilizing DFN training across both datasets.
Dropout:
Provided a marginal positive impact on Sunspot DFNs.
Slightly hindered CO2 DFN performance, indicating it can disrupt learning of highly regular patterns at the chosen rate.
Pruning:
Slightly detrimental for Sunspot MLPs (model likely not over-parameterized).
Highly beneficial for CO2 MLPs (significantly improved generalization), suggesting effective complexity reduction.
📂 Project Structure
sunspot_tables.m: Main MATLAB script for the sunspot data analysis. Contains all core network implementation (initialization, forward pass, backpropagation, training loops with momentum/dropout) and MLP functions (training with/without pruning), as well as plotting and metric calculation helpers.
co2.m: MATLAB script adapted from sunspot_tables.m for the CO2 data analysis. Uses the same core helper functions to ensure consistent methodology.
sunspot.dat: Historical sunspot number dataset.
co2.csv: Mauna Loa atmospheric CO2 concentration dataset.
🚀 How to Run the Code
Clone the Repository:
MATLAB Installation: Ensure you have MATLAB (R2023b or compatible version) installed.
Open MATLAB: Launch MATLAB.
Navigate to Project Folder: In MATLAB's Current Folder browser, navigate to the cloned sunspot-co2-forecasting directory.
Run Scripts:
To run the sunspot analysis: Type sunspot_tables in the MATLAB Command Window and press Enter.
To run the CO2 analysis: Type co2 in the MATLAB Command Window and press Enter.
Expected Output:
The scripts will display training progress in the Command Window (SSE and MSE per epoch) and generate several plots visualizing training fit, test set forecasts, and MSE convergence curves for each model. Final performance tables will also be displayed in the Command Window.
⚠️ Dependencies
MATLAB (R2023b or later recommended)
📧 Contact
For any questions or collaborations, feel free to reach out:
Oluwaseun Onimole - oonim001@gold.ac.uk
