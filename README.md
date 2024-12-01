# Predicting Cryptocurrency Volatility with HAR Models

## Overview
This project explores the use of HAR (Heterogeneous AutoRegressive) models for predicting the volatility of cryptocurrencies. For demonstration purposes, Bitcoin (BTC) and Ethereum (ETH) are analyzed using historical data sourced from Yahoo Finance. The project also includes a comparative analysis of different HAR model variants, their performance against GARCH models, and an ensemble approach combining both model types.

## Features
1. **Data Collection:**
   - Cryptocurrency data (BTC, ETH) is downloaded from Yahoo Finance.
   
2. **Model Variants:**
   - **HAR Model:** Standard heterogeneous autoregressive model.
   - **HAR-Q Model:** HAR model with quadratic terms.
   - **HAR-J Model:** HAR model incorporating jumps.

3. **Comparative Analysis:**
   - Performance of HAR variants is compared with the GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model.

4. **Ensemble Modeling:**
   - Combines HAR and GARCH models to evaluate the ensemble's predictive performance.

## Structure
- **`article/`**
  - Contains a detailed report discussing the methodology, results, and insights derived from the project.
- **`models/`**
  - Implementation of HAR, HAR-Q, HAR-J, and GARCH models.

- **`code/`**
  - Using models and comparing them

## Results
- **Model Comparisons:**
  - Performance metrics for HAR, HAR-Q, HAR-J, and GARCH models.
- **Ensemble Performance:**
  - Improved accuracy and robustness compared to individual models.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
