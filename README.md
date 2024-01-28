# Player Recommendation Project

## Overview

This project implements a player recommendation system using various machine learning algorithms. Users can interactively select a team and a player to receive personalized recommendations.

## Project Structure

- `Player Data.csv`: Dataset containing information about players.
- `Team Data.csv`: Dataset containing information about teams.

### Code Files

- `ann_algo.py`: Implements the Artificial Neural Networks (ANN) recommendation algorithm.
- `knn_algo.py`: Implements the K-Nearest Neighbors (KNN) recommendation algorithm.
- `random_forest_algo.py`: Implements the Random Forest recommendation algorithm.
- `main.py`: Main script that orchestrates the user interaction and recommendation process.

## Setup

1. Install dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

## Usage

- Follow the prompts to select a team and a player.
- Choose a recommendation algorithm (KNN, ANN, or Random Forest).
- View the recommended players and detailed comparisons.

## Dependencies

- matplotlib==3.4.3
- pandas==1.3.3
- scikit-learn==0.24.2
- numpy==1.21.2

## License

This project is licensed under the [MIT License](LICENSE).
