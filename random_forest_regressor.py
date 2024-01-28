# random_forest_algo.py

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np


def recommend_player_random_forest(players, selected_player, top_n=5, random_state=42, verbose=True, **rf_params):
    # Select features for similarity calculation
    features = players[['Age', 'Goals', 'Assists', 'Hours']]

    # Filter players with the same position as the selected player
    same_position_players = players[players['Pos'] == selected_player['Pos']]

    # Drop the selected player from the dataset if present
    same_position_players = same_position_players[same_position_players['Rk'] != selected_player['Rk']]

    # If there are no players with the same position, return None
    if same_position_players.empty:
        if verbose:
            print("\nNo players with the same position for recommendation.")
        return None

    # Initialize Random Forest model
    rf_model = RandomForestRegressor(random_state=random_state, **rf_params)

    # Define features and target variable
    X = same_position_players[features.columns]

    y = same_position_players.index  # You may need to adjust this based on your data

    # Fit the Random Forest model
    rf_model.fit(X, y)

    # Predict similarity scores
    similarity_scores = rf_model.predict([selected_player[features.columns]])

    # Get the indices of the top N most similar players
    recommended_index = similarity_scores.argsort()[-top_n:][::-1]
    # Retrieve the top N recommended players
    recommended_player = same_position_players.iloc[recommended_index]

    if verbose:
        print(selected_player[['Player', 'Pos', 'Age', 'Goals', 'Assists', 'Hours']])
        print("\nRecommended Player using Random Forest Algorithm:")
        print(recommended_player[['Player', 'Pos', 'Age', 'Goals', 'Assists', 'Hours']])

    return recommended_player


def plot_random_forest_comparison(selected_player, recommended_players):
    # Features for the detailed comparison
    features = ['Age', 'Goals', 'Assists', 'Hours']

    # Number of features
    num_features = len(features)

    # Set up positions for the grouped bar chart
    positions = np.arange(num_features)
    bar_width = 0.1  # Width of each bar

    # Create subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the selected player's values
    ax.bar(positions, selected_player[features], bar_width, label=selected_player['Player'])

    # Plot the recommended players' values
    for i, (_, recommended_player) in enumerate(recommended_players.iterrows()):
        ax.bar(positions + (i + 1) * bar_width, recommended_player[features], bar_width,
               label=recommended_player['Player'])

    # Set labels and title
    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_title('Detailed Player Comparison (Random Forest)')
    ax.set_xticks(positions + 0.5 * bar_width)
    ax.set_xticklabels(features)

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()
