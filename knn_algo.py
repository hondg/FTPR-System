import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def recommend_player_knn(players, selected_player, k=5):
    # Select features for similarity calculation
    features = players[['Age', 'Goals', 'Assists', 'Hours']]

    # Filter players with the same position as the selected player
    same_position_players = players[players['Pos'] == selected_player['Pos']]

    # Drop the selected player from the dataset if present
    same_position_players = same_position_players[same_position_players['Rk'] != selected_player['Rk']]

    # If there are no players with the same position, return None
    if same_position_players.empty:
        print("\nNo players with the same position for recommendation.")
        return None

    # Fit KNN model
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(same_position_players[features.columns])

    # Find the k most similar players
    _, indices = knn_model.kneighbors([selected_player[features.columns]])

    # Get recommended players
    recommended_players = same_position_players.iloc[indices[0]]

    print("\nRecommended Players using kNN Algo:")
    print(recommended_players[['Player', 'Pos', 'Squad', 'Age', 'Goals', 'Assists', 'Hours']])

    return recommended_players


def plot_similarity_comparison(selected_player, recommended_players):
    features = ['Age', 'Goals', 'Assists', 'Hours']
    num_features = len(features)

    # Create subplots for each feature
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(15, 4))

    # Plot similarity comparison for each feature
    for i, feature in enumerate(features):
        axes[i].barh([selected_player['Player']] + recommended_players['Player'].tolist(),
                    [selected_player[feature]] + recommended_players[feature].tolist(),
                    color=['blue'] + ['green'] * len(recommended_players),
                    alpha=0.7)
        axes[i].set_title(feature)
        axes[i].set_xlim(0, max(selected_player[feature], max(recommended_players[feature])) + 5)

    plt.tight_layout()
    plt.show()
