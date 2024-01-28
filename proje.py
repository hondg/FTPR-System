import matplotlib.pyplot as plt
import pandas as pd
import ann_algo
import knn_algo
import random_forest_regressor as rfr


# Load the datasets
players_df = pd.read_csv("Player Data.csv", encoding='ISO-8859-2', delimiter=';')
teams_df = pd.read_csv("Team Data.csv", encoding='ISO-8859-2', delimiter=';')
# Convert the 'Minutes' column to hours
players_df['Hours'] = players_df['Min'] / 60

# Select relevant features
player_features = ['Rk', 'Player', 'Pos', 'Age', 'Nation', 'Goals', 'Assists', 'Hours', 'Squad']
team_features = ['Rk', 'Squad', 'Country', 'W', 'L', 'D']


# Drop unnecessary columns
players_df = players_df[player_features]
teams_df = teams_df[team_features]


# Handle missing values if necessary
players_df = players_df.dropna()
teams_df = teams_df.dropna()

print(teams_df)
print("\n")
print(players_df)
print("\n")

# Descriptive statistics
print("Player Statistics:")
print(players_df.describe())

print("\nTeam Statistics:")
print(teams_df.describe())

# Example: Histogram of player ages
plt.hist(players_df['Age'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Player Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


def display_teams(teams):
    print("Available Teams:")
    for index, team in teams.iterrows():
        print(f"{index + 1}. {team['Squad']} ({team['Country']})")


def select_team():
    team_index = int(input("Enter the number of the team you want to select: ")) - 1
    selected = teams_df.iloc[team_index]
    print(f"You selected the team: {selected['Squad']} ({selected['Country']})")
    return selected


def display_players_in_team(players, team):
    team_players = players[players['Squad'] == team['Squad']]
    if team_players.empty:
        print(f"\nNo players found for {team['Squad']} ({team['Country']}).")
    else:
        print(f"\nPlayers in {team['Squad']} ({team['Country']}):")
        print(team_players[['Rk', 'Player', 'Pos', 'Age', 'Nation', 'Goals', 'Assists', 'Hours', 'Squad']])


display_teams(teams_df)
selected_team = select_team()
display_players_in_team(players_df, selected_team)

# Assuming the user has selected a player from the displayed list
selected_player_index = int(input("\nEnter the number of the player you want to select: ")) - 1
selected_player = players_df.iloc[selected_player_index]

# Display the selected player's information
print(f"\nSelected Player: {selected_player['Player']} ({selected_player['Pos']})")

"""# Using KNN Algorithms
# Recommend players with the same position using KNN
knn_recommended_players = knn_algo.recommend_player_knn(players_df, selected_player)

# Plot similarity comparison for the selected and recommended players
knn_algo.plot_similarity_comparison(selected_player, knn_recommended_players)


# Using ANN Algorithms
# Recommend players with the same position using ANN
ann_recommended_player = ann_algo.recommend_player_ann(players_df, selected_player)

#
ann_algo.plot_detailed_comparison(selected_player, ann_recommended_player)"""

# User chooses the algorithm
algorithm_choice = input("Choose the algorithm (1: KNN, 2: ANN, 3: Random Forest): ")

if algorithm_choice == '1':
    # Use KNN
    recommended_players = knn_algo.recommend_player_knn(players_df, selected_player)
    knn_algo.plot_similarity_comparison(selected_player, recommended_players)
elif algorithm_choice == '2':
    # Use ANN
    recommended_players = ann_algo.recommend_player_ann(players_df, selected_player)
    ann_algo.plot_detailed_comparison(selected_player, recommended_players)
elif algorithm_choice == '3':
    # Use Random Forest
    recommended_players = rfr.recommend_player_random_forest(players_df, selected_player)
    rfr.plot_random_forest_comparison(selected_player, recommended_players)
else:
    print("Invalid choice. Please choose a valid algorithm.")
