import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import matplotlib.pyplot as plt


class MovieRecommender:
    def __init__(self, dataset_path):
        # future-proofing for future datasets
        self.movies = pd.read_csv(dataset_path)

        # Prepare data
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split(','))
        self.tfidf_matrix = None
        self.item_similarity = None

        self.users = {}  # Dictionary to store user data

    def prepare_content_based_filtering(self):
        # Generate TF-IDF matrix for content-based filtering
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres'].apply(lambda x: ' '.join(x)))

    def prepare_item_item_similarity(self):
        # Calculate item-item similarity based on genres
        self.item_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def add_user(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = {'watched': []}
            print(f"User {user_id} added successfully.")
        else:
            print(f"User {user_id} already exists.")

    def find_movie_id_by_title(self, title):
        # Search for a movie by title and return its ID
        movie = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]
        if not movie.empty:
            return movie.iloc[0]['id']
        else:
            return None

    def add_movie_to_watched(self, user_id, movie_title):
        if user_id in self.users:
            movie_id = self.find_movie_id_by_title(movie_title)
            if movie_id:
                self.users[user_id]['watched'].append(movie_id)
                print(f"Movie '{movie_title}' (ID: {movie_id}) added to User {user_id}'s watched list.")
            else:
                print(f"Movie '{movie_title}' not found in the dataset.")
        else:
            print("User ID does not exist.")

    def recommend_movies(self, user_id, num_recommendations=5):
        if user_id not in self.users:
            print("User ID does not exist.")
            return

        watched_movies = self.users[user_id]['watched']

        # Content-Based Recommendations
        content_recommendations = []
        if self.tfidf_matrix is not None and watched_movies:
            last_watched_idx = self.movies[self.movies['id'] == watched_movies[-1]].index[0]
            similarity_scores = linear_kernel(self.tfidf_matrix[last_watched_idx], self.tfidf_matrix).flatten()

            # Sort by similarity scores
            similar_indices = similarity_scores.argsort()[-num_recommendations - 1: -1][::-1]
            content_recommendations = [self.movies.iloc[i]['title'] for i in similar_indices if i not in watched_movies]

        # Item-Item Similarity Recommendations
        item_item_recommendations = []
        if self.item_similarity is not None:
            for movie_id in watched_movies:
                movie_idx = self.movies[self.movies['id'] == movie_id].index[0]
                similar_items = self.item_similarity[movie_idx]
                similar_indices = similar_items.argsort()[-num_recommendations - 1: -1][::-1]
                for idx in similar_indices:
                    if self.movies.iloc[idx]['id'] not in watched_movies:
                        item_item_recommendations.append(self.movies.iloc[idx]['title'])

        # Combine content-based and item-item similarity recommendations
        recommendations = list(set(content_recommendations + item_item_recommendations))[:num_recommendations]

        print(f"Hybrid Recommendations for User {user_id}: {recommendations}")

    def print_watchlist(self, user_id):
        if user_id not in self.users:
            print("User ID does not exist.")
            return

        watched_movies = self.users[user_id]['watched']
        if not watched_movies:
            print(f"User {user_id} has no movies in their watchlist.")
            return

        print(f"User {user_id}'s Watchlist:")
        for movie_id in watched_movies:
            movie_title = self.movies[self.movies['id'] == movie_id]['title'].values[0]
            print(f"- {movie_title} (ID: {movie_id})")

    def generate_user_graphs(self, user_id):
        if user_id not in self.users:
            print("User ID does not exist.")
            return

        watched_movies = self.users[user_id]['watched']
        if not watched_movies:
            print("No movies in the watched list for this user.")
            return

        user_movies = self.movies[self.movies['id'].isin(watched_movies)]

        # figure 1: Distribution of movie ratings
        plt.figure(figsize=(10, 6))
        user_movies['vote_average'].hist(bins=10, color='skyblue', edgecolor='black')
        plt.title("Distribution of Movie Ratings", fontsize=16)
        plt.xlabel("Rating", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # figure 2. Revenue vs Ratings
        plt.figure(figsize=(10, 6))
        plt.scatter(user_movies['vote_average'], user_movies['revenue'], alpha=0.7, color='purple')
        plt.title("Revenue vs Ratings", fontsize=16)
        plt.xlabel("Average Rating", fontsize=12)
        plt.ylabel("Revenue", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        # figure 3. Top 5 Movie Genres
        genres_list = user_movies['genres'].explode()
        top_genres = genres_list.value_counts().head(5)
        plt.figure(figsize=(10, 6))
        top_genres.plot(kind='bar', color='green', edgecolor='black')
        plt.title("Top 5 Movie Genres", fontsize=16)
        plt.xlabel("Genre", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


if __name__ == "__main__":
    dataset_path = "TMDB_movie_dataset_v11.csv"
    recommender = MovieRecommender(dataset_path=dataset_path)

    recommender.prepare_content_based_filtering()

    user_id = int(input("Enter your user ID: "))
    recommender.add_user(user_id)

    while True:
        movie_title = input("Enter a movie title to add to your watched list (or type 'END' to finish): ")
        if movie_title.upper() == "END":
            break
        recommender.add_movie_to_watched(user_id, movie_title)

    recommender.print_watchlist(user_id)
    recommender.recommend_movies(user_id)
    recommender.generate_user_graphs(user_id)
    plt.show(block=True)
