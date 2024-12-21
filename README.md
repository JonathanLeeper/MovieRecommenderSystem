# MovieRecommenderSystem

When it comes to the film industry, we have had countless years of many different films, projects, and indie titles that are worthy of being known to the public. We can easily have recommendations handed to us via word of mouth, marketing, and popularity, but it can be difficult to determine the relationship between every single movie. This aims to develop a recommendation system using filtering techniques to provide those personalized suggestions to the user.

## Background/History

Recommender systems have evolved as a critical component of the entertainment industry, particularly with the growth of big data and machine learning techniques. Early systems relied on simple content-based approaches, where recommendations were based on movie attributes (e.g., genre, director, cast). More advanced systems now utilize collaborative filtering, leveraging user preferences and behaviors to improve recommendation accuracy.

The dataset for this project is sourced from the TMDB 5000 Movie Dataset on Kaggle, which contains metadata for 5,000 movies. By integrating  hybrid concept of item-item similarity and content-based filtering, this project aims to improve upon existing methods and develop a robust recommendation system.

## Data Explanation

The TMDB dataset comprises key information about the movies within the dataset. This includes the movie title, the genres for each film, keywords associated with the movie, revenue, runtime, and voter average rating. For data cleaning, it is necessary to fill values for those we are currently working with – in this case, our genre’s value. This also needs to be split, as they are currently utilizing multiple genres within one field.

##Methods

The recommender system employs two main techniques: Content-Based Filtering & Item-Item Similarity.
•	Content-Based Filtering focuses on metadata (e.g., genres, keywords) to recommend movies with similar features. The techniques for this include TF-IDF vectorization and cosine similarity.
•	Item-Item similarity uses the movies currently within that watch list and compares them with movies that are not considered watched and compares them with the content-based filtering on if they would be compatible to be added.

## Analysis

For the EDA for this dataset, the insights include top genres, rating distribution, and revenue vs. ratings. We can create these graphs for the entire dataset but will need to be refined for a user and their own metrics. Below, I have shown what is included within the dataset altogether, while the presentation showcases a user-by-user basis of these visualizations.

Results from the implemented models can be compared using recall and precision, if the dataset would allow for this. The graphs will illustrate the accuracy of predictions and the trade-offs between techniques.

## Conclusion

While the goal of this project was to create a user recommendation system based on collaborative filtering and content-based filtering, due to the dataset limitations we were unable to work through this, as well as not having a userbase to help with the collaborative filtering. However, for our program to implement the two current models and apply recommendations based on the genres, it is a step in the right direction for proper implementation and goals.

## Assumptions

The assumptions beforehand were that the rating column and the genre column would be extremely helpful in the previous models, as well as good metadata for each included movie within it.

## Limitations

Within this dataset, our main current limitations are the dataset, and its data stored, a cold-start problem for recommendations, and the size of the dataset.

## Challenges

The largest challenge for future implementations and improvements for this code would be to work with other columns within this dataset and implement them accordingly. For reference, I was able to find a great source online on how a user was able to create a recommender system using this dataset, which is an extreme project that includes all the fields within the set. However, for this project and its scope, this would be a project currently outside the process of limitations. For future use, I would love to incorporate these types of goals into my own recommender system.

##Future Uses/Additional Applications

 A large goal for this project would be to also include television shows, as their reach and content can be considered like a movie format. It would also be a goal to integrate user data and store this into a file for future use and additional input, which would help the longevity and eventual outcome of this model and program, as well as a GUI for easier use.

 One goal of mine would be to evaluate an older program I created, which is a video game analyzed dataset. This could work in the same way, with different filtering techniques, and come away with recommendations for the user as a video game recommender system.

## Ethical Assessment

Ethical considerations at this moment are not extreme. Ethical concerns could arise by implementing stored user data, however, the program does not have this functionality at this time.

## Audience Questions

1.	How does the system handle new users with no interaction history?
This returns no recommendations for this user.
2.	What metrics were used to evaluate performance?
When trying to create this, the limitations of the TMDB dataset made it so I was unable to convert the user input of movie title name and correctly output precision and recall, which would give the greatest evaluation out of all options. A great implementation for a large-scale implementation of this project would be a user rating those recommendations – similar to a program such as Steam’s discovery queue. This can help the model understand what the user would want to be recommended.
3.	How does your system mitigate bias toward popular movies?
With the limitations of the dataset at hand, there currently is no way to mitigate bias towards popular movies, as the dataset is incomplete in what is being handled. If the dataset were to be trimmed, this would also result in bias towards popular movies – more research needs to be done for this to be properly mitigated.
4.	What challenges did you face in preprocessing the dataset?
When processing the dataset, if this were to remove columns that were incomplete, this would wipe a lot of the data that would remove recommendations other than other popular movies. 
5.	How scalable is the system for larger datasets?
This system is not entirely made for larger datasets, as the TF-IDF matrix is not optimized for those larger sets. This could be worked around by implementing k-nearest neighbor, which would require training the data.
6.	How do collaborative and content-based filtering interact?
Within the dataset, as I have not yet implemented a rating system for users, a collaborative filtering model has not been able to be implemented. This has been replaced by a item-item similarity recommendations, which hybrids together with the content based recommendations to create a recommendations list for the user.
7.	How does the system ensure ethical considerations like user privacy?
As of now, no personal data is stored from the user, as the only accepted parameter from the user is ID as of now, so no ethical considerations at this moment.
8.	Can the model adapt to real-time user behavior?
The current model does not – however, this can be implemented by real-time recommendations during user input of movies.
9.	What are the limitations of the TMDB dataset?
Incomplete metadata & no user interaction data. Without these two metrics, it would not be possible to generate the type of values and outputs I would be requesting.
10.	What future improvements do you recommend?
For this specific model, I would recommend many things.
•	A stronger dataset, which would help mitigate bias, cold start solutions, and user interactivity
•	Incorporate user ratings as they add movies, which would be strong within a GUI
•	Improving the scalability when using a stronger dataset, such as implementing those nearest neighbors models.

## References

Asaniczka. (2024, December 14). Full TMDB movies dataset 2024 (1M movies). Kaggle. https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
 Erikbruin. (2020, February 29). Movie recommendation systems for TMDB. Kaggle. https://www.kaggle.com/code/erikbruin/movie-recommendation-systems-for-tmdb 
