# 🎥 Netflix-Recommender

Our team was tasked to built a Netflix Recommendation System using restricted Boltzmann machine (RBM). As this was a competition we had to pit our prediction scores with other teams as well as a Linear Regression model.

  - **Context + Inspiration:** Recommendation systems are becoming more popular from Youtube to Spotify and Online Shopping. We are provided with a dataset that contains information about user's opinions of movies in the form of a rating. The task lies in finding what users might rate for movies they have not yet watched
  - The plain RBM model severely underperformed against the Linear Regression model. We subsequently improved the accuracy by adding a few extensions. The below are the list of extensions I contributed to:
      - Random Searching
      - Momentum
      - Biases
      - L1 Regularisation
      - Adaptive Learning Rate
      - Mini-Batches
  - _Challenges: It was a bottomless pit, a lot of the resources I found did not prove helpful as it did not translate to an improvement in prediction. This is mainly due to poor implementation. If there were more time, I will try using a Bayesian optimisation approach as well as consider rounding off values to the nearest integer to simulate a user's pick_
  - **_Technologies used:_** Vanilla Python
