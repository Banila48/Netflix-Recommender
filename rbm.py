import numpy as np
import projectLib as lib

# set highest rating
K = 5

def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K))

def sig(x):
    ### TO IMPLEMENT ###
    # x is a real vector of size n
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    ret = []
    for i in x:
        ret.append(1/(1+np.exp(-i)))
    return ret   

def visibleToHiddenVec(v, w):
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F

    Fsize = np.shape(w)[1]
    x = []
    for hidden in range(Fsize):
        sumproduct = np.multiply(v,w[:,hidden,:]).sum()
        x.append(sumproduct)
    ret = sig(x)
    return np.asarray(ret)

def hiddenToVisible(h, w):
    ### TO IMPLEMENT ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie
    
    Msize = np.shape(w)[0]
    ret = []
    for visible in range(Msize):
        x = []
        for rating in range(5):
            sumproduct = np.multiply(h,w[visible,:,rating]).sum()
            x.append(sumproduct)
        values = sig(x)
        ret.append(values)
    return np.asarray(ret)

def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq):
    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5

    # Learning
    p = visibleToHiddenVec(v, w)
    
    # Unlearning
    wq_3d = wq[np.newaxis,:,:]
    vBar = hiddenToVisible(sample(p), wq_3d)
    ret = vBar[0]
    return ret
   

def predictRatingMax(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability
    prating = 1 + np.argmax(ratingDistribution, axis=0)
    return prating


def predictRatingExp(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over
    # the softmax applied to ratingDistribution
    total = sum(np.exp(ratingDistribution))
    for i in range(len(ratingDistribution)):
        ratingDistribution[i] = np.exp(ratingDistribution[i])/total
    pred = 0
    for i in range(1, len(ratingDistribution) + 1):
        pred += i * ratingDistribution[i - 1]
    return pred

def predictMovieForUser(q, user, W, training, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, training, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUser(user, W, training, predictType="exp"):
    ### TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    trStats = lib.getUsefulStats(training)
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for movie in trStats["u_movies"]]



####### Creating Bias Functions
## in order to reduce consistent errors in both visible and hidden nodes, we will be duplicating the above few functions
## by adding in a bias component

def visibleToHiddenVecBias(v, w, hb):
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F

    # hb is a vector of size F, representing the hidden biases for each hidden node.
    Fsize = np.shape(w)[1]
    x = [] 
    for hidden in range(Fsize):
        sumproduct = np.multiply(v,w[:,hidden,:]).sum()

        ## add bias
        sumproduct += hb[hidden]
        x.append(sumproduct)
    ret = sig(x)
    return np.asarray(ret)

def hiddenToVisibleBias(h, w, vb):
    ### TO IMPLEMENT ###
    # (hidden == j, visible == i, rating == k)
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.

    # vb is an array of size m x 5, representing the visible biases for each visible node and rating.
    Msize = np.shape(w)[0]
    ret = []
    for visible in range(Msize):
        x = []
        for rating in range(5):
            sumproduct = np.multiply(h,w[visible,:,rating]).sum()

            ## add bias
            sumproduct += vb[visible][rating]
            x.append(sumproduct)
        values = sig(x)
        ret.append(values)
    return np.asarray(ret)

def getPredictedDistributionBias(v, w, wq, hb, vb):
    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5

    # Learning
    p = visibleToHiddenVecBias(v, w, hb)

    # Unlearning
    wq_3d = wq[np.newaxis,:,:]
    vBar = hiddenToVisibleBias(sample(p), wq_3d, vb)
    ret = vBar[0]

    return ret

def predictMovieForUserBias(q, user, W, hb, vb, training, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"

    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predictBias(movies, users, W, hb, vb, training, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE

    # This function uses biases
    return [predictMovieForUserBias(movie, user, W, hb, vb, training, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUserBias(user, W, hb, vb, training, predictType="exp"):
    ### TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    # This function uses biases
    trStats = lib.getUsefulStats(training)
    # print(f"Predicting for user {user}")
    return [predictMovieForUserBias(movie, user, W, hb, vb, training, predictType=predictType) for movie in trStats["u_movies"]]

##### Implement mini-batches gradient descent
## Purpose is to have a lower generalization error, faster convergence via a regularization effect
## More efficient

def create_mini_batches(visitingOrder, batch_size):
 
    # Randomize data point
    mini_batches = []
    n_minibatches = len(visitingOrder) // batch_size 

    for i in range(n_minibatches):
        mini_batch = visitingOrder[i * batch_size:(i + 1) * batch_size]
        mini_batches.append(mini_batch)
    if len(visitingOrder) % batch_size != 0:
        mini_batch = visitingOrder[(i + 1) * batch_size:len(visitingOrder)]
        mini_batches.append(mini_batch) # if the length of visitingOrder is not a multiple of batch size
    return mini_batches

