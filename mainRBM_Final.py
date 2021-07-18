import numpy as np
import rbm
import projectLib as lib
import time
import os
import pandas as pd


##from sklearn.model_selection import RandomizedSearchCV
##from sklearn.model_selection import cross_val_score
from openpyxl import Workbook
from openpyxl.chart import (
    LineChart,
    Reference,
)

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)






"""
Thought Process

1)  Implementing most of the extensions in the doc (Momentum, LR, Early Stopping, Reg, Biases)
2) Perform Random Search
    - Randomly tuned the hypermaters to find the most optimal configurations in the least iteration
3) Train model using mini batches gradient descent
4) Observe results and prep for next optimisation

For convenience all oscillating variables will have "_" behind

"""
# Create list of values for Hyperparameters Testing
K = 5
F_ = [8,10,12] # Changed this for Submission 2
##F_ = [x-4 for x in F_] # Added this for the dropout layer

batch_size_ = [5, 10, 15] # Changed this for Submission 2

reg_lambda_ = [0.0010, 0.0011, 0.0012] # Regularisation. Test from 10^-3 to 10^-2

# ADAPTIVE LEARNING RATE
# To implement adaptive learning rate, we need to have an initial learning rate
# Each epoch learning rate is alpha/epoch where alpha is the initial learning rate

lr_ = [0.03,0.06, 0.09] # Initial Learning Rate

# MOMENTUM
## A medium article recommend the initial momentum beta to be from 0.5 to 0.9
## The momentum term increases dimensions whose gradients points in the same directions and reduces updates for
## dimensions whose gradients change directions. Thus we get faster convergence and reduced oscillation

mom_init_ = [0.5, 0.6, 0.7]
mom_final_ = [0.65, 0.75, 0.85,] # Adjusted this for Submission 2

# Bias Learning Rate
vblr_ = [0.01, 0.02, 0.03] # Changed
hblr_ = [0.015, 0.025, 0.035] # Changed

# Create Workbook
wb = Workbook()
final_results = []


# Random Searching
count = 0
num_osc = 5 # times we want to oscillate
osc_start_time = time.time()
prob = [0.0228*2, 0.1359*2, 0.6826] # Probability vector following Normal distribution. Probability need to sum up to 1
params = []
best_vlRMSEs = []
best_epochs = []
best_Ws = [] # weights
best_vbs = []
best_hbs =[]

### Defining droput_layer function that drops out elements in the input X with probabability "p"
##def dropout_layer(X, p):
##    assert 0 <= p <= 1
##    # In this case, all elements are dropped out
##    if p == 1:
##        return np.zeros_like(X)
##    # In this case, all elements are kept
##    if p == 0:
##        return X
##    mask = np.random.uniform(0, 1, X.shape) > p
##    return mask.astype(np.float32) * X / (1.0 - p)
    

for i in range (num_osc): 
    count += 1
    print(f"---------------------------------Running Oscillation #{count}---------------------------------")

    # Model HyperParameters
    K = 5
    F = np.random.choice(F_, p=prob)
    epochs = 30
    grad_lr_init = np.random.choice(lr_, p=prob)

    # Momentum
    mom_init = np.random.choice(mom_init_, p=prob)
    mom_final = np.random.choice(mom_final_, p=prob)

    # Learning Rate Schedule
    """
    We decided to implement three common types of decay following the article on adjusting the adaptive learning rate
    Time-based decay: lr = lr0/(1+kt) where lr, k are hyperparameters and t is the iteration number.
    Step decay: lr = lr0 * drop^floor(epoch / epochs_drop) 
    Exponential decay: lr = lr0 * e^(−kt)
    
    """
    decay_type = "time"
    time_decay_k_grad = grad_lr_init / epochs # Time-based decay
    drop = 0.5 # Step decay
    epochs_drop = 10.0 # Step decay
    exp_k = 0.1 # Exp decay

    # Regularisation
    reg_lambda = np.random.choice(reg_lambda_, p=prob)

    # Mini-batch
    batch_size = np.random.choice(batch_size_ ,p=prob)

    # Biases
    ## Visible
    vblr_init = np.random.choice(vblr_, p=prob)
    time_decay_k_vb = vblr_init / epochs  

    ## Hidden
    hblr_init = np.random.choice(hblr_, p=prob)
    time_decay_k_hb = hblr_init / epochs  

    # Compile Parameters
    param_list = [epochs, F, batch_size, reg_lambda,
                          grad_lr_init, vblr_init, hblr_init, mom_init, mom_final]
    print(param_list)
    params.append(param_list)

    # Prep Workbook
    workbook_name = f"Oscillation_Test"
    sheet_name = f"Oscillation_{count}"
    prediction_name = f"Prediction_Test"
    predict = True

    ## Renaming Sheets
    ws = wb.create_sheet(sheet_name)
    ws.title = sheet_name
    labels = ["Epoch", "trRMSE", "vlRMSE"]
    ws.append(labels)
    c1 = LineChart()
    c1.title = f"RMSE vs Epoch, epochs={epochs}, F={F}, batch_size={batch_size}, reg_lambda={reg_lambda}, " \
               f"lr={grad_lr_init}, vblr={vblr_init}, hblr={hblr_init}, " \
               f"beta_final={mom_final}, beta_init={mom_init}"
    c1.style = 12
    c1.y_axis.title = 'RMSE'
    c1.x_axis.title = 'Epoch'


    # Model Parameters
    
    W = rbm.getInitialWeights(trStats["n_movies"], F, K)

    # Biases
    vb = np.zeros((trStats["n_movies"], K))
    hb = np.zeros(F) 

    # VALIDATION RMSE
    best_vlRMSE = 100000



    ### Train Model
    """
    Thought Process
    We want to minimise the risk of overfitting
    In order to generalise our data better, we will be calculating the gradient descent and updating each iteration of it
    across a batch of multiple users. (create_mini_batches in rbm.py)

    We will need 3 for loops

    The function will repeat in each epoch and then each mini-batch and finally for each user.

    Last step is to add the gradient update for each user to each mini batch

    Do not confuse the gradient of weights and gradient of the bias yet
    
    Reference:
    1) https://www.mathworks.com/matlabcentral/answers/215197-rbm-linear-to-binary-layer
    2) https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    3) https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf
    4) https://www.cs.toronto.edu/~hinton/absps/cdmiguel.pdf
    
    """
    train_start_time = time.time()

    for epoch in range(1, epochs+1):
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)
        epoch_start_time = time.time()
        visitingOrder = rbm.create_mini_batches(visitingOrder, batch_size) 

        # Mini-batch loop
        for mini_batch in range(len(visitingOrder)):
            grad = np.zeros(W.shape)
            vbGrad = np.zeros(vb.shape)
            hbGrad = np.zeros(hb.shape)

            batch_pos_visible = np.zeros((trStats["n_movies"], K))
            batch_neg_visible = np.zeros((trStats["n_movies"], K))
            batch_pos_hidden = np.zeros(F)
            batch_neg_hidden = np.zeros(F)

            posprods = np.zeros(W.shape)
            negprods = np.zeros(W.shape)

            # Iterate through each user 
            
            for user in visitingOrder[mini_batch]:
                # Get the ratings of that user
                ratingsForUser = lib.getRatingsForUser(user, training)
                
                # Create universal arrays in order for the Training model to access movie IDs of the movie each user has rated
                ## 2D arrays with binary values 
                user_pos_visible = np.zeros((trStats["n_movies"], K))
                user_neg_visible = np.zeros((trStats["n_movies"], K))

                # Build the visible input
                v = rbm.getV(ratingsForUser)

                # Get the weights associated to movies the user has seen
                weightsForUser = W[ratingsForUser[:, 0], :, :]

                ### LEARNING ###
                # Propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVecBias(v, weightsForUser, hb)
                # Get positive gradient
                # Note that we only update the movies that this user has seen!
                posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)
                pos_hidden = rbm.visibleToHiddenVecBias(v, weightsForUser, hb)
                batch_pos_hidden += pos_hidden
                pos_visible = v

                ### UNLEARNING ###
                # Sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb)
                # Propagate back to get "negative data"
                negData = rbm.hiddenToVisibleBias(sampledHidden, weightsForUser, vb)
                # Propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVecBias(negData, weightsForUser, hb)
                # Get negative gradient
                # Note that we only update the movies that this user has seen!
                negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)
                neg_hidden = rbm.visibleToHiddenVecBias(negData, weightsForUser, hb)
                batch_neg_hidden += neg_hidden
                neg_visible = negData

                # Update universal arrays
                for i in range(ratingsForUser.shape[0]):
                    user_pos_visible[ratingsForUser[i][0]] = pos_visible[i]
                    user_neg_visible[ratingsForUser[i][0]] = neg_visible[i]

                # Add to batch
                batch_pos_visible+= user_pos_visible
                batch_neg_visible += user_neg_visible


                # Learning Rate Decay
                # Choose weight learning rate for this epoch
                grad_lr = grad_lr_init
                if decay_type == "time":
                    grad_lr = grad_lr_init / (1 + time_decay_k_grad * (epoch-1))
                elif decay_type == "step":
                    grad_lr = grad_lr_init * drop ** np.floor((epoch-1) / epochs_drop)
                elif decay_type == "exp":
                    grad_lr = grad_lr_init * np.exp(-exp_k * (epoch-1))
                else:
                    print("Error: Wrong Decay Type")

                # Choose visible bias learning rate for this epoch
                vblr = vblr_init 
                if decay_type == "time":
                    vblr = vblr_init / (1 + time_decay_k_vb * (epoch - 1))
                elif decay_type == "step":
                    vblr = vblr_init * drop ** np.floor((epoch - 1) / epochs_drop)
                elif decay_type == "exp":
                    vblr = vblr_init * np.exp(-exp_k * (epoch - 1))
                else:
                    print("Error: Wrong Decay Type")

                # Choose hidden bias learning rate for this epoch
                hblr = hblr_init
                if decay_type == "time":
                    hblr = hblr_init / (1 + time_decay_k_hb * (epoch - 1))
                elif decay_type == "step":
                    hblr = hblr_init * drop ** np.floor((epoch - 1) / epochs_drop)
                elif decay_type == "exp":
                    hblr = hblr_init * np.exp(-exp_k * (epoch - 1))
                else:
                    print("Error: Wrong Decay Type")


            # Calculate gradient for weights, accounting for batch size and cost of weights
            diff = posprods - negprods
            grad = grad_lr * (diff / len(visitingOrder[mini_batch]) - reg_lambda * W)

            # Calculate gradient for visible biases, accounting for batch size
            vbDiff = batch_pos_visible - batch_neg_visible
            vbGrad = vblr/len(visitingOrder[mini_batch]) * vbDiff

            # Calculate gradient for hidden biases, accounting for batch size
            hbDiff = batch_pos_hidden - batch_neg_hidden
            hbGrad = hblr/len(visitingOrder[mini_batch]) * hbDiff
            

            # Momentum
            # No momentum for first epoch
            if epoch == 1:
                beta_grad = grad
                beta_vbGrad = vbGrad
                beta_hbGrad = hbGrad
            # Use mom_init for first 4 epochs
            elif epoch < 5:
                beta_grad = mom_init * beta_grad + grad
                beta_vbGrad = mom_init * beta_vbGrad + vbGrad
                beta_hbGrad = mom_init * beta_hbGrad + hbGrad
            # Use mom_final for subsequent epochs
            else:
                beta_grad = mom_final * beta_grad + grad
                beta_vbGrad = mom_final * beta_vbGrad + vbGrad
                beta_hbGrad = mom_final * beta_hbGrad + hbGrad

            # Update weight and bias at the end of each batch
            W += beta_grad
            vb += beta_vbGrad
            hb += beta_hbGrad

        # Print Epoch RMSE
        # We predict over the training set
        tr_r_hat = rbm.predictBias(trStats["movies"], trStats["users"], W, hb, vb, training)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
        
        
        

        # We predict over the validation set
        vl_r_hat = rbm.predictBias(vlStats["movies"], vlStats["users"], W, hb, vb, training)
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

        # Print Epoch Stats
        epoch_time = (time.time() - epoch_start_time)/60
        print(f"### EPOCH {epoch} ###")
        print(f"Training loss = {trRMSE}")
        print(f"Validation loss = {vlRMSE}")
        print(f"Epoch {epoch} took {epoch_time} minutes.")
        result = [epoch, trRMSE, vlRMSE]
        ws.append(result)
        final_results.append(result)

        # Early Stopping using best weights
        if vlRMSE <= best_vlRMSE:
            best_vlRMSE = vlRMSE
            best_W = W
            best_epoch = epoch
            best_vb = vb
            best_hb = hb


### Dropout
##    dropout_layer(W, 0.8)
    
    # Best weights
   
    
    W = best_W
    vb = best_vb
    hb = best_hb
    best_vlRMSEs.append(best_vlRMSE)
    best_epochs.append(best_epoch)
    best_Ws.append(best_W)
    best_vbs.append(best_vb)
    best_hbs.append(best_hb)
    print(f"The best vlRMSE is {best_vlRMSE} from Epoch {best_epoch}.")

    
    print("TRAINING DONE")
    train_time = (time.time() - train_start_time)/60
    print(f"Training took {train_time} minutes.")

    # Save to Excel
    data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=epochs)
    c1.add_data(data, titles_from_data=True)
    ws.add_chart(c1, "A10")

    ### End Oscillation



# Compilation

# Save to Workbook
wb.remove(wb["Sheet"])
# wb.save(f"{file_name}({F},{epochs},{gradientLearningRate_init}).xlsx")
wb.save(f"{workbook_name}({np.min(best_vlRMSEs)}).xlsx")

pd.DataFrame(final_results).to_csv(f'{np.min(best_vlRMSEs)}.csv')

# Print best result from each Oscillation 
print("ALL OSCILLATIONS COMPLETED --------------------------")
oscillations_time = (time.time() - osc_start_time)/60
print(f"Oscillations took {oscillations_time} minutes.")
print(f"Best vlRMSEs: {best_vlRMSEs}")
print(f"Best epochs: {best_epochs}")
best_osc = np.argmin(best_vlRMSEs)
print(f"The best vlRMSE {np.min(best_vlRMSEs)} was from Oscillation #{best_osc+1}.")


### Prediction

if predict:
    print(f"PREDICTING WITH PARAMETERS FROM OSCILLATION {best_osc+1}: ----------------------\n"
          f"Epochs={params[best_osc][0]}\n"
          f"F={params[best_osc][1]}\n"
          f"batch_size={params[best_osc][2]}\n"
          f"reg_lambda={params[best_osc][3]}\n"
          f"gradientLearningRate_init={params[best_osc][4]}\n"
          f"vbLearningRate_init={params[best_osc][5]}\n"
          f"hbLearningRate_init={params[best_osc][6]}\n"
          f"momentum_beta_final={params[best_osc][7]}\n"
          f"momentum_beta_init={params[best_osc][8]}")

    # Predict using best weights and biases
    predict_start_time = time.time()
    best_W, best_hb, best_vb = best_Ws[best_osc], best_hbs[best_osc], best_vbs[best_osc]
    predictedRatings = np.array([rbm.predictForUserBias(user, W, hb, vb, training) for user in trStats["u_users"]])

    print("PREDICTION DONE")
    predict_time = (time.time() - predict_start_time)/60
    print(f"Prediction took {predict_time} minutes.")

    # Save prediction to txt file
    np.savetxt(f"{prediction_name}({np.min(best_vlRMSEs)})({epochs},{F},{batch_size},{reg_lambda},"
               f"{grad_lr_init},{vblr_init},{hblr_init},"
               f"{mom_final},{mom_init}).txt", predictedRatings)

