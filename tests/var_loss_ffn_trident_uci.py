"""
Rescuing performance degradation due to high noise my minimizing variance of the finla layer.
- Consider the 32-hidden layer network.
- Modify the loss function to include variance loss:
    total_loss = ce_loss + alpha * var_loss
- var_loss = variance of the output of the final layer across the batch
- alpha is a hyperparameter that controls the trade-off between the cross-entropy loss and the
- sweep over the 
Notes: TODO

"""