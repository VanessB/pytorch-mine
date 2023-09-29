import math
import torch


class DVRSecondTerm(torch.autograd.Function):
    """
    Implementation of the second term of DVRLoss, unbiased gradient.
    """
    
    # Denominator regularization.
    EPS = 1e-6
    
    @staticmethod
    def forward(ctx, T_marginal: torch.tensor, moving_average: float,
                precomputed_logmeanexp_T_marginal: torch.tensor=None) -> torch.tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        T_marginal : torch.tensor
             Critic network value on a batch from the product of the marginal distributions.
        moving_average : float
             Current moving average of expectation of exp(T),
             which is used for back propagation.
        precomputed_logmeanexp_T_marginal : torch.tensor, optional
             Precomputed value of the second term of DVRLoss (optional).
        """
        
        # Needed for the gradient computation.
        # moving_average is not required for the forward pass.
        ctx.save_for_backward(T_marginal, moving_average)
        
        if precomputed_logmeanexp_T_marginal is None:
            return torch.logsumexp(T_marginal, dim=0) - math.log(T_marginal.shape[0])
        else:
            return precomputed_logmeanexp_T_marginal


    @staticmethod
    def backward(ctx, grad_output: torch.tensor) -> torch.tensor:
        """
        Backward pass.
        
        Parameters
        ----------
        grad_output : torch.tensor
            Output value gradient.
        """

        T_marginal, moving_average = ctx.saved_tensors
        grad = grad_output * T_marginal.exp().detach() / (moving_average * T_marginal.shape[0] + DVRSecondTerm.EPS)
        return grad, None#, None


class DVRLoss(torch.nn.Module):
    """
    Kullback-Leibler divergence in the Donsker-Varadhan form implementation.
    
    Attributes
    ----------
    biased : bool
        Whether to use estimate with unbiased gradient.
    alpha : float
        Exponential moving average coefficient.
    moving_average : float
        Moving average of expectation of exp(T).
    
    Methods
    -------
    forward
        Forward pass.
    """


    def __init__(self, biased: bool=False, alpha: float=0.01):      
        if not isinstance(biased, bool):
            raise TypeError("Parameter `biased' has to be boolean")
            
        if not isinstance(alpha, float):
            raise TypeError("Parameter `alpha' has to be float")
            
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Parameter `alpha' has to be within the range [0; 1]")
            
        super().__init__()
        
        self.biased = biased
        self.alpha  = alpha
        self.moving_average = None
        
        
    def forward(self, T_joined: torch.tensor, T_marginal: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        T_joined : torch.tensor
            Critic network value on a batch from the joined distribution.
        T_marginal : torch.tensor
            Critic network value on a batch from the product of the marginal distributions.
        """
        
        mean_T_joined = torch.mean(T_joined, dim=0)
        logmeanexp_T_marginal = torch.logsumexp(T_marginal, dim=0) - math.log(T_marginal.shape[0])
                                                                               
        if self.biased:
            return -mean_T_joined + logmeanexp_T_marginal
        else:
            self.update_moving_average(torch.exp(logmeanexp_T_marginal).detach().item())
            return -mean_T_joined + DVRSecondTerm.apply(
                T_marginal,
                torch.tensor(self.moving_average, dtype=torch.float32)#,
                #logmeanexp_T_marginal
            )
            
            
    def update_moving_average(self, current_average: float):
        """
        Update the moving average.
        
        Parameters
        ----------
        current_average : float
            Current batch average.
        """
        
        if self.moving_average is None:
            self.moving_average = current_average
        else:
            self.moving_average = self.alpha * current_average + (1.0 - self.alpha) * self.moving_average