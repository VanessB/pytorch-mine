import torch
import math


class NeuralT(torch.nn.Module):
    """
    Basic class for neural network that computes T-statistics.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    
    def get_mutual_information(self, dataloader, loss: callable, device, permute: bool=True) -> float:
        """
        Mutual information estimation.
        
        Parameters
        ----------
        dataloader
            Data loader. Must yield tuples (x,y,z).
        loss : callable
            Mutual information neural estimation loss.
        device
            Comoutation device.
        permute : bool, optional
            Permute every batch to get product of marginal distributions.
        """
        
        # Disable training.
        was_in_training = self.training
        self.eval()
        
        sum_loss = 0.0
        total_elements = 0
        
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                x, y, z = batch
                batch_size = x.shape[0]
            
                T_joined   = self(x.to(device), y.to(device))
                T_marginal = self(z.to(device), y.to(device), permute=permute)
                
                sum_loss += loss(T_joined, T_marginal).detach().cpu().item() * batch_size
                total_elements += batch_size
                
        mutual_information = -sum_loss / total_elements
                
        # Enable training if was enabled before.
        self.train(was_in_training)
        
        return mutual_information


class BasicDenseT(NeuralT):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=100) -> None:
        super().__init__()
        
        self.linear_1 = torch.nn.Linear(X_dim + Y_dim, inner_dim)
        self.linear_2 = torch.nn.Linear(inner_dim, inner_dim)
        self.linear_3 = torch.nn.Linear(inner_dim, 1)
        
        self.activation = torch.nn.LeakyReLU()
        
        
    def forward(self, x: torch.tensor, y: torch.tensor, permute: bool=False) -> torch.tensor:
        if permute:
            x = x[torch.randperm(x.shape[0])]
        
        layer = torch.cat((x, y), dim=1)
        
        # First layer.
        layer = self.linear_1(layer)
        layer = self.activation(layer)
        
        # Second layer.
        layer = self.linear_2(layer)
        layer = self.activation(layer)
        
        # Third layer.
        layer = self.linear_3(layer)
        
        return layer
    
    
class BasicConv2dT(NeuralT):
    def __init__(self, X_size: int, Y_size: int) -> None:
        super().__init__()
        
        log2_remaining_size = 2
        
        # Convolution layers.
        X_convolutions_n = int(math.floor(math.log2(X_size))) - log2_remaining_size
        self.X_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(1, 8, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(8, 8, kernel_size=3, padding='same') for index in range(X_convolutions_n - 1)])
            
        Y_convolutions_n = int(math.floor(math.log2(X_size))) - log2_remaining_size
        self.Y_convolutions = torch.nn.ModuleList([torch.nn.Conv2d(1, 8, kernel_size=3, padding='same')] + \
                [torch.nn.Conv2d(8, 8, kernel_size=3, padding='same') for index in range(Y_convolutions_n - 1)])
            
        self.maxpool2d = torch.nn.MaxPool2d((2,2))

        # Dense layer.
        remaining_dim = 8 * 2**(2*log2_remaining_size)
        self.linear_1 = torch.nn.Linear(remaining_dim + remaining_dim, 100)
        self.linear_2 = torch.nn.Linear(100, 100)
        self.linear_3 = torch.nn.Linear(100, 1)
        
        self.activation = torch.nn.LeakyReLU()
        
        
    def forward(self, x: torch.tensor, y: torch.tensor, permute: bool=False) -> torch.tensor:
        if permute:
            x = x[torch.randperm(x.shape[0])]
            
        # Convolution layers.
        for conv2d in self.X_convolutions:
            x = conv2d(x)
            x = self.maxpool2d(x)
            x = self.activation(x)
            
        for conv2d in self.Y_convolutions:
            y = conv2d(y)
            y = self.maxpool2d(y)
            y = self.activation(y)
            
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        
        layer = torch.cat((x, y), dim=1)
        
        # First dense layer.
        layer = self.linear_1(layer)
        layer = self.activation(layer)
        
        # Second dense layer.
        layer = self.linear_2(layer)
        layer = self.activation(layer)
        
        # Third dense layer.
        layer = self.linear_3(layer)
        
        return layer