import torch


class NeuralT(torch.nn.Module):
    """
    Базовый класс для нейронной сети, вычисляющей статистику T.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    
    def get_mutual_information(self, dataloader, loss: callable, device, permute: bool=True) -> float:
        """
        Оценка взаимной информации.
        
        Параметры
        ---------
        dataloader
            Загрузщик данных. Должен выдавать тройки (x,y,z).
        loss : callable
            Лосс для вариационной оценки взаимной информации.
        device
            Устройство, на котором требуется проводить вычисления.
        permute : bool, optional
            Перемешивать ли каждый батч заново для маргинального распределения.
        """
        
        # Выход из режима обучения.
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
                
        # Возвращение модели к исходному режиму.
        self.train(was_in_training)
        
        return mutual_information


class BasicDenseT(NeuralT):
    def __init__(self, X_dim: int, Y_dim: int) -> None:
        super().__init__()
        
        self.dropout = torch.nn.Dropout(0.1)
        
        self.linear_1 = torch.nn.Linear(X_dim + Y_dim, 100)
        self.linear_2 = torch.nn.Linear(100, 100)
        self.linear_3 = torch.nn.Linear(100, 1)
        
        self.activation = torch.nn.LeakyReLU()
        
        
    def forward(self, x: torch.tensor, y: torch.tensor, permute: bool=False) -> torch.tensor:
        if permute:
            x = x[torch.randperm(x.shape[0])]
        
        layer = torch.cat((x, y), dim=1)
        
        # Первый слой.
        layer = self.linear_1(layer)
        layer = self.activation(layer)
        
        # Второй слой.
        layer = self.linear_2(layer)
        layer = self.activation(layer)
        
        # Третий слой.
        layer = self.linear_3(layer)
        
        return layer