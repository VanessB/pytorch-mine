import math
import torch


class DVRSecondTerm(torch.autograd.Function):
    """
    Класс, реализующий вычисление второго члена DVRLoss с несмещённым градиентом.
    """
    
    # Регуляризация знаменателя.
    EPS = 1e-6
    
    @staticmethod
    def forward(ctx, T_marginal: torch.tensor, moving_average: float,
                precomputed_logmeanexp_T_marginal: torch.tensor=None) -> torch.tensor:
        """
        Прямое прохождение.
        
        Параметры
        ---------
        T_marginal : torch.tensor
             Значение статистики на батче, полученном из произведения маргинальных распределений.
        moving_average : float
             Текущее значение движущегося среднего матожидания exp(T),
             используемого для обратного распространения ошибки.
        precomputed_logmeanexp_T_marginal : torch.tensor, optional
             Предвычисленное значение второго члена DVRLoss.
        """
        
        # Потребуется для расчёта градиента.
        # Для прямого прохода moving_average не требуется.
        ctx.save_for_backward(T_marginal, moving_average)
        
        if precomputed_logmeanexp_T_marginal is None:
            return torch.logsumexp(T_marginal, dim=0) - math.log(T_marginal.shape[0])
        else:
            return precomputed_logmeanexp_T_marginal


    @staticmethod
    def backward(ctx, grad_output: torch.tensor) -> torch.tensor:
        """
        Обратное прохождение.
        
        Параметры
        ---------
        grad_output : torch.tensor
            Градиент по выходному значению.
        """

        T_marginal, moving_average = ctx.saved_tensors
        grad = grad_output * T_marginal.exp().detach() / (moving_average * T_marginal.shape[0] + DVRSecondTerm.EPS)
        return grad, None#, None


class DVRLoss(torch.nn.Module):
    """
    Класс, реализующий функцию потерь через расстояние Кульбака-Лейблера в представлении Донскера-Вардана.
    
    Атрибуты
    --------
    biased : bool
        Использоать ли смещённую оценку градиента?
    alpha : float
        Коэффициент для расчёта движущегося среднего.
    moving_average : float
        Движущееся среднее матожидания exp(T).
    
    Методы
    ------
    forward
        Прямое прохождение.
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
        Прямое прохождение.
        
        Параметры
        ---------
        T_joined : torch.tensor
            Значение статистики на батче, полученном из совместного распределения.
        T_marginal : torch.tensor
            Значение статистики на батче, полученном из произведения маргинальных распределений.
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
        Обновление бегущего среднего.
        
        Параметры
        ---------
        current_average : float
            Среднее, полученное по очередному батчу.
        """
        
        if self.moving_average is None:
            self.moving_average = current_average
        else:
            self.moving_average = self.alpha * current_average + (1.0 - self.alpha) * self.moving_average