import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin


class MaskedLinearRegression(nn.Module):
    def __init__(self, mask: torch.Tensor, use_bias: bool = False):
        super().__init__()

        self.register_buffer("mask", mask)

        weights = nn.Parameter(torch.rand(mask.shape))
        self.register_parameter("weights", weights)
        
        self._use_bias = use_bias
        if use_bias:
            bias = nn.Parameter(torch.rand(mask.shape[1]))
            self.register_parameter("bias", bias)

    def forward(self, x: torch.Tensor):
        masked_weights = self.weights * self.mask
        out = x @ masked_weights 
        if self._use_bias:
            out += self.bias
        return out

    def get_masked_weight_matrix(self) -> torch.Tensor:
        return self.weights * self.mask
    

class MaskedRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, mask: np.ndarray, use_bias=False, ridge_lambda=0.001, epochs=1000, lr=0.001):
        self.mask = mask
        self.ridge_lambda = ridge_lambda
        self.epochs = epochs
        self.lr = lr
        self.use_bias = use_bias
        self._param_names = ["use_bias", "mask", "ridge_lambda", "epochs", "lr"]

        self._init_model()

    def _init_model(self):
        self._model = MaskedLinearRegression(torch.Tensor(self.mask), self.use_bias)
        
    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        assert (self.mask.shape == (input_dim, output_dim))
        assert (X.shape[0] == y.shape[0])

        X = torch.Tensor(X)
        y = torch.Tensor(y)

        self._init_model()
        self._model.train()
        mse_loss_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=0)
        
        progress_bar = tqdm(range(self.epochs))
        for epoch in progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            y_pred = self._model(X)

            mse_loss = mse_loss_criterion(y_pred, y)
            ridge_loss = self.ridge_lambda * self._model.get_parameter("weights").norm(2)
            loss = mse_loss + ridge_loss

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        self.W_ = self._model.get_masked_weight_matrix().detach().numpy()

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        input_dim = self.mask.shape[0]
        assert (X.shape[1] == input_dim)

        X = torch.Tensor(X)

        self._model.eval()
        y_pred = self._model(X)
        return y_pred.detach().numpy()

    def get_params(self, deep: bool = True) -> dict:
        params = { param_name: getattr(self, param_name) for param_name in self._param_names }
        params.update(super().get_params(deep))
        return params
    
    def set_params(self, **params):
        for param_name in self._param_names:
            value = params.pop(param_name, None)
            if value:
                setattr(self, param_name, value)
        super().set_params(**params)
        return self

