import math
import torch
import torch.nn as nn

class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        elif init == "jax":
            self._jax_init()
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _jax_init(self):
        input_size = self.weight.shape[-1]
        std = math.sqrt(1 / input_size)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)


class BaseNMREncoder(nn.Module):
    def __init__(self, n_size=4000, n_token=100, n_layers=4, d_hidden=768, d_out=768, activation='LeakyReLU', bias=True, **kwargs):
        super().__init__()
        assert n_size % n_token == 0
        self.n_size = n_size
        self.n_token = n_token
        self.n_grouped = self.n_size // self.n_token
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.bias = bias
        self.activation = getattr(torch.nn, activation)()


class NMRMLPEncoder(BaseNMREncoder):
    def __init__(self, final_init="final", **kwargs):
        super().__init__(**kwargs)

        layers = [Linear(self.n_grouped, self.d_hidden, self.bias), self.activation]
        for _ in range(self.n_layers):
            layers += [Linear(self.d_hidden, self.d_hidden, self.bias), self.activation]
        layers.append(Linear(self.d_hidden, self.d_out, self.bias, init=final_init))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.reshape([x.shape[0], self.n_token, self.n_grouped])
        return self.main(x)

