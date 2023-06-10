if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from functools import partial
from einops import rearrange, repeat
import opt_einsum as oe

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from src.models.sequence.ss.kernel import SSKernel
from src.models.nn import LinearActivation, Activation, DropoutNd, Normalization

class H3(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=None,
            channels=1,
            bidirectional=False,
            # Arguments for position-wise feedforward components
            layer1='none', # 'none' | 'conv' | 'ssm'
            act1='id',
            glu1='id',
            mult1='id',
            act2='id',
            glu2='id',
            mult2='id',
            inner_linear=False,
            inner_ln=False,
            postact='glu',
            initializer=None,
            weight_norm=False,
            dropout=0.0, tie_dropout=False,
            transposed=True,
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF ('id' for no activation, None to remove FF layer)
        initializer: initializer on FF
        weight_norm: weight normalization on FF
        hyper_act: use a "hypernetwork" multiplication (experimental)
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        gate: add gated activation (GSS)
        bottleneck: reduce SSM dimension (GSS)
        shift: experimental option, shouldn't affect results

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed


        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2

        # Input projection
        self.input_projection = LinearActivation(self.H, self.H*3, transposed=True)

        # SSM Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)

        # First layer
        self.layer1 = layer1
        if layer1 == 'none':
            pass
            # self.layer1 = nn.Identity()
        elif layer1 == 'conv':
            self.kernel1 = nn.Conv1d(
                self.H, self.H, 3,
                padding=3 - 1, groups=self.H
            )
        elif layer1 == 'ssm':
            self.kernel1 = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)
        else: raise NotImplementedError

        # Pointwise
        self.act1 = Activation(act1)
        self.act2 = Activation(act2)
        self.glu1 = Activation(glu1)
        self.glu2 = Activation(glu2)
        self.mult1 = Activation(mult1)
        self.mult2 = Activation(mult2)

        if inner_linear:
            self.inner_linear = LinearActivation(
                self.H*self.channels,
                self.H,
                transposed=True,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )
        else:
            self.inner_linear = nn.Identity()

        if inner_ln:
            self.inner_ln = Normalization(self.H, transposed=True)
        else:
            self.inner_ln = nn.Identity()

        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.H*self.channels,
                self.H,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )



    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
            u = u * mask

        # Input projection
        u = self.input_projection(u)
        u1, u2, u3 = rearrange(u, 'b (z h) l -> z b h l', z=3)

        # First layer
        if self.layer1 == 'none':
            pass
        elif self.layer1 == 'conv':
            u1 = self.kernel1(u1)[..., :L]
        elif self.layer1 == 'ssm':
            L_kernel = L if self.L is None else min(L, round(self.L / rate))
            k, _ = self.kernel1(L=L_kernel, rate=rate, state=None) # (C H L) (B C H L)
            if self.bidirectional:
                k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
                k = F.pad(k0, (0, L)) \
                        + F.pad(k1.flip(-1), (L, 0))
            k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
            u_f = torch.fft.rfft(u1, n=L_kernel+L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f)
            y = torch.fft.irfft(y_f, n=L_kernel+L)[..., :L] # (B C H L)
            assert y.size(1) == 1
            u1 = y.squeeze(1)
        u1 = self.act1(u1)
        u2 = self.glu1(u2) * u1
        u2 = self.mult1(u2)
        u2 = self.inner_linear(u2)
        u2 = self.inner_ln(u2)


        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))

        k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
        u_f = torch.fft.rfft(u2, n=L_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=L_kernel+L)[..., :L] # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u2, self.D)
        assert y.size(1) == 1
        u2 = y.squeeze(1)
        u2 = self.act2(u2)
        u3 = self.glu2(u3) * u2
        u3 = self.mult2(u3)
        y = u3


        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None



        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y, next_state

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)
        if self.layer1 == 'ssm':
            self.kernel1._setup_step(**kwargs)

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        u = self.input_projection(u.unsqueeze(-1)).squeeze(-1)
        u1, u2, u3 = rearrange(u, 'b (z h) -> z b h', z=3)

        if self.layer1 == 'ssm':
            state1, state = state

        if self.layer1 == 'none':
            pass
        elif self.layer1 == 'ssm':
            y, next_state1 = self.kernel1.step(u1, state1)
            assert y.size(1) == 1
            u1 = y.squeeze(1) # (B H)
        else: raise NotImplementedError

        u1 = self.act1(u1)
        u2 = self.glu1(u2) * u1
        u2 = self.mult1(u2)
        u2 = u2[:, None, :]
        u2 = self.inner_linear(u2)
        u2 = self.inner_ln(u2)
        u2 = u2[:, 0, :]


        y, next_state = self.kernel.step(u2, state) # (B C H)
        y = y + u2.unsqueeze(-2) * self.D
        assert y.size(1) == 1
        u2 = rearrange(y, 'b c h -> b (c h)')
        u2 = self.act2(u2)
        u3 = self.glu2(u3) * u2
        u3 = self.mult2(u3)
        y = u3
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)

        if self.layer1 == 'ssm':
            next_state = (next_state1, next_state)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        state = self.kernel.default_state(*batch_shape)
        if self.layer1 == 'ssm':
            state1 = self.kernel1.default_state(*batch_shape)
            return (state1, state)
        else:
            return state

    @property
    def d_state(self):
        return self.H * self.N

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


def test_state(**kwargs):
    # B = 1
    # H = 64
    # N = 64
    # L = 1024
    B = 2
    H = 3
    N = 4
    L = 8
    s4 = H3(H, d_state=N, l_max=L, **kwargs)
    s4.to(device)
    s4.eval()
    # for module in s4.modules():
        # if hasattr(module, 'setup_step'): module.setup_step()
    s4.setup_step()

    u = torch.ones(B, H, L).to(device)
    y, _ = s4(u)
    print("output:\n", y, y.shape)
    # print("final state:\n", final_state, final_state.shape)

    # Use Stepping
    s4.setup_step()
    state = s4.default_state(B)
    # state = initial_state.clone()
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = s4.step(u_, state=state)
        ys.append(y_)
    ys = torch.stack(ys, dim=-1)
    print("step outputs:\n", ys)
    print("step final state:\n", state)


if __name__ == '__main__':
    from benchmark import utils
    torch.manual_seed(42)

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    # test_state(random_init=True, mode='nplr', measure='legt', rank=2, channels=2)
    # test_state(random_init=False, mode='diag', measure='legs', rank=1)
    # test_state(measure='legs', rank=1)
    test_state(layer1='ssm')
