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
from src.models.nn import LinearActivation, Activation, DropoutNd

class S4(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=None,
            channels=1,
            bidirectional=False,
            # Arguments for position-wise feedforward components
            activation='gelu',
            postact='glu',
            initializer=None,
            weight_norm=False,
            hyper_act=None,
            dropout=0.0, tie_dropout=False,
            bottleneck=None,
            gate=None,
            gate_act=None,
            transposed=True,
            verbose=False,
            shift=False,
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
        self.shift = shift

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.H = self.H // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.H,
                transposed=self.transposed,
                initializer=initializer,
                # activation=activation,
                # activate=True,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=self.transposed,
                initializer=initializer,
                activation=gate_act,
                activate=True,
                weight_norm=weight_norm,
            )
            if (self.bottleneck in [1, None]) and self.channels == gate:
                self.output_gate = nn.Identity()
            else:
                self.output_gate = LinearActivation(
                    self.H*self.channels,
                    self.d_model * gate,
                    transposed=self.transposed,
                    initializer=initializer,
                    activation=None,
                    activate=False,
                    weight_norm=weight_norm,
                )

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2


        # SSM Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model * (1 if gate is None else gate),
                self.d_model,
                # self.H*self.channels,
                # self.d_model*(1 if self.gate is None else self.gate),
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

        if not self.transposed: u = u.transpose(-1, -2)
        if self.gate is not None:
            v = self.input_gate(u)
        if self.bottleneck is not None:
            u = self.input_linear(u)
        if not self.transposed: u = u.transpose(-1, -2)

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        if self.shift:
            # Try flip and pad to correct for potential off-by-one
            k_f = torch.fft.rfft(F.pad(k.flip(-1), (L, 0)), n=2*L) # (C H L)
            u_f = torch.fft.rfft(F.pad(u.flip(-1), (L, 0)), n=2*L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
            y = torch.fft.irfft(y_f, n=L_kernel+L)[..., L:].flip(-1) # (B C H L)
        else:
            k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
            u_f = torch.fft.rfft(u, n=L_kernel+L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f)
            y = torch.fft.irfft(y_f, n=L_kernel+L)[..., :L] # (B C H L)



        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None


        # y = self.activation(y)

        # Optional "hyper-network" multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y
        y = self.dropout(y)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.transposed: y = y.transpose(-1, -2)


        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)

        return y, next_state

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.H * self.N

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


def test_state(random_init=False, **kwargs):
    # B = 1
    # H = 64
    # N = 64
    # L = 1024
    B = 2
    H = 3
    N = 4
    L = 8
    s4 = S4(H, d_state=N, l_max=L, **kwargs)
    s4.to(device)
    s4.eval()
    # for module in s4.modules():
        # if hasattr(module, 'setup_step'): module.setup_step()
    s4.setup_step()

    u = torch.ones(B, H, L).to(device)
    initial_state = s4.default_state(B)
    if random_init:
        if initial_state.size(-1) == N:
            initial_state = initial_state[..., :N//2]
            initial_state = torch.randn_like(initial_state)
            initial_state = torch.cat([initial_state, initial_state.conj()], dim=-1)
        else:
            initial_state = torch.randn_like(initial_state)

    state = initial_state.clone()
    y, final_state = s4(u, state=state)
    print("output:\n", y, y.shape)
    print("final state:\n", final_state, final_state.shape)

    # Use Stepping
    s4.setup_step()
    state = initial_state.clone()
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = s4.step(u_, state=state)
        ys.append(y_)
    ys = torch.stack(ys, dim=-1)
    print("step outputs:\n", ys)
    print("step final state:\n", state)

    # Use Chunking

    chunks = 4
    state = initial_state.clone()
    ys = []
    for u_ in u.chunk(chunks, dim=-1):
        y_, state = s4(u_, state=state)
        ys.append(y_)
    ys = torch.cat(ys, dim=-1)
    print("chunk outputs:\n", ys)
    print("chunk final state:\n", state)
    print("chunk output error:")
    utils.compare_outputs(y, ys)
    print("chunk final state error:")
    utils.compare_outputs(final_state, state)


if __name__ == '__main__':
    from benchmark import utils
    torch.manual_seed(42)

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    # test_state(random_init=True, mode='nplr', measure='legt', rank=2, channels=2)
    # test_state(random_init=False, mode='diag', measure='legs', rank=1)
    test_state(random_init=True, mode='diag', measure='legs', rank=1, disc='zoh', channels=3)
