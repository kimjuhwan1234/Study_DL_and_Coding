import math
import torch
import weakref
import numbers
import warnings
import torch.nn as nn
from torch.nn import init
from torch import _VF, Tensor
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from typing import List, Optional, overload, Tuple

__all__ = [
    "RNNBase",
    "RNN",
]

_rnn_impls = {
    "RNN_TANH": _VF.rnn_tanh,
    "RNN_RELU": _VF.rnn_relu,
}


def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


class RNNBase(nn.Module):
    r"""Base class for RNN modules (RNN, LSTM, GRU).

    Implements aspects of RNNs shared by the RNN, LSTM, and GRU classes, such as module initialization
    and utility methods for parameter storage management.

    .. note::
        The forward method is not implemented by the RNNBase class.

    .. note::
        LSTM and GRU classes override some methods implemented by RNNBase.
    """

    __constants__ = [
        "mode",
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
        "proj_size",
    ]
    __jit_unused_properties__ = ["all_weights"]

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int

    def __init__(
            self,
            mode: str,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            proj_size: int = 0,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self._flat_weight_refs: List[Optional[weakref.ReferenceType[Parameter]]] = []
        num_directions = 2 if bidirectional else 1

        if (
                not isinstance(dropout, numbers.Number)
                or not 0 <= dropout <= 1
                or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                f"num_layers greater than 1, but got dropout={dropout} and "
                f"num_layers={num_layers}"
            )

        if not isinstance(hidden_size, int):
            raise TypeError(
                f"hidden_size should be of type int, got: {type(hidden_size).__name__}"
            )
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        if proj_size < 0:
            raise ValueError(
                "proj_size should be a positive integer or zero to disable projections"
            )
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        elif mode == "RNN_TANH":
            gate_size = hidden_size
        elif mode == "RNN_RELU":
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )

                w_ih = Parameter(
                    torch.empty((gate_size, layer_input_size), **factory_kwargs)
                )
                w_hh = Parameter(
                    torch.empty((gate_size, real_hidden_size), **factory_kwargs)
                )
                b_ih = Parameter(torch.empty(gate_size, **factory_kwargs))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.empty(gate_size, **factory_kwargs))
                layer_params: Tuple[Tensor, ...] = ()
                if self.proj_size == 0:
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)
                else:
                    w_hr = Parameter(
                        torch.empty((proj_size, hidden_size), **factory_kwargs)
                    )
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else:
                        layer_params = (w_ih, w_hh, w_hr)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                if self.proj_size > 0:
                    param_names += ["weight_hr_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._init_flat_weights()

        self.reset_parameters()

    def _init_flat_weights(self):
        self._flat_weights = [
            getattr(self, wn) if hasattr(self, wn) else None
            for wn in self._flat_weights_names
        ]
        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]
        self.flatten_parameters()

    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super().__setattr__(attr, value)

    def flatten_parameters(self) -> None:
        """Reset parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]  # type: ignore[union-attr]
        dtype = first_fw.dtype  # type: ignore[union-attr]
        for fw in self._flat_weights:
            if (
                    not isinstance(fw, Tensor)
                    or not (fw.dtype == dtype)
                    or not fw.is_cuda
                    or not torch.backends.cudnn.is_acceptable(fw)
            ):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = {
            p.data_ptr() for p in self._flat_weights  # type: ignore[union-attr]
        }
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights,  # type: ignore[arg-type]
                        num_weights,
                        self.input_size,
                        rnn.get_cudnn_mode(self.mode),
                        self.hidden_size,
                        self.proj_size,
                        self.num_layers,
                        self.batch_first,
                        bool(self.bidirectional),
                    )

    def _apply(self, fn, recurse=True):
        self._flat_weight_refs = []
        ret = super()._apply(fn, recurse)

        # Resets _flat_weights
        # Note: be v. careful before removing this, as 3rd party device types
        # likely rely on this behavior to properly .to() modules like LSTM.
        self._init_flat_weights()

        return ret

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        if not torch.jit.is_scripting():
            if (
                    input.dtype != self._flat_weights[0].dtype  # type: ignore[union-attr]
                    and not torch._C._is_any_autocast_enabled()
            ):
                raise ValueError(
                    f"input must have the type {self._flat_weights[0].dtype}, got type {input.dtype}"
                    # type: ignore[union-attr]
                )
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f"input must have {expected_input_dim} dimensions, got {input.dim()}"
            )
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}"
            )

    def get_expected_hidden_size(
            self, input: Tensor, batch_sizes: Optional[Tensor]
    ) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.proj_size,
            )
        else:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.hidden_size,
            )
        return expected_hidden_size

    def check_hidden_size(
            self,
            hx: Tensor,
            expected_hidden_size: Tuple[int, int, int],
            msg: str = "Expected hidden size {}, got {}",
    ) -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def _weights_have_changed(self):
        # Returns True if the weight tensors have changed since the last forward pass.
        # This is the case when used with torch.func.functional_call(), for example.
        weights_changed = False
        for ref, name in zip(self._flat_weight_refs, self._flat_weights_names):
            weight = getattr(self, name) if hasattr(self, name) else None
            if weight is not None and ref is not None and ref() is not weight:
                weights_changed = True
                break
        return weights_changed

    def check_forward_args(
            self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]
    ):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if self.proj_size != 0:
            s += ", proj_size={proj_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def _update_flat_weights(self):
        if not torch.jit.is_scripting():
            if self._weights_have_changed():
                self._init_flat_weights()

    def __getstate__(self):
        # If weights have been changed, update the _flat_weights in __getstate__ here.
        self._update_flat_weights()
        # Don't serialize the weight references.
        state = self.__dict__.copy()
        del state["_flat_weight_refs"]
        return state

    def __setstate__(self, d):
        super().__setstate__(d)
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        # In PyTorch 1.8 we added a proj_size member variable to LSTM.
        # LSTMs that were serialized via torch.save(module) before PyTorch 1.8
        # don't have it, so to preserve compatibility we set proj_size here.
        if "proj_size" not in d:
            self.proj_size = 0

        if not isinstance(self._all_weights[0][0], str):
            num_layers = self.num_layers
            num_directions = 2 if self.bidirectional else 1
            self._flat_weights_names = []
            self._all_weights = []
            for layer in range(num_layers):
                for direction in range(num_directions):
                    suffix = "_reverse" if direction == 1 else ""
                    weights = [
                        "weight_ih_l{}{}",
                        "weight_hh_l{}{}",
                        "bias_ih_l{}{}",
                        "bias_hh_l{}{}",
                        "weight_hr_l{}{}",
                    ]
                    weights = [x.format(layer, suffix) for x in weights]
                    if self.bias:
                        if self.proj_size > 0:
                            self._all_weights += [weights]
                            self._flat_weights_names.extend(weights)
                        else:
                            self._all_weights += [weights[:4]]
                            self._flat_weights_names.extend(weights[:4])
                    else:
                        if self.proj_size > 0:
                            self._all_weights += [weights[:2]] + [weights[-1:]]
                            self._flat_weights_names.extend(
                                weights[:2] + [weights[-1:]]
                            )
                        else:
                            self._all_weights += [weights[:2]]
                            self._flat_weights_names.extend(weights[:2])
            self._flat_weights = [
                getattr(self, wn) if hasattr(self, wn) else None
                for wn in self._flat_weights_names
            ]

        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]

    @property
    def all_weights(self) -> List[List[Parameter]]:
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        # Need to copy these caches, otherwise the replica will share the same
        # flat weights list.
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        return replica


class RNN(RNNBase):
    @overload
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            nonlinearity: str = "tanh",
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        ...

    @overload
    def __init__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        if len(args) > 3:
            self.nonlinearity = args[3]
            args = args[:3] + args[4:]
        else:
            self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(
                f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        super().__init__(mode, *args, **kwargs)

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
            self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
            self, input: PackedSequence, hx: Optional[Tensor] = None
    ) -> Tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):  # noqa: F811
        self._update_flat_weights()

        num_directions = 2 if self.bidirectional else 1
        orig_input = input

        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            # script() is unhappy when max_batch_size is different type in cond branches, so we duplicate
            if hx is None:
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"RNN: Expected input to be 2D or 3D, got {input.dim()}D tensor instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                hx = torch.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        self.check_forward_args(input, hx, batch_sizes)
        assert self.mode == "RNN_TANH" or self.mode == "RNN_RELU"
        if batch_sizes is None:
            if self.mode == "RNN_TANH":
                result = _VF.rnn_tanh(
                    input,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
            else:
                result = _VF.rnn_relu(
                    input,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                    self.batch_first,
                )
        else:
            if self.mode == "RNN_TANH":
                result = _VF.rnn_tanh(
                    input,
                    batch_sizes,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )
            else:
                result = _VF.rnn_relu(
                    input,
                    batch_sizes,
                    hx,
                    self._flat_weights,  # type: ignore[arg-type]
                    self.bias,
                    self.num_layers,
                    self.dropout,
                    self.training,
                    self.bidirectional,
                )

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:  # type: ignore[possibly-undefined]
            output = output.squeeze(batch_dim)  # type: ignore[possibly-undefined]
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, unsorted_indices)
