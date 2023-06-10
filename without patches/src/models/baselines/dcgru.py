# Code taken from https://github.com/tsy935/eeg-gnn-ssl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import numpy as np
import torch
import torch.nn as nn


class DiffusionGraphConv(nn.Module):
    def __init__(
        self,
        num_supports,
        input_dim,
        hid_dim,
        num_nodes,
        max_diffusion_step,
        output_dim,
        bias_start=0.0,
        filter_type="laplacian",
    ):
        """
        Diffusion graph convolution
        Args:
            num_supports: number of supports, 1 for 'laplacian' filter and 2
                for 'dual_random_walk'
            input_dim: input feature dim
            hid_dim: hidden units
            num_nodes: number of nodes in graph
            max_diffusion_step: maximum diffusion step
            output_dim: output feature dim
            filter_type: 'laplacian' for undirected graph, and 'dual_random_walk'
                for directed graph
        """
        super(DiffusionGraphConv, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(
            torch.FloatTensor(size=(self._input_size * num_matrices, output_dim))
        )
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, supports, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes,
        # input_dim/hidden_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # (batch, num_nodes, input_dim+hidden_dim)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size

        x0 = inputs_and_state  # (batch, num_nodes, input_dim+hidden_dim)
        # (batch, 1, num_nodes, input_dim+hidden_dim)
        x = torch.unsqueeze(x0, dim=1)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                # (batch, num_nodes, input_dim+hidden_dim)
                x1 = torch.matmul(support, x0)
                # (batch, ?, num_nodes, input_dim+hidden_dim)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    # (batch, num_nodes, input_dim+hidden_dim)
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(
                        x, x2
                    )  # (batch, ?, num_nodes, input_dim+hidden_dim)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * self._max_diffusion_step + 1  # Adds for x itself
        # (batch, num_nodes, num_matrices, input_hidden_size)
        x = torch.transpose(x, dim0=1, dim1=2)
        # (batch, num_nodes, input_hidden_size, num_matrices)
        x = torch.transpose(x, dim0=2, dim1=3)
        x = torch.reshape(
            x, shape=[batch_size, self._num_nodes, input_size * num_matrices]
        )
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices]
        )
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
        self,
        input_dim,
        num_units,
        max_diffusion_step,
        num_nodes,
        filter_type="laplacian",
        nonlinearity="tanh",
        use_gc_for_ru=True,
    ):
        """
        Args:
            input_dim: input feature dim
            num_units: number of DCGRU hidden units
            max_diffusion_step: maximum diffusion step
            num_nodes: number of nodes in the graph
            filter_type: 'laplacian' for undirected graph, 'dual_random_walk' for directed graph
            nonlinearity: 'tanh' or 'relu'. Default is 'tanh'
            use_gc_for_ru: decide whether to use graph convolution inside rnn. Default True
        """
        super(DCGRUCell, self).__init__()
        self._activation = torch.tanh if nonlinearity == "tanh" else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        if filter_type == "laplacian":  # ChebNet graph conv
            self._num_supports = 1
        elif filter_type == "random_walk":  # Forward random walk
            self._num_supports = 1
        elif filter_type == "dual_random_walk":  # Bidirectional random walk
            self._num_supports = 2
        else:
            self._num_supports = 1

        self.dconv_gate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2,
            filter_type=filter_type,
        )
        self.dconv_candidate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type,
        )

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def forward(self, supports, inputs, state):
        """
        Args:
            inputs: (B, num_nodes * input_dim)
            state: (B, num_nodes * num_units)
        Returns:
            output: (B, num_nodes * output_dim)
            state: (B, num_nodes * num_units)
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(fn(supports, inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, split_size_or_sections=int(output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        # batch_size, self._num_nodes * output_size
        c = self.dconv_candidate(supports, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


class DCRNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        max_diffusion_step,
        hid_dim,
        num_nodes,
        num_rnn_layers,
        dcgru_activation=None,
        filter_type="laplacian",
        device=None,
    ):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type,
            )
        )

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type,
                )
            )
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state
                )
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device
            )  # (seq_len, batch_size, num_nodes * rnn_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device
        )  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        max_diffusion_step,
        num_nodes,
        hid_dim,
        output_dim,
        num_rnn_layers,
        dcgru_activation=None,
        filter_type="laplacian",
        # device=None,
        dropout=0.0,
    ):
        super(DCGRUDecoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        # self._device = device
        self.dropout = dropout

        cell = DCGRUCell(
            input_dim=hid_dim,
            num_units=hid_dim,
            max_diffusion_step=max_diffusion_step,
            num_nodes=num_nodes,
            nonlinearity=dcgru_activation,
            filter_type=filter_type,
        )

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type,
            )
        )
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            decoding_cells.append(cell)

        self.decoding_cells = nn.ModuleList(decoding_cells)
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)  # dropout before projection layer

    def forward(
        self, inputs, initial_hidden_state, supports, teacher_forcing_ratio=None
    ):
        """
        Args:
            inputs: shape (seq_len, batch_size, num_nodes, output_dim)
            initial_hidden_state: the last hidden state of the encoder, shape (num_layers, batch, num_nodes * rnn_units)
            supports: list of supports from laplacian or dual_random_walk filters
            teacher_forcing_ratio: ratio for teacher forcing
        Returns:
            outputs: shape (seq_len, batch_size, num_nodes * output_dim)
        """
        seq_length, batch_size, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.output_dim)
        )  # .to(self._device)

        # tensor to store decoder outputs
        outputs = torch.zeros(
            seq_length, batch_size, self.num_nodes * self.output_dim
        )  # .to(self._device)

        current_input = go_symbol  # (batch_size, num_nodes * input_dim)
        for t in range(seq_length):
            next_input_hidden_state = []
            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    supports, current_input, hidden_state
                )
                current_input = output
                next_input_hidden_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)

            projected = self.projection_layer(
                self.dropout(output.reshape(batch_size, self.num_nodes, -1))
            )
            projected = projected.reshape(batch_size, self.num_nodes * self.output_dim)
            outputs[t] = projected

            if teacher_forcing_ratio is not None:
                teacher_force = random.random() < teacher_forcing_ratio  # a bool value
                current_input = inputs[t] if teacher_force else projected
            else:
                current_input = projected

        return outputs


########## Model for seizure classification/detection ##########
class DCRNNModel_classification(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        num_rnn_layers,
        rnn_units,
        enc_input_dim,
        max_diffusion_step,
        dcgru_activation,
        filter_type,
        num_classes,
        dropout,
    ):
        super(DCRNNModel_classification, self).__init__()

        self.d_input = d_input
        self.d_output = d_output

        self.num_nodes = d_input
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        # self._device = device
        self.num_classes = num_classes

        self.encoder = DCRNNEncoder(
            input_dim=enc_input_dim,
            max_diffusion_step=max_diffusion_step,
            hid_dim=rnn_units,
            num_nodes=self.num_nodes,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation=dcgru_activation,
            filter_type=filter_type,
        )

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, support, *args, **kwargs):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """

        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]
        input_seq = input_seq.view(batch_size, max_seq_len, self.num_nodes, -1)

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).to(
            input_seq.device
        )  # .to(self._device)

        # last hidden state of the encoder is the context
        # (max_seq_len, batch, rnn_units*num_nodes)
        _, final_hidden = self.encoder(
            input_seq, init_hidden_state, support.unsqueeze(0)
        )
        # (batch_size, max_seq_len, rnn_units*num_nodes)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)

        # extract last relevant output
        # last_out = utils.last_relevant_pytorch(
        #     output, seq_lengths, batch_first=True
        # )  # (batch_size, rnn_units*num_nodes)
        last_out = output[:, -1, :]
        # (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        # last_out = last_out.to(self._device)

        # # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits, None
        # return last_out


########## Model for seizure classification/detection ##########


########## Model for next time prediction ##########
class DCRNNModel_nextTimePred(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        num_rnn_layers,
        rnn_units,
        enc_input_dim,
        max_diffusion_step,
        dcgru_activation,
        filter_type,
        dropout,
        use_curriculum_learning=False,
    ):
        super(DCRNNModel_nextTimePred, self).__init__()

        self.d_input = d_input
        self.d_output = d_output

        num_nodes = d_input
        # num_rnn_layers = num_rnn_layers
        # rnn_units = rnn_units
        # enc_input_dim = enc_input_dim
        dec_input_dim = d_output  # args.output_dim
        output_dim = d_output  # args.output_dim
        # max_diffusion_step = max_diffusion_step

        self.num_nodes = d_input
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        # self._device = device
        self.output_dim = output_dim
        # self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(use_curriculum_learning)

        self.encoder = DCRNNEncoder(
            input_dim=enc_input_dim,
            max_diffusion_step=max_diffusion_step,
            hid_dim=rnn_units,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation=dcgru_activation,
            filter_type=filter_type,
        )
        self.decoder = DCGRUDecoder(
            input_dim=dec_input_dim,
            max_diffusion_step=max_diffusion_step,
            num_nodes=num_nodes,
            hid_dim=rnn_units,
            output_dim=output_dim,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation=dcgru_activation,
            filter_type=filter_type,
            # device=device,
            dropout=dropout,
        )

    def forward(
        self,
        encoder_inputs,
        decoder_inputs,
        supports,
        batches_seen=None,
        *args,
        **kwargs
    ):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            encoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            supports: list of supports from laplacian or dual_random_walk filters
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, max_seq_len = encoder_inputs.shape[0], encoder_inputs.shape[1]
        encoder_inputs = encoder_inputs.view(
            batch_size, max_seq_len, self.num_nodes, -1
        )

        batch_size, max_seq_len = decoder_inputs.shape[0], decoder_inputs.shape[1]
        decoder_inputs = decoder_inputs.view(
            batch_size, max_seq_len, self.num_nodes, -1
        )
        breakpoint()

        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).cuda()

        # encoder
        # (num_layers, batch, rnn_units*num_nodes)
        encoder_hidden_state, _ = self.encoder(
            encoder_inputs, init_hidden_state, supports
        )

        # decoder
        # if (
        #     self.training
        #     and self.use_curriculum_learning
        #     and (batches_seen is not None)
        # ):
        #     teacher_forcing_ratio = utils.compute_sampling_threshold(
        #         self.cl_decay_steps, batches_seen
        #     )
        # else:
        #     teacher_forcing_ratio = None
        teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )  # (seq_len, batch_size, num_nodes * output_dim)
        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs


########## Model for next time prediction ##########
