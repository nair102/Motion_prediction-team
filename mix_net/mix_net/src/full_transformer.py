    
import json
import torch
import torch.nn as nn
from collections import OrderedDict
import math

class newMixNet2(nn.Module):
    """Neural Network to predict the future trajectory of a vehicle based on its history.
    It predicts mixing weights for mixing the boundaries, the centerline and the raceline.
    Also, it predicts section wise constant accelerations and an initial velocity to
    compute the velocity profile from.
    """

    def __init__(self, params):
        """Initializes a MixNet object."""
        super(newMixNet2, self).__init__()

        self._params = params

        # Input embedding layer:
        self._ip_emb = nn.Linear(2, params["encoder"]["in_size"])

        # History encoder Transformer with batch_first=True:
        Hist_layer = nn.TransformerEncoderLayer(
            d_model=params["encoder"]["hidden_size"],
            nhead=params["encoder"]["nhead"],
            dim_feedforward=params["encoder"]["dim_feedforward"],
            dropout=params["encoder"]["dropout"],
            activation='relu',
            batch_first=True    # <== Changed to batch-first
        )
        self._enc_hist = nn.TransformerEncoder(
            Hist_layer,
            num_layers=params["encoder"]["num_layers"]   # Changed for consistency with others
        )

        # Left Boundary encoder Transformer with batch_first=True:
        left_encoder_layer = nn.TransformerEncoderLayer(
            d_model=params["encoder"]["hidden_size"],
            nhead=params["encoder"]["nhead"],
            dim_feedforward=params["encoder"]["dim_feedforward"],
            dropout=params["encoder"]["dropout"],
            activation='relu',
            batch_first=True    # <== Changed to batch-first
        )
        self._enc_left_bound = nn.TransformerEncoder(
            left_encoder_layer,
            num_layers=params["encoder"]["num_layers"]
        )

        # Right Boundary encoder Transformer with batch_first=True:
        right_encoder_layer = nn.TransformerEncoderLayer(
            d_model=params["encoder"]["hidden_size"],
            nhead=params["encoder"]["nhead"],
            dim_feedforward=params["encoder"]["dim_feedforward"],
            dropout=params["encoder"]["dropout"],
            activation='relu',
            batch_first=True    # <== Changed to batch-first
        )
        self._enc_right_bound = nn.TransformerEncoder(
            right_encoder_layer,
            num_layers=params["encoder"]["num_layers"]
        )
        
        # Linear stack that outputs the path mixture ratios:
        self._mix_out_layers = self._get_linear_stack(
            in_size=params["encoder"]["hidden_size"] * 3,
            hidden_sizes=params["mixer_linear_stack"]["hidden_sizes"],
            out_size=params["mixer_linear_stack"]["out_size"],
            name="mix",
        )

        # Linear stack for outputting the initial velocity:
        self._vel_out_layers = self._get_linear_stack(
            in_size=params["encoder"]["hidden_size"],
            hidden_sizes=params["init_vel_linear_stack"]["hidden_sizes"],
            out_size=params["init_vel_linear_stack"]["out_size"],
            name="vel",
        )

        # Dynamic embedder between the encoder and the decoder:
        self._dyn_embedder = nn.Linear(
            params["encoder"]["hidden_size"] * 3, params["acc_decoder"]["in_size"]
        )

        # Acceleration decoder:
        self._acc_decoder = nn.LSTM(
            params["acc_decoder"]["in_size"],
            params["acc_decoder"]["hidden_size"],
            1,
            batch_first=True,
        )

        # Output linear layer of the acceleration decoder:
        self._acc_out_layer = nn.Linear(params["acc_decoder"]["hidden_size"], 1)

        # Set device:
        if params["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA as device for MixNet")
        else:
            self.device = torch.device("cpu")
            print("Using CPU as device for MixNet")

        self.to(self.device)
    
    def _add_positional_encoding(self, x):
        """
        Adds positional encoding to a tensor.
        Expects x of shape [batch_size, seq_len, embedding_dim].
        """
        batch_size, seq_len, embedding_dim = x.size()
        pe = torch.zeros(seq_len, embedding_dim, device=self.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=self.device).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, embedding_dim]
        return x + pe
    
    def _concat_positional_encoding(self, x):
        seq_len, batch_size, embedding_dim = x.size()
        pe = torch.zeros(seq_len, embedding_dim, device=self.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float, device=self.device) *
                             (-math.log(10000.0) / embedding_dim))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Expand pe to match the batch dimension: (seq_len, batch_size, embedding_dim)
        pe = pe.unsqueeze(1).repeat(1, batch_size, 1)
        # The resulting tensor shape will be: (seq_len, batch_size, embedding_dim * 2)
        return torch.cat((x, pe), dim=2)

    def forward(self, hist, left_bound, right_bound):
        """Implements the forward pass of the model.

        Args:
            hist: [tensor with shape=(batch_size, hist_len, 2)]
            left_bound: [tensor with shape=(batch_size, boundary_len, 2)]
            right_bound: [tensor with shape=(batch_size, boundary_len, 2)]

        Returns:
            mix_out: [tensor with shape=(batch_size, out_size)]
            vel_out: [tensor with shape=(batch_size, 1)]
            acc_out: [tensor with shape=(batch_size, num_acc_sections)]
        """

        # Embed the inputs:
        hist_emb = self._ip_emb(hist.to(self.device))
        left_emb = self._ip_emb(left_bound.to(self.device))
        right_emb = self._ip_emb(right_bound.to(self.device))
        
        # With batch_first=True, the embeddings are already [batch, seq_len, feature]
        hist_emb = self._add_positional_encoding(hist_emb)
        left_emb = self._add_positional_encoding(left_emb)
        right_emb = self._add_positional_encoding(right_emb)

        # History encoding using Transformer:
        hist_enc = self._enc_hist(hist_emb)
        # Mean across the sequence dimension (dim=1)
        hist_h = hist_enc.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_size]

        # Left Boundary encoding using Transformer:
        left_enc = self._enc_left_bound(left_emb)
        left_h = left_enc.mean(dim=1, keepdim=True)

        # Right Boundary encoding using Transformer:
        right_enc = self._enc_right_bound(right_emb)
        right_h = right_enc.mean(dim=1, keepdim=True)

        # Concatenate and squeeze encodings:
        enc = torch.cat((hist_h, left_h, right_h), dim=2)  # [batch_size, 1, hidden_size*3]
        enc = enc.squeeze(1)  # [batch_size, hidden_size*3]

        # Path mixture through softmax:
        mix_out = torch.softmax(self._mix_out_layers(enc), dim=1)

        # Initial velocity:
        vel_out = self._vel_out_layers(hist_h.squeeze(1))
        vel_out = torch.sigmoid(vel_out)
        vel_out = vel_out * self._params["init_vel_linear_stack"]["max_vel"]

        # Acceleration decoding:
        dec_input = torch.relu(self._dyn_embedder(enc)).unsqueeze(dim=1)
        dec_input = dec_input.repeat(1, self._params["acc_decoder"]["num_acc_sections"], 1)
        acc_out, _ = self._acc_decoder(dec_input)
        acc_out = torch.squeeze(self._acc_out_layer(torch.relu(acc_out)), dim=2)
        acc_out = torch.tanh(acc_out) * self._params["acc_decoder"]["max_acc"]

        return mix_out, vel_out, acc_out

    def load_model_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        print("Successfully loaded model weights from {}".format(weights_path))

    def _get_linear_stack(self, in_size: int, hidden_sizes: list, out_size: int, name: str):
        """Creates a stack of linear layers with the given sizes and ReLU activation."""
        layer_sizes = [in_size] + hidden_sizes + [out_size]
        layer_list = []
        for i in range(len(layer_sizes) - 1):
            layer_name = name + "linear" + str(i + 1)
            act_name = name + "relu" + str(i + 1)
            layer_list.append((layer_name, nn.Linear(layer_sizes[i], layer_sizes[i + 1])))
            layer_list.append((act_name, nn.LeakyReLU()))
        # Remove the last activation:
        layer_list = layer_list[:-1]
        return nn.Sequential(OrderedDict(layer_list))

    def get_params(self):
        """Accessor for the network parameters."""
        return self._params


if __name__ == "__main__":
    param_file = "mod_prediction/utils/mix_net/params/net_params.json"
    with open(param_file, "r") as fp:
        params = json.load(fp)

    batch_size = 32
    hist_len = 30
    bound_len = 50

    # Random inputs:
    hist = torch.rand((batch_size, hist_len, 2))
    left_bound = torch.rand((batch_size, bound_len, 2))
    right_bound = torch.rand((batch_size, bound_len, 2))

    net = newMixNet2(params)
    mix_out, vel_out, acc_out = net(hist, left_bound, right_bound)

    # Sanity check of output shapes:
    print("mix_out shape: {}, expected: {}".format(mix_out.shape, (batch_size, params["mixer_linear_stack"]["out_size"])))
    print("vel_out shape: {}, expected: {}".format(vel_out.shape, (batch_size, 1)))
    print("acc_out shape: {}, expected: {}".format(acc_out.shape, (batch_size, params["acc_decoder"]["num_acc_sections"])))
    print(vel_out)
