import torch
import torch.nn as nn

from dataset import TriggerWordDataset
from config import get_config

class GRUCell(nn.Module):
    def __init__(self, device : torch.device, in_features : int = 64, hidden_size : int = 512):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size

        self.device = device

        self.W_z, self.W_r, self.W_h, self.b_z, self.b_r, self.b_h = self.init_weights_and_biases()

    def init_weights_and_biases(self):
        # Create weight tensors
        W_z_tensor = torch.zeros((self.in_features + self.hidden_size, self.hidden_size), dtype=torch.float32)
        W_r_tensor = torch.zeros((self.in_features + self.hidden_size, self.hidden_size), dtype=torch.float32)
        W_h_tensor = torch.zeros((self.in_features + self.hidden_size, self.hidden_size), dtype=torch.float32)

        # initialize the weights
        W_z = nn.Parameter(torch.nn.init.xavier_uniform_(W_z_tensor), requires_grad=True)
        W_r = nn.Parameter(torch.nn.init.xavier_uniform_(W_r_tensor), requires_grad=True)
        W_h = nn.Parameter(torch.nn.init.xavier_uniform_(W_h_tensor), requires_grad=True)

        # Initialize bias tensors
        b_z = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32), requires_grad=True)
        b_r = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float32), requires_grad=True) # for encouraging the model to remember more initially
        b_h = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32), requires_grad=True)

        return W_z, W_r, W_h, b_z, b_r, b_h

    def forward(self, x, prev_hidden=None):
        
        if prev_hidden is None:
            # create a hidden state
            prev_hidden = torch.zeros(x.shape[0], self.hidden_size, device=self.device, dtype=torch.float32) # hidden_state shape: [batch_size, hidden_size] | x shape: [batch_size, number_of_filters, number_of_time_steps]
        
        hidden = prev_hidden

        x_hidden_cat = torch.cat((hidden, x), dim=1)

        # Calculate the reset gate
        r_t = torch.sigmoid(torch.mm(x_hidden_cat, self.W_r) + self.b_r)

        # Calculate the update gate
        z_t = torch.sigmoid(torch.mm(x_hidden_cat, self.W_z) + self.b_z)

        # apply reset gate to hidden state
        reset_hidden = r_t * hidden
        combined_reset_hidden = torch.cat((reset_hidden, x), dim=1)

        h_t_ = torch.tanh(torch.mm(combined_reset_hidden, self.W_h) + self.b_h)
            
        hidden = (1 - z_t) * hidden + z_t * h_t_

        return hidden

class MultiLayerGRU(nn.Module):
    def __init__(self, in_features : int, conv_out_features : int, hidden_size : int, num_layers, device : torch.device):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.conv_out_features = conv_out_features
        self.num_layers = num_layers
        self.device = device

        self.layers = nn.ModuleList()
        self.layers.append(GRUCell(in_features=self.conv_out_features, hidden_size=self.hidden_size, device=self.device))

        # We do this because we know that the first GRUCell with have an output of (b, -, 512) so the (1 + n)th GRUCell would be able to take an input of (b, -, 512)

        for _ in range(self.num_layers - 1):
            self.layers.append(GRUCell(in_features=self.hidden_size, hidden_size=self.hidden_size, device=self.device))
        
    def forward(self, x, hidden_states=None): # the output from the CNN layer
        x = x.permute(0, 2, 1)
        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        final_hidden_states = []

        current_input = x
        for layer_idx, layer in enumerate(self.layers):
            layer_hidden = hidden_states[layer_idx]
            layer_outputs = []

            for t in range(x.size(1)):
                layer_hidden = layer(current_input[:, t, :], prev_hidden=layer_hidden)
                layer_outputs.append(layer_hidden)

            layer_output = torch.stack(layer_outputs, dim=1)
            
            # Why do use use the combination of multiple hidden layers ?
            # When we combine the mwe get the hidden state of the whole input, if we use the layer_hidden, that will mean that for this case we are using
            # the information of the last timestep which we actually do not want to do.
            current_input = layer_output

            final_hidden_states.append(layer_hidden)

        return current_input, final_hidden_states
        

class TriggerWordDetector(nn.Module):
    def __init__(self, in_features : int, conv_out_features : int, out_features : int, hidden_size : int, num_layers : int, dropout : float, device : torch.device): # we expect 3 different predictions - 1.positive, 2.negative, 3.background
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.conv_out_features = conv_out_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        self.layer_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_features, out_channels=self.conv_out_features, kernel_size=3),
            nn.BatchNorm1d(num_features=self.conv_out_features),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.dropout)
        )

        self.gru = MultiLayerGRU(in_features=self.in_features, conv_out_features=self.conv_out_features, hidden_size=self.hidden_size, num_layers=self.num_layers, device=self.device)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=128),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.out_features)
        )

    def forward(self, x):
        # Pass the input through the convolutional layer 
        x = self.layer_conv(x)

        # Pass the input through the GRU layers
        hidden, final_hidden_states = self.gru(x, None)

        # Pass the input through the classifier
        classifier_output = self.classifier(hidden)

        return classifier_output



def create_model(cfg, device : torch.device):
    model = TriggerWordDetector(
        in_features=cfg["in_features"],
        conv_out_features=cfg["conv_out_features"],
        out_features=cfg["out_features"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        device=device
    )

    model.to(device)
    return model 


# if __name__ == "__main__":
#     ds = TriggerWordDataset("./annotations_file.csv")
#     mfcc, label, sr = ds[6]

#     cfg = get_config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = create_model(cfg=cfg, device=device)
#     logits = model(mfcc.unsqueeze(0))
#     probability_for_each_timestep = torch.softmax(logits, dim=2)
#     avg_prob = torch.mean(probability_for_each_timestep, dim=1)
#     print(avg_prob)

