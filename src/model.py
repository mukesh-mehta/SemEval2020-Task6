import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class SimpleLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, embedding_vector, emb_dim=100,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vector))
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 1)
    
    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds

# DeepMoji Model
class DeepMoji(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_state_size, num_layers,
                 output_dim, pad_idx, dropout=0.5, bidirectional=True):

        super(DeepMoji, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_state_size = hidden_state_size
        self.num_layers=num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.bilstm_one = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_state_size,
            bidirectional=self.bidirectional
        )
        
        # do something about layer two input
        
        self.bilstm_two = nn.LSTM(
            input_size=2 * self.hidden_state_size, hidden_size=self.hidden_state_size,
            bidirectional=self.bidirectional
        )
        self.attn_layer = AttentionLayer(self.hidden_state_size, self.num_layers, self.embedding_dim)

        self.output_layer = nn.Linear(self.hidden_state_size * 2 * self.num_layers, self.output_dim)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        # [batch_size, src_sent_len]
        return mask
    
    def forward(self, inp, src_len):
        # inp = [sent_length, batch_size]
        # src_len = [batch_size]
        
        embedded = self.dropout_layer(self.embedding(inp))
        # embedded = [sent_length, batch_size, embedding_dim]
        
        embedded_packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        
        bilstm_out_1_packed, (_, _) = self.bilstm_one(embedded_packed)
        # bilstm_out_1 = [seq_len, batch_size, 2 * hidden_state_size] 
        
        bilstm_out_2_packed, (_, _) = self.bilstm_two(bilstm_out_1_packed) 
        # bilstm_out_1 = [seq_len, batch_size, 2 * hidden_state_size]
        
        bilstm_out_1, _ = nn.utils.rnn.pad_packed_sequence(bilstm_out_1_packed)
        bilstm_out_2, _ = nn.utils.rnn.pad_packed_sequence(bilstm_out_2_packed)
                
        bilstm_stacked = torch.cat((bilstm_out_1, bilstm_out_2), dim=2)
        # bilstm_stacked = [seq_len, batch_size, 4 * hidden_state_size]
                
        mask = self.create_mask(inp)
        
        attn_weights = self.attn_layer(embedded, bilstm_stacked, mask).unsqueeze(1)
        # attn_weights = [batch_size, 1, seq_len]
        
        attended_hidden_representation = torch.bmm(attn_weights, bilstm_stacked.permute(1, 0, 2)).squeeze(1)
        # attended_hidden_representation = [batch_size, 4 * hidden_state]
        
        # print("attended shape == {}".format(attended_hidden_representation.shape))

        outputs = self.output_layer(attended_hidden_representation)
        # outputs = [batch_size, output_dim]
        # print("Output shape == {}".format(outputs.shape))

        return outputs

class AttentionLayer(nn.Module):
    def __init__(self, hidden_state_size, num_layers, embedding_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
#         self.attn = nn.Linear((self.num_layers * 2 * self.hidden_state_size) + self.embedding_dim, self.hidden_state_size)
        
        self.attn = nn.Parameter(torch.rand((self.num_layers * 2 * self.hidden_state_size) + self.embedding_dim))
        
        
    def forward(self, embedded, lstm_outputs, mask):
        # embedded = [src_sent_len, batch_size, embedding_dim]
        # lstm_outputs = [src_sent_len, batch_size, enc_hidden_state_size*2*2]
        
        batch_size = embedded.shape[1]
        # calculating energies
        
        embed_concat = torch.cat((embedded, lstm_outputs), dim=2)
        # embed_concat = [src_sent_len, batch_size, embedding_dim + hidden_state_size*4]
        
        attn = self.attn.repeat(batch_size, 1).unsqueeze(2)
        # attn = [batch_size, 4*hidden_state + embedding_dim, 1]
        
        attention = torch.bmm(embed_concat.permute(1,0,2), attn).squeeze(2)
        # attention = [batch_size, seq_len]
        
        attn_weights = attention.masked_fill(mask == 0, -1e10)
        # attn_weights = [batch_size, seq_len]
        
        attn_weights = F.softmax(attention, dim=1)
        # attn_weights = [batch_size, seq_len]
        return attn_weights