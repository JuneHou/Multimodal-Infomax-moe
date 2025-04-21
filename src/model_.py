import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet
from modules.module import *
from modules.config import MoEConfig

from transformers import BertModel, BertConfig

# class ModalProjection(nn.Module):
#     """ Projection for a single modality. """
#     def __init__(self, in_dim, out_dim):
#         super(ModalProjection, self).__init__()
#         self.fc = nn.Linear(in_dim, out_dim)
#         self.act = nn.ReLU()
#         self.norm = nn.LayerNorm(out_dim)

#     def forward(self, x):
#         return self.norm(self.act(self.fc(x)))

class ModalProjection(nn.Module):
    """ Projection for a single modality. """
    def __init__(self, in_dim, out_dim):
        super(ModalProjection, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        # self.norm = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class VariableVariancePred(nn.Module):
    """ Variable Variance Prediction
        Variance of N sample given the target:
        var = 1/N * sum_i (x_i - mu)^2
    
    """

    def __init__(self, input_dim, hidden_dim, dropout, output_dim = 1, n_sample = 1):
        super(VariableVariancePred, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_sample = n_sample

        self.drop = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, train = False):
        # input torch.Size([32, 384])
        x = self.drop(x)
        # x.shape = torch.Size([32, 128])
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        std = torch.exp(self.fc_var(x).clamp(-20, 2))

        dist = torch.distributions.Normal(mu, std)

        if self.n_sample > 1:
            x = dist.rsample(torch.Size([self.n_sample]))  # (n_sample, batch, output_dim)
            x = x.permute(1, 0, 2)  # (batch, n_sample, output_dim)
        else:
            x = dist.sample()  # (batch, output_dim)

        return x, mu, std


        # # Just for testing
        # if train:
        #     return std, x
        # else:
        #     return std, mu




class MultiModalEncoder(nn.Module):
    """ MultiModalEncoder including modality specific projections and transformer encoder for fusion """
    def __init__(self, hp, text_dim=768, vis_dim=16, aud_dim=16, hidden_dim=128, n_heads=8, n_layers=3, device=None):
        super(MultiModalEncoder, self).__init__()
        self.device = torch.device("cuda")
        self.hp = hp
        output_dim = 0
        if 'text' in hp.modality:
            self.text_proj = ModalProjection(text_dim, hidden_dim)
            output_dim += hidden_dim
        if 'video' in hp.modality:
            self.visual_proj = ModalProjection(vis_dim, hidden_dim)
            output_dim += hidden_dim
        if 'audio' in hp.modality:
            self.acoustic_proj = ModalProjection(aud_dim, hidden_dim)
            output_dim += hidden_dim
        self.fusion_transformer = TransformerCrossEncoder(
            hp,
            embed_dim= hidden_dim,
            num_heads=n_heads,
            layers=hp.moe_layers,
            device=self.device,
            output_dim=output_dim)  # Pass the device object)

        self.fusion_prj = SubNet(
            in_size = hidden_dim * hp.num_modality,
            hidden_size = 128,
            n_class = hp.n_class,
            dropout = 0.1
        )

        # self.fusion_prj = VariableVariancePred(
        #     input_dim=hidden_dim * hp.num_modality,
        #     hidden_dim=hidden_dim,
        #     dropout=0.1,
        #     output_dim=hp.n_class,
        #     n_sample=1
        # )

        self.to(device)

    def forward(self, text, visual, acoustic, text_weights, visual_weights, acoustic_weights):
        # Assume inputs are already in the appropriate device and reshaped as needed
        combined = []
        modality = []
        if text is not None:
            text = self.text_proj(text)
            text = text * text_weights.view(-1, 1)
            combined.append(text)
            modality.append('text')
        if visual is not None:
            visual = self.visual_proj(visual)
            visual = visual * visual_weights.view(-1, 1)
            combined.append(visual)
            modality.append('visual')
        if acoustic is not None:
            acoustic = self.acoustic_proj(acoustic)
            acoustic = acoustic * acoustic_weights.view(-1, 1)
            combined.append(acoustic)
            modality.append('acoustic')
        # combined = torch.stack([text, visual, acoustic], dim=1)  # (batch, 3, 128)
        fused_output, _ = self.fusion_transformer(combined, modality)  # Back to (batch, 3, 128)
        cat_hidden = torch.cat(fused_output, dim=1)

        _, fused_output = self.fusion_prj(cat_hidden) # logits is for categorical labels

        # if self.hp.n_class == 2:
        #     fused_output = torch.sigmoid(fused_output)
        # elif self.hp.n_class > 2:
        #     fused_output = torch.softmax(fused_output, dim=-1)

        return fused_output

class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.device = torch.device("cuda")
        self.hp = hp
        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        if 'text' in hp.modality:
            self.text_enc = LanguageEmbeddingLayer(hp)
        if 'video' in hp.modality:
            self.visual_enc = RNNEncoder(
                in_size = hp.d_vin,
                hidden_size = hp.d_vh,
                out_size = hp.d_vout,
                num_layers = hp.n_layer,
                dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
                bidirectional = hp.bidirectional
            )
        if 'audio' in hp.modality:
            self.acoustic_enc = RNNEncoder(
                in_size = hp.d_ain,
                hidden_size = hp.d_ah,
                out_size = hp.d_aout,
                num_layers = hp.n_layer,
                dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
                bidirectional = hp.bidirectional
            )

        self.multi_modal_encoder = MultiModalEncoder(
            hp,
            text_dim=768, 
            vis_dim=hp.d_vout, 
            aud_dim=hp.d_aout, 
            hidden_dim=128, 
            n_heads=8, 
            n_layers=hp.moe_layers, 
            device=self.device)

        
            
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, \
        text_weights, visual_weights, acoustic_weights, y=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        if 'text' in self.hp.modality:
            enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
            text = enc_word[:,0,:] # (batch_size, emb_size)
            text_weights = torch.tensor(text_weights, dtype=torch.float32, device=text.device)
            # text = text * text_weights.view(-1, 1)
            # torch.Size([32, 768])
        else :
            text = None
        if 'audio' in self.hp.modality:
            acoustic = self.acoustic_enc(acoustic, a_len)
            acoustic_weights = torch.tensor(acoustic_weights, dtype=torch.float32, device=acoustic.device)
            # acoustic = acoustic * acoustic_weights.view(-1, 1)
            # torch.Size([32, 16])
        else :
            acoustic = None
        if 'video' in self.hp.modality:
            visual = self.visual_enc(visual, v_len)
            visual_weights = torch.tensor(visual_weights, dtype=torch.float32, device=visual.device)
            # visual = visual * visual_weights.view(-1, 1)
            # torch.Size([261, 32, 20]) to torch.Size([32, 16]) 
        else :
            visual = None
        fused_output = self.multi_modal_encoder(text, visual, acoustic, text_weights, visual_weights, acoustic_weights) 

        return fused_output