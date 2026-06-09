import torch
import torch.nn as nn

from utils.embedding import createCrossAttentionMask




class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config["feat_dim"],
            num_heads=config["num_heads"],
            kdim=config["feat_dim"],
            vdim=config["feat_dim"],
            batch_first=True  
        )

    def forward(self, query, key, value, attn_mask=None):
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask == 1 
            batch_size = query.size(0)
            num_heads = self.multihead_attn.num_heads
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(1, num_heads, 1, 1)
            attn_mask = attn_mask.view(batch_size * num_heads, attn_mask.size(2), attn_mask.size(3))
            attn_mask = attn_mask.to(dtype=query.dtype) * (-1e9)
            
        output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask  
        )
        return output, attn_weights.detach().cpu()
    
class Adaptor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(config["feat_dim"])
        
        self.proj_f1 = nn.Linear(config["input_dim"], config["feat_dim"])
        self.proj_f2 = nn.Linear(config["input_dim"], config["feat_dim"])

        self.adaptor_f21 = nn.Linear(config["feat_dim"], config["feat_dim"])
        self.adaptor_f12 = nn.Linear(config["feat_dim"], config["feat_dim"])

    def forward(self, f1, f2):
        f1 = self.relu(self.proj_f1(f1))
        f2 = self.relu(self.proj_f2(f2))
        f1 = self.relu(f1 + self.adaptor_f21(f2))
        f2 = self.relu(f2 + self.adaptor_f12(f1))
        return f1, f2, self.adaptor_f21.weight.detach().cpu(), self.adaptor_f12.weight.detach().cpu()


class CrossAdaptiveAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config["feat_dim"])
        self.adaptor = Adaptor(config)
        self.attention = Attention(config)
    def forward(self, f1, f2, mask_12):
        f1, f2, weights21, weights12 = self.adaptor(f1, f2)
        f1 = self.norm(f1)
        f2 = self.norm(f2)
        output, att_scores = self.attention(f1, f2, f2, mask_12)
        output = self.norm(output)
        return output, f1, f2, att_scores
    
class Alignment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_1 = nn.Linear(config["feat_dim"], config["feat_dim"])
        self.proj_2 = nn.Linear(config["feat_dim"], config["feat_dim"])
        self.proj_3 = nn.Linear(config["feat_dim"], config["feat_dim"])

        self.config = config
        self.relu = nn.ReLU()

    def forward(self, f1, f2, f_f):
        f_1, f_2, f_f = self.relu(self.proj_1(f1)), self.relu(self.proj_2(f2)), self.relu(self.proj_3(f_f))

        features = torch.cat([f_1, f_2, f_f], dim=-1)

        return features
    

class PHLAModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.relu = nn.ReLU()
        self.CAA = CrossAdaptiveAttention(config)

        self.align_fusion = Alignment(config)

        self.cls = nn.Linear(self.config["feat_dim"] * self.config["seqMaxLength"] * 3, 2)

    def forward(self, inputDict, mask=None):


        mask_ph = createCrossAttentionMask(
                valid_len_a=inputDict["peptide"]["length"],
                valid_len_b=inputDict["hla"]["length"],
                len_a=self.config["seqMaxLength"],
                len_b=self.config["seqMaxLength"],
                batch_size=inputDict["peptide"]["length"].size(0)
            )
                # batch_maxSeqLength * feat_dim
        f_p = inputDict["peptide"]["embedding"]

        # batch_maxSeqLength * feat_dim
        f_h = inputDict["hla"]["embedding"]
        f_f, f_p, f_h, att_scores = self.CAA(f_p, f_h, mask_ph)
        
        f_f, f_p, f_h = self.relu(f_f), self.relu(f_p), self.relu(f_h)

        features = self.align_fusion(f_p, f_h, f_f)

        prediction = self.cls(self.relu(features.reshape([features.size(0), -1])))
        return prediction, f_p, f_h, att_scores
    

class PTCRModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.relu = nn.ReLU()
        self.CAA = CrossAdaptiveAttention(config)

        self.align_fusion = Alignment(config)

        self.cls = nn.Linear(self.config["feat_dim"] * self.config["seqMaxLength"] * 3, 2)

    def forward(self, inputDict, mask=None):


        mask_pt = createCrossAttentionMask(
                valid_len_a=inputDict["peptide"]["length"],
                valid_len_b=inputDict["tcr"]["length"],
                len_a=self.config["seqMaxLength"],
                len_b=self.config["seqMaxLength"],
                batch_size=inputDict["peptide"]["length"].size(0)
            )
                # batch_maxSeqLength * feat_dim
        f_p = inputDict["peptide"]["embedding"]

        # batch_maxSeqLength * feat_dim
        f_t = inputDict["tcr"]["embedding"]
        f_f, f_p, f_t, att_scores = self.CAA(f_p, f_t, mask_pt)
        
        f_f, f_p, f_t = self.relu(f_f), self.relu(f_p), self.relu(f_t)

        features = self.align_fusion(f_p, f_t, f_f)

        prediction = self.cls(self.relu(features.reshape([features.size(0), -1])))
        return prediction, f_p, f_t, att_scores