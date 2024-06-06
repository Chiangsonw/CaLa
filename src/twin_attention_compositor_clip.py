import torch
from torch import nn
import torch.nn.functional as F
import sys
from utils import device

class TwinAttentionCompositorCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Linear(2560,640)
        self.relu = nn.ReLU(inplace=True)
        self.reference_as_query_attention = torch.nn.MultiheadAttention(embed_dim=640, num_heads=1, dropout=0.0, batch_first=True)
        self.target_as_query_attention = torch.nn.MultiheadAttention(embed_dim=640, num_heads=1, dropout=0.0, batch_first=True)

    def forward(self, reference_embeddings:torch.tensor, target_embeddings:torch.tensor):
        bs, hi, h, w = reference_embeddings.size()
        #embeddings to tokens  bs x length x hidden    bs 81 2560
        reference_embeddings = reference_embeddings.view(bs,h*w,hi)
        target_embeddings = target_embeddings.view(bs,h*w,hi)
        #dim compact bs 81 640  linear降维
        reference_tokens = self.relu(self.fc(reference_embeddings))
        target_tokens =self.relu(self.fc(target_embeddings))
        cls_token = torch.randn(bs, 1, 640).to(device, non_blocking=True)
        #cat cls token  bs 82 640
        reference_tokens = torch.cat([cls_token, reference_tokens], dim=1)
        target_tokens = torch.cat([cls_token, target_tokens], dim=1)
        
        # 4 layers
        output1, _ = self.reference_as_query_attention(query=reference_tokens, key=target_tokens, value=target_tokens)
        output1, _ = self.reference_as_query_attention(query=reference_tokens, key=output1, value=output1)
        output1, _ = self.reference_as_query_attention(query=reference_tokens, key=output1, value=output1)
        output1, _ = self.reference_as_query_attention(query=reference_tokens, key=output1, value=output1)

        #4 layers
        output2, _ = self.target_as_query_attention(query=target_tokens, key=reference_tokens, value=reference_tokens)
        output2, _ = self.target_as_query_attention(query=target_tokens, key=output2, value=output2)
        output2, _ = self.target_as_query_attention(query=target_tokens, key=output2, value=output2)
        output2, _ = self.target_as_query_attention(query=target_tokens, key=output2, value=output2)


        output1_features = output1[:,0,:]
        output2_features = output2[:,0,:]
        output_features = (output1_features + output2_features) / 2
        return output_features



