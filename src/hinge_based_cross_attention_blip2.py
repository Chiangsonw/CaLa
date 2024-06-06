import torch
from torch import nn
import torch.nn.functional as F
from utils import device
import math

class HingebasedCrossAttentionBLIP2(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        # attention proj
        self.query_ref1 = nn.Linear(embed_dim,embed_dim)
        self.key_text1 = nn.Linear(embed_dim,embed_dim)
        # self.query_text1 = nn.Linear(embed_dim,embed_dim)
        self.key_tar1 = nn.Linear(embed_dim,embed_dim)
        self.value1 = nn.Linear(embed_dim,embed_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.query_ref2 = nn.Linear(embed_dim,embed_dim)
        self.key_text2 = nn.Linear(embed_dim,embed_dim)
        self.key_tar2 = nn.Linear(embed_dim,embed_dim)
        self.value2 = nn.Linear(embed_dim,embed_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, reference_embeds, caption_embeds, target_embeds):
        psudo_T = self.hca_T_share_text(reference_embeds, caption_embeds, target_embeds)
        return psudo_T

    def hca_T_share_text(self, reference_embeds, caption_embeds, target_embeds):
        bs , len_r , dim = reference_embeds.shape
        attA = self.multiply(self.query_ref1(reference_embeds), self.key_text1(caption_embeds)) / math.sqrt(dim)
        attB = self.multiply(self.key_text1(caption_embeds), self.key_tar1(target_embeds)) / math.sqrt(dim)
        attC = self.dropout1(F.softmax(torch.matmul(attA, attB), dim=-1))
        psudo_T = torch.matmul(attC , self.value1(target_embeds))
        return psudo_T[:,0,:]
    
    def hca_T_R_share_text(self, reference_embeds, caption_embeds, target_embeds):
        bs , len_r , dim = reference_embeds.shape
        attA1 = self.multiply(self.query_ref1(reference_embeds), self.key_text1(caption_embeds)) / math.sqrt(dim)
        attB1 = self.multiply(self.key_text1(caption_embeds), self.key_tar1(target_embeds)) / math.sqrt(dim)
        attC1 = self.dropout1(F.softmax(torch.matmul(attA1, attB1), dim=-1))
        psudo_T = torch.matmul(attC1 , self.value1(target_embeds))

        attA2 = self.multiply(self.query_ref2(target_embeds), self.key_text2(caption_embeds)) / math.sqrt(dim)
        attB2 = self.multiply(self.key_text2(caption_embeds), self.key_tar2(reference_embeds)) / math.sqrt(dim)
        attC2 = self.dropout2(F.softmax(torch.matmul(attA2, attB2), dim=-1))
        psudo_R = torch.matmul(attC2 , self.value2(reference_embeds))

        return psudo_T[:,0,:], psudo_R[:,0,:]
    
    def hca_T_multihead_4layer(self, reference_embeds, caption_embeds, target_embeds):
        bs , len_r , dim = reference_embeds.shape
        attA1 = self.multiply(self.query_ref1(reference_embeds), self.key_text1(caption_embeds)) / math.sqrt(dim)
        attB1 = self.multiply(self.key_text1(caption_embeds), self.key_tar1(target_embeds)) / math.sqrt(dim)
        attC1 = self.dropout1(F.softmax(torch.matmul(attA1, attB1), dim=-1))
        psudo_T1 = torch.matmul(attC1 , self.value1(target_embeds))

        attA2 = self.multiply(self.query_ref2(reference_embeds), self.key_text2(caption_embeds)) / math.sqrt(dim)
        attB2 = self.multiply(self.key_text2(caption_embeds), self.key_tar2(target_embeds)) / math.sqrt(dim)
        attC2 = self.dropout2(F.softmax(torch.matmul(attA2, attB2), dim=-1))
        psudo_T2 = torch.matmul(attC2 , self.value2(target_embeds))

        attA3 = self.multiply(self.query_ref3(reference_embeds), self.key_text3(caption_embeds)) / math.sqrt(dim)
        attB3 = self.multiply(self.key_text3(caption_embeds), self.key_tar3(target_embeds)) / math.sqrt(dim)
        attC3 = self.dropout3(F.softmax(torch.matmul(attA3, attB3), dim=-1))
        psudo_T3 = torch.matmul(attC3 , self.value3(target_embeds))

        attA4 = self.multiply(self.query_ref4(reference_embeds), self.key_text4(caption_embeds)) / math.sqrt(dim)
        attB4 = self.multiply(self.key_text4(caption_embeds), self.key_tar4(target_embeds)) / math.sqrt(dim)
        attC4 = self.dropout4(F.softmax(torch.matmul(attA4, attB4), dim=-1))
        psudo_T4 = torch.matmul(attC4 , self.value4(target_embeds))
        
        return (psudo_T1[:,0,:] + psudo_T2[:,0,:] + psudo_T3[:,0,:] + psudo_T4[:,0,:]) / 4

    def hca_T_multihead_2layer(self, reference_embeds, caption_embeds, target_embeds):
        bs , len_r , dim = reference_embeds.shape
        attA1 = self.multiply(self.query_ref1(reference_embeds), self.key_text1(caption_embeds)) / math.sqrt(dim)
        attB1 = self.multiply(self.key_text1(caption_embeds), self.key_tar1(target_embeds)) / math.sqrt(dim)
        attC1 = self.dropout1(F.softmax(torch.matmul(attA1, attB1), dim=-1))
        psudo_T1 = torch.matmul(attC1 , self.value1(target_embeds))

        attA2 = self.multiply(self.query_ref2(reference_embeds), self.key_text2(caption_embeds)) / math.sqrt(dim)
        attB2 = self.multiply(self.key_text2(caption_embeds), self.key_tar2(target_embeds)) / math.sqrt(dim)
        attC2 = self.dropout2(F.softmax(torch.matmul(attA2, attB2), dim=-1))
        psudo_T2 = torch.matmul(attC2 , self.value2(target_embeds))
        
        return (psudo_T1[:,0,:] + psudo_T2[:,0,:]) / 2
    
    def multiply(self, embedsA, embedsB):
        bs, len_a , dim = embedsA.shape
        bs, len_b , dim = embedsB.shape

        # 扁平化
        embedsA = embedsA.view(bs, -1, dim)  # 形状为 bs x (length_a * dim)
        embedsB = embedsB.view(bs, -1, dim)  # 形状为 bs x (length_b * dim)

        # 点积计算
        attention_scores_flat = torch.matmul(embedsA, embedsB.transpose(-1, -2))  # 转置 Key 的维度

        # 还原形状
        attention_scores = attention_scores_flat.view(bs, len_a, len_b)

        return attention_scores


        