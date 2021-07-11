"""FragmentVC model architecture."""

from typing import Tuple, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .convolutional_transformer import Smoother, Extractor
from .utils import generate_square_subsequent_mask, generate_position_embedding
from .modules import CausalCNN1d, SelfAttentionPooling


class FragmentVC(nn.Module):
    """
    FragmentVC uses Wav2Vec feature of the source speaker to query and attend
    on mel spectrogram of the target speaker.
    """

    def __init__(self, d_feat=128, d_model=512):
        super().__init__()

        self.rhythm_extractor = RhythmExtractor(d_model)

        self.duration_predicter = DurationPredicter(d_feat, d_model)

        self.unet = UnetBlock(d_feat, d_model)

        self.smoothers = nn.TransformerEncoder(
            Smoother(d_model, 2, 1024), num_layers=3)

        self.mel_linear = nn.Linear(d_model, 80)

        self.stop_pred_linear = nn.Linear(80, 1)

        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
        )

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        tgts: Optional[Tensor] = None,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Optional[Tensor]], Tensor]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
            tgts: (batch, 80, tgt_len)
        """

        batch_size = srcs.size(0)
        device = srcs.device

        # rhythm_infos: (batch, d_model)
        rhythm_infos = self.rhythm_extractor(refs)

        go_frame = torch.zeros(batch_size, 80, 1).to(device)
        if tgts is None:
            tgts = go_frame
        else:
            tgts = torch.cat([go_frame, tgts], 2)

        # tgts: (batch, tgt_len, 768)
        tgts, attn = self.duration_predicter(
            tgts, srcs, rhythm_infos, ref_masks=src_masks)

        attn_masks = generate_square_subsequent_mask(tgts.size(1)).to(device)
        # out: (src_len, batch, d_model)
        out, attns = self.unet(
            tgts, refs, tgt_masks=attn_masks, ref_masks=ref_masks)

        # out: (src_len, batch, d_model)
        out = self.smoothers(out, mask=attn_masks)

        # out: (src_len, batch, 80)
        out = self.mel_linear(out)
        # stop_pred: (src_len, batch, 1)
        stop_pred = self.stop_pred_linear(out)
        # stop_pred: (batch, src_len)
        stop_pred = stop_pred.squeeze(-1).T

        # out: (batch, 80, src_len)
        out = out.transpose(1, 0).transpose(2, 1)
        refined = self.post_net(out)
        out_post = out + refined

        # out: (batch, 80, src_len)
        return out, out_post, attn, attns, stop_pred


class UnetBlock(nn.Module):
    """Hierarchically attend on references."""

    def __init__(self, d_feat: int, d_model: int):
        super(UnetBlock, self).__init__()

        self.conv1 = nn.Conv1d(80, d_model, 3, padding=1,
                               padding_mode="replicate")
        self.conv2 = nn.Conv1d(d_model, d_model, 3,
                               padding=1, padding_mode="replicate")
        self.conv3 = nn.Conv1d(d_model, d_model, 3,
                               padding=1, padding_mode="replicate")

        self.prenet = nn.Sequential(
            nn.Linear(d_feat, d_feat), nn.ReLU(), nn.Linear(d_feat, d_model),
        )

        self.extractor1 = Extractor(d_model, 2, 1024)
        self.extractor2 = Extractor(d_model, 2, 1024)
        self.extractor3 = Extractor(d_model, 2, 1024)

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        tgt_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, d_feat)
            tgt_masks: (batch, src_len, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """

        # tgt: (batch, tgt_len, d_model)
        tgt = self.prenet(srcs)
        # tgt: (tgt_len, batch, d_model)
        tgt = tgt.transpose(0, 1)

        # ref*: (batch, d_model, mel_len)
        ref1 = self.conv1(refs)
        ref2 = self.conv2(F.relu(ref1))
        ref3 = self.conv3(F.relu(ref2))

        # out*: (tgt_len, batch, d_model)
        out, attn1 = self.extractor1(
            tgt,
            ref3.transpose(1, 2).transpose(0, 1),
            tgt_mask=tgt_masks,
            memory_key_padding_mask=ref_masks,
        )
        out, attn2 = self.extractor2(
            out,
            ref2.transpose(1, 2).transpose(0, 1),
            tgt_mask=tgt_masks,
            memory_key_padding_mask=ref_masks,
        )
        out, attn3 = self.extractor3(
            out,
            ref1.transpose(1, 2).transpose(0, 1),
            tgt_mask=tgt_masks,
            memory_key_padding_mask=ref_masks,
        )

        # out: (tgt_len, batch, d_model)
        return out, [attn1, attn2, attn3]


class RhythmExtractor(nn.Module):
    """
    Extract rhythm information from reference.
    """

    def __init__(self, d_model):
        super(RhythmExtractor, self).__init__()
        self.conv1 = nn.Conv1d(80, d_model, 3, padding=1,
                               padding_mode="replicate")
        self.conv2 = nn.Conv1d(d_model, d_model, 3,
                               padding=1, padding_mode="replicate")
        self.conv3 = nn.Conv1d(d_model, d_model, 3,
                               padding=1, padding_mode="replicate")

        self.pooling = SelfAttentionPooling(d_model)

    def forward(
        self,
        refs: Tensor,
    ) -> Tensor:
        """Forward function.

        Args:
            srcs: (batch, 80, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """

        # ref*: (batch, d_model, ref_len)
        ref1 = self.conv1(refs)
        ref2 = self.conv2(F.relu(ref1))
        ref3 = self.conv3(F.relu(ref2))

        # refs: (batch, ref_len, d_model)
        refs = ref3.transpose(1, 2)
        # refs: (batch, d_model)
        refs = self.pooling(refs)

        return refs


class DurationPredicter(nn.Module):
    """
    Duration Predicter.
    """

    def __init__(self, d_feat, d_model):
        super(DurationPredicter, self).__init__()
        self.conv1 = CausalCNN1d(80, d_model, 3,
                                 padding_mode="replicate")
        self.conv2 = CausalCNN1d(d_model, d_model, 3,
                                 padding_mode="replicate")
        self.conv3 = CausalCNN1d(d_model, d_model, 3,
                                 padding_mode="replicate")
        self.linear_proj1 = nn.Linear(d_model * 2, d_feat)
        self.linear_proj2 = nn.Linear(768, d_feat)

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        rhythm_infos: Tensor,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            srcs: (batch, 80, src_len)
            refs: (batch, ref_len, 768)
            rhythm_infos: (batch, d_model)
            ref_masks: (batch, ref_len)
        """

        # srcs: (batch, d_model, src_len)
        srcs = self.conv1(srcs)
        srcs = self.conv2(F.relu(srcs))
        srcs = self.conv3(F.relu(srcs))

        # srcs: (batch, src_len, d_model)
        srcs = srcs.transpose(1, 2)

        # rhythm_infos: (batch, src_len, d_model)
        rhythm_infos = rhythm_infos.unsqueeze(1)
        rhythm_infos = torch.cat([rhythm_infos] * srcs.size(1), 1)

        # srcs: (batch, src_len, d_model * 2)
        srcs = torch.cat([srcs, rhythm_infos], 2)
        # srcs: (batch, src_len, d_feat)
        srcs = self.linear_proj1(srcs)
        refs = self.linear_proj2(refs)

        device = srcs.device

        querys = srcs.transpose(0, 1)
        querys += generate_position_embedding(list(querys.size())).to(device)
        querys = querys.transpose(0, 1)

        keys = refs.transpose(0, 1)
        keys += generate_position_embedding(list(keys.size())).to(device)
        keys = keys.transpose(0, 1)

        # outs: (batch, src_len, d_feat)
        outs, attn = Attention(querys, keys, refs, ref_masks)

        return outs, attn


def Attention(
    querys: Tensor,
    keys: Tensor,
    values: Tensor,
    ref_masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Attention function.

    Args:
        querys: (batch, query_len, d_feat)
        keys: (batch, key_len, d_feat)
        values: (batch, key_len, d_feat)
        ref_masks: (batch, key_len)
    """

    querys = querys / math.sqrt(querys.size(2))
    # (batch, query_len, d_feat) x (batch, d_feat, key_len)
    # -> (batch, query_len, key_len)
    attn = torch.bmm(querys, keys.transpose(-2, -1))

    if ref_masks is not None:
        ref_masks = ref_masks.unsqueeze(1)
        attn_mask = torch.zeros_like(ref_masks, dtype=torch.float)
        attn_mask.masked_fill_(ref_masks, float("-inf"))
        attn += attn_mask

    attn = F.softmax(attn, dim=-1)
    # (batch, query_len, key_len) x (batch, key_len, d_feat)
    # -> (batch, query_len, d_feat)
    output = torch.bmm(attn, values)

    output += querys

    return output, attn
