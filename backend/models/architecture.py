"""
Exact model class definitions — copied from EDISumm_Remote_Final.ipynb.

Classes defined here:
    ContextAwareAttention, MAF, VisualTransformer,
    BARTTextOnly, SimpleImageSumm, EISumm, EDISumm,
    EISummVGG, EDISummVGG

Helper functions:
    enc_mask(), causal_mask()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

# Set at module load time; loader.py may override before instantiating models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These must match config.py
BART_PATH   = "facebook/bart-base"
VGG_DIM     = 512
BART_DIM    = 768
MAX_SRC_LEN = 360
MAX_TGT_LEN = 133


# ── Mask helpers (pure PyTorch) ───────────────────────────────────

def enc_mask(amask, dtype):
    """[B,T] → [B,1,1,T] additive mask for encoder."""
    return (1.0 - amask[:, None, None, :].float()) * torch.finfo(dtype).min


def causal_mask(B, T, dtype):
    """Upper-triangular causal mask [B,1,T,T] for decoder."""
    m = torch.full((T, T), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.triu(m, 1)[None, None].expand(B, 1, T, T)


# ── ContextAwareAttention — Equations 2-4 ────────────────────────

class ContextAwareAttention(nn.Module):
    """Context-gated multihead attention (CAA) module."""

    def __init__(self, dim_model, dim_context, dropout_rate=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim_model, 1, dropout=dropout_rate,
                                           bias=True, batch_first=True)
        self.u_k  = nn.Linear(dim_context, dim_model, bias=False)
        self.w1_k = nn.Linear(dim_model, 1, bias=False)
        self.w2_k = nn.Linear(dim_model, 1, bias=False)
        self.u_v  = nn.Linear(dim_context, dim_model, bias=False)
        self.w1_v = nn.Linear(dim_model, 1, bias=False)
        self.w2_v = nn.Linear(dim_model, 1, bias=False)

    def forward(self, q, k, v, context):
        """Apply context-gated attention."""
        kc = self.u_k(context)
        vc = self.u_v(context)
        lk = torch.sigmoid(self.w1_k(k) + self.w2_k(kc))
        lv = torch.sigmoid(self.w1_v(v) + self.w2_v(vc))
        out, _ = self.attn(q, (1 - lk) * k + lk * kc, (1 - lv) * v + lv * vc)
        return out


# ── MAF — Equations 5-6 ──────────────────────────────────────────

class MAF(nn.Module):
    """Multimodal Attention Fusion: injects visual context into text hidden states."""

    def __init__(self, dim_model, dropout_rate, src_len=MAX_SRC_LEN):
        super().__init__()
        self.vct  = nn.Linear(1, src_len, bias=False)   # seq expand: 1 → T
        self.vca  = ContextAwareAttention(dim_model, BART_DIM, dropout_rate)
        self.gate = nn.Linear(2 * dim_model, dim_model)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, text, visual):
        """Fuse visual vector into text representation."""
        vc  = visual.permute(0, 2, 1)           # [B,768,1]
        vc  = self.vct(vc).permute(0, 2, 1)     # [B,T,768]
        Hv  = self.vca(text, text, text, vc)
        gv  = torch.sigmoid(self.gate(torch.cat((Hv, text), dim=-1)))
        return self.norm(text + gv * Hv)


# ── VisualTransformer — refines VGG features ─────────────────────

class VisualTransformer(nn.Module):
    """Transformer encoder applied to VGG projection for visual refinement."""

    def __init__(self, d=BART_DIM, layers=4, heads=8):
        super().__init__()
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, d, batch_first=True), layers)

    def forward(self, x):
        """Refine visual tokens through self-attention."""
        return self.enc(x)


# ── Model 1: BART Text-Only ──────────────────────────────────────

class BARTTextOnly(nn.Module):
    """BART seq2seq baseline — ignores image input."""

    def __init__(self):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(BART_PATH)

    def forward(self, input_ids, attention_mask, labels, img_vec):
        """Standard BART forward pass; img_vec is accepted but unused."""
        out = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate_summary(self, input_ids, attention_mask, img_vec,
                         max_new_tokens=80, num_beams=4, length_penalty=2.0):
        """Beam-search decoding; img_vec unused."""
        return self.bart.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            early_stopping=True, no_repeat_ngram_size=3, length_penalty=length_penalty)


# ── Model 2: Simple Fusion ───────────────────────────────────────

class SimpleImageSumm(nn.Module):
    """Simple cross-attention fusion: VGG vector fused into encoder hidden states."""

    def __init__(self):
        super().__init__()
        self.bart  = BartForConditionalGeneration.from_pretrained(BART_PATH)
        H          = self.bart.config.d_model
        self.proj  = nn.Sequential(nn.Linear(VGG_DIM, H), nn.LayerNorm(H), nn.GELU())
        self.cattn = nn.MultiheadAttention(H, 8, batch_first=True, dropout=0.1)
        self.gate  = nn.Linear(H * 2, H)
        self.norm  = nn.LayerNorm(H)

    def _vis(self, iv):
        """Project VGG vector → [B,1,768]."""
        return self.proj(iv).unsqueeze(1)

    def _fuse(self, t, img):
        """Gate-fuse image token into text hidden states."""
        B, T, H = t.shape
        I = img.expand(B, T, H)
        Hv, _  = self.cattn(t, I, I)
        return self.norm(t + torch.sigmoid(self.gate(torch.cat([t, Hv], -1))) * Hv)

    def forward(self, input_ids, attention_mask, labels, img_vec):
        """Fused encoder forward with cross-entropy loss."""
        img   = self._vis(img_vec)
        enc   = self.bart.model.encoder(input_ids=input_ids,
                    attention_mask=attention_mask, return_dict=True)
        fused = self._fuse(enc.last_hidden_state, img)
        sl    = labels.clone()
        sl[sl == -100] = self.bart.config.pad_token_id
        di    = shift_tokens_right(sl, self.bart.config.pad_token_id,
                    self.bart.config.decoder_start_token_id)
        dec   = self.bart.model.decoder(input_ids=di, encoder_hidden_states=fused,
                    encoder_attention_mask=attention_mask, return_dict=True)
        logits = self.bart.lm_head(dec.last_hidden_state)
        return (F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                ignore_index=-100, label_smoothing=0.1), logits)

    @torch.no_grad()
    def generate_summary(self, input_ids, attention_mask, img_vec,
                         max_new_tokens=80, num_beams=4, length_penalty=2.0):
        """Beam-search with fused encoder outputs."""
        img   = self._vis(img_vec)
        enc   = self.bart.model.encoder(input_ids=input_ids,
                    attention_mask=attention_mask, return_dict=True)
        fused = self._fuse(enc.last_hidden_state, img)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            num_beams=num_beams, early_stopping=True,
            no_repeat_ngram_size=3, length_penalty=length_penalty)


# ── Model 3: EI-Summ (Encoder Fusion Only) ──────────────────────

class EISumm(nn.Module):
    """Encoder-Image Summarizer: MAF injected at encoder layer 3."""

    def __init__(self):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(BART_PATH)
        H         = self.bart.config.d_model
        self.proj = nn.Sequential(nn.Linear(VGG_DIM, H), nn.LayerNorm(H), nn.GELU())
        self.vtx  = VisualTransformer()
        self.maf  = MAF(H, 0.2, MAX_SRC_LEN)
        self.fl   = 3   # fusion at encoder layer 3 (paper ablation Table 2)

    def _vis(self, iv):
        """Project and refine VGG vector through VisualTransformer → [B,1,768]."""
        return self.vtx(self.proj(iv).unsqueeze(1))

    def _enc(self, input_ids, amask, vis):
        """Custom encoder forward: injects MAF at layer self.fl."""
        e = self.bart.model.encoder
        h = e.embed_tokens(input_ids)
        if hasattr(e, "embed_scale"):
            h = h * e.embed_scale
        h  = e.layernorm_embedding(h + e.embed_positions(input_ids))
        h  = F.dropout(h, p=e.dropout, training=self.training)
        am = enc_mask(amask, h.dtype)
        for i, lyr in enumerate(e.layers):
            if i == self.fl:
                h = self.maf(h, vis)
            out = lyr(h, attention_mask=am, layer_head_mask=None)
            h   = out[0] if isinstance(out, tuple) else out
        return h

    def forward(self, input_ids, attention_mask, labels, img_vec):
        """Encoder-fused forward with cross-entropy loss."""
        vis  = self._vis(img_vec)
        fenc = self._enc(input_ids, attention_mask, vis)
        sl   = labels.clone()
        sl[sl == -100] = self.bart.config.pad_token_id
        di   = shift_tokens_right(sl, self.bart.config.pad_token_id,
                   self.bart.config.decoder_start_token_id)
        dec  = self.bart.model.decoder(input_ids=di, encoder_hidden_states=fenc,
                   encoder_attention_mask=attention_mask, return_dict=True)
        logits = self.bart.lm_head(dec.last_hidden_state)
        return (F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                ignore_index=-100, label_smoothing=0.1), logits)

    @torch.no_grad()
    def generate_summary(self, input_ids, attention_mask, img_vec,
                         max_new_tokens=80, num_beams=4, length_penalty=2.0):
        """Beam-search with MAF-fused encoder outputs."""
        vis  = self._vis(img_vec)
        fenc = self._enc(input_ids, attention_mask, vis)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=fenc),
            attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            num_beams=num_beams, early_stopping=True,
            no_repeat_ngram_size=3, length_penalty=length_penalty)


# ── Model 4: EDI-Summ (Encoder + Decoder Fusion) ────────────────

class EDISumm(nn.Module):
    """Encoder-Decoder-Image Summarizer: MAF in both encoder and decoder at layer 3."""

    def __init__(self):
        super().__init__()
        self.bart    = BartForConditionalGeneration.from_pretrained(BART_PATH)
        H            = self.bart.config.d_model
        self.proj    = nn.Sequential(nn.Linear(VGG_DIM, H), nn.LayerNorm(H), nn.GELU())
        self.vtx     = VisualTransformer()
        self.maf     = MAF(H, 0.2, MAX_SRC_LEN)       # encoder MAF
        self.dec_maf = MAF(H, 0.2, MAX_TGT_LEN)       # decoder MAF — same as paper
        self.fl      = 3

    def _vis(self, iv):
        """Project and refine VGG vector through VisualTransformer → [B,1,768]."""
        return self.vtx(self.proj(iv).unsqueeze(1))

    def _enc(self, input_ids, amask, vis):
        """Custom encoder forward: injects MAF at layer self.fl."""
        e = self.bart.model.encoder
        h = e.embed_tokens(input_ids)
        if hasattr(e, "embed_scale"):
            h = h * e.embed_scale
        h  = e.layernorm_embedding(h + e.embed_positions(input_ids))
        h  = F.dropout(h, p=e.dropout, training=self.training)
        am = enc_mask(amask, h.dtype)
        for i, lyr in enumerate(e.layers):
            if i == self.fl:
                h = self.maf(h, vis)
            out = lyr(h, attention_mask=am, layer_head_mask=None)
            h   = out[0] if isinstance(out, tuple) else out
        return h

    def _dec(self, di, eh, amask, vis):
        """Custom decoder forward: injects dec_maf at layer self.fl."""
        d = self.bart.model.decoder
        h = d.embed_tokens(di)
        if hasattr(d, "embed_scale"):
            h = h * d.embed_scale
        h = d.layernorm_embedding(h + d.embed_positions(di))
        h = F.dropout(h, p=d.dropout, training=self.training)
        B, S, H = h.shape
        T  = di.shape[1]
        cm = causal_mask(B, T, h.dtype)
        em = enc_mask(amask, h.dtype)   # [B,1,1,src] — broadcasts correctly
        for i, lyr in enumerate(d.layers):
            if i == self.fl:
                # Use MAF same as paper — NOT cross-attention
                h = self.dec_maf(h, vis)
            out = lyr(h, attention_mask=cm,
                        encoder_hidden_states=eh,
                        encoder_attention_mask=em,
                        layer_head_mask=None,
                        cross_attn_layer_head_mask=None)
            h   = out[0] if isinstance(out, tuple) else out
        return h

    def forward(self, input_ids, attention_mask, labels, img_vec):
        """Encoder+decoder fused forward with cross-entropy loss."""
        vis  = self._vis(img_vec)
        fenc = self._enc(input_ids, attention_mask, vis)
        sl   = labels.clone()
        sl[sl == -100] = self.bart.config.pad_token_id
        di   = shift_tokens_right(sl, self.bart.config.pad_token_id,
                   self.bart.config.decoder_start_token_id)
        fdec = self._dec(di, fenc, attention_mask, vis)
        logits = self.bart.lm_head(fdec)
        return (F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                ignore_index=-100, label_smoothing=0.1), logits)

    @torch.no_grad()
    def generate_summary(self, input_ids, attention_mask, img_vec,
                         max_new_tokens=80, num_beams=4, length_penalty=2.0):
        """Beam-search with MAF-fused encoder (decoder fusion is training-only)."""
        vis  = self._vis(img_vec)
        fenc = self._enc(input_ids, attention_mask, vis)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=fenc),
            attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            num_beams=num_beams, early_stopping=True,
            no_repeat_ngram_size=3, length_penalty=length_penalty)


# ── Model 5: EI-Summ VGG (Unfrozen VGG backbone) ────────────────

class EISummVGG(EISumm):
    """EI-Summ variant with an end-to-end unfrozen VGG-16 backbone.

    Accepts raw image tensors [B,3,224,224] instead of pre-extracted vectors.
    The parent class API is otherwise identical.
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights="IMAGENET1K_V1")
        self.vgg_features = vgg.features  # outputs [B,512,7,7]

    def _extract_vgg(self, image_tensor):
        """Run VGG-16 features + avg pool → [B,512]."""
        f = self.vgg_features(image_tensor)                        # [B,512,7,7]
        return F.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1)  # [B,512]

    def forward(self, input_ids, attention_mask, labels, img_vec):
        """img_vec is a raw image tensor [B,3,224,224]."""
        return super().forward(input_ids, attention_mask, labels,
                               self._extract_vgg(img_vec))

    @torch.no_grad()
    def generate_summary(self, input_ids, attention_mask, img_vec,
                         max_new_tokens=80, num_beams=4, length_penalty=2.0):
        """img_vec is a raw image tensor [B,3,224,224]."""
        return super().generate_summary(input_ids, attention_mask,
                                        self._extract_vgg(img_vec),
                                        max_new_tokens, num_beams, length_penalty)


# ── Model 6: EDI-Summ VGG (Unfrozen VGG backbone) ───────────────

class EDISummVGG(EDISumm):
    """EDI-Summ variant with an end-to-end unfrozen VGG-16 backbone.

    Accepts raw image tensors [B,3,224,224] instead of pre-extracted vectors.
    The parent class API is otherwise identical.
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights="IMAGENET1K_V1")
        self.vgg_features = vgg.features  # outputs [B,512,7,7]

    def _extract_vgg(self, image_tensor):
        """Run VGG-16 features + avg pool → [B,512]."""
        f = self.vgg_features(image_tensor)                        # [B,512,7,7]
        return F.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1)  # [B,512]

    def forward(self, input_ids, attention_mask, labels, img_vec):
        """img_vec is a raw image tensor [B,3,224,224]."""
        return super().forward(input_ids, attention_mask, labels,
                               self._extract_vgg(img_vec))

    @torch.no_grad()
    def generate_summary(self, input_ids, attention_mask, img_vec,
                         max_new_tokens=80, num_beams=4, length_penalty=2.0):
        """img_vec is a raw image tensor [B,3,224,224]."""
        return super().generate_summary(input_ids, attention_mask,
                                        self._extract_vgg(img_vec),
                                        max_new_tokens, num_beams, length_penalty)
