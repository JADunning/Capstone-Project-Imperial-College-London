Mini-lesson 20.6: Building a transformer in PyTorch
Transformers have revolutionised natural language processing by enabling models to capture relationships in sequences more effectively than convolutional neural networks. At their core, transformers use attention mechanisms to weigh the influence of each token in a sequence when computing representations, allowing for better context understanding and parallel processing.

In this mini-lesson, you will build a simplified transformer model from scratch using PyTorch. This hands-on approach will deepen your understanding of key components such as multi-head self-attention, positional encoding and feed-forward networks by implementing them step by step.

If you'd like a refresher on embeddings, activation functions, and optimisation techniques, refer to Mini-lessons 1.2, 1.3, 1.8, 1.17, 1.19 and 1.20 from Module 1: Mathematical Concepts in ML/AI.

Step-by-step approach for building a transformer in PyTorch
Select each tab to learn more about each step.

Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
These imports are required to build the transformer architecture.

Token embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_size)
    def forward(self, x):
    return self.embedding(x)
A lookup table that converts each word (token index) into a dense vector of fixed size (embed_size). Neural networks cannot work directly with text (strings). They need numerical vectors that capture semantic meaning. The embedding layer transforms discrete token indices into continuous vector representations while preserving batch and sequence dimensions. This is crucial for the transformer to learn meaningful features for each token in context. This dimensional transformation is fundamental in all transformer models and helps bridge symbolic text data with the numerical operations in deep learning models.

Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
Positional encoding is a mathematical trick that encodes position information (word order) using sine and cosine functions. This is required because transformers have no inherent sense of word order. Positional encoding adds number patterns that tell the model where each word is in a sentence. Without this, the transformer would treat sentences like ‘dog bites man’ and ‘man bites dog’ as identical.

Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        queries = self.query_linear(query).view(N, query_len, self.heads, self.head_dim)
        keys = self.key_linear(key).view(N, key_len, self.heads, self.head_dim)
        values = self.value_linear(value).view(N, value_len, self.heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out
This step is the core mechanism of the transformer. It computes relationships between all of the words in the input. Multi-head attention works by first projecting words into three forms: Q, K and V. The model compares queries and keys to calculate attention scores, scales them and applies softmax to turn the scores into probabilities. These probabilities determine how much weight each V should receive, producing contextualised word representations. Instead of doing this once, multi-head attention does it several times in parallel with different projections – these are the ‘heads’. Each head looks at the data in a different way, capturing varied patterns. These may include things such as word order, meaning or long-distance connections, but the roles are not fixed. Their outputs are combined to give the model a richer understanding of the text.

Feed-forward layer
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
This is a simple two-layer fully connected neural network applied independently to each word position. After attention combines information from different words, the feed-forward layer applies a nonlinear transformation that expands the model’s capacity to learn richer patterns. You can think of it as the step when the model processes and refines the gathered information before moving forward.

Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(embed_size, heads)
        self.ff = FeedForward(embed_size, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
This layer contains multi-head self-attention, a feed-forward network and norm and dropout layers for stability. An encoder is for understanding the input. Stacking multiple such layers lets the model capture deeper and more abstract relationships in the source sentence.

Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, heads)
        self.cross_attn = MultiHeadAttention(embed_size, heads)
        self.ff = FeedForward(embed_size, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, tgt_mask):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x
A transformer decoder layer generates output tokens step by step by integrating masked self-attention (to prevent peeking at future tokens), cross-attention (to attend to encoder outputs) and a feed-forward network. Each component is followed by dropout, residual connections and layer normalisation to stabilise training. This structure enables the decoder to produce coherent, context-aware sequences – ideal for tasks such as summarisation and translation.

Full transformer summariser
class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=2, heads=4, forward_expansion=4, dropout=0.1, max_len=100):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)      

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)  

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_mask & subsequent_mask
        return tgt_mask  

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)      

        enc_out = self.token_emb(src)
        enc_out = self.pos_enc(enc_out)
        enc_out = self.dropout(enc_out)     

        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)     

        dec_out = self.token_emb(tgt)
        dec_out = self.pos_enc(dec_out)
        dec_out = self.dropout(dec_out)      

        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        final_out = self.fc_out(dec_out)
        return final_out
This class is the heart of a summarisation system. It’s where raw text gets transformed into meaningful, compressed output. The TransformerSummarizer class defines a complete transformer-based sequence-to-sequence model for text summarisation, combining token embeddings, positional encodings, multiple encoder and decoder layers and a final output projection. It uses source masks to ignore padding and target masks to enforce autoregressive decoding, ensuring the model only focuses on valid tokens and cannot access future positions during training or inference. The encoder processes the input text, the decoder generates the summary step by step, and the final linear layer maps decoder outputs to vocabulary predictions.

Vocabulary and tokeniser
word2idx = {
    '[PAD]': 0, '[SOS]': 1, '[EOS]': 2,
    'the': 3, 'cat': 4, 'sat': 5, 'on': 6, 'mat': 7,
    'a': 8, 'dog': 9, 'is': 10, 'here': 11,
}
idx2word = {v:k for k,v in word2idx.items()}

def tokenize(text):
    return [word2idx.get(word, 0) for word in text.lower().split()]

def detokenize(indices):
    words = []
    for idx in indices:
        if idx == word2idx['[EOS]']:
            break
        words.append(idx2word.get(idx, '[UNK]'))
    return ' '.join(words)
This code defines a basic tokeniser and detokeniser that convert words to integers and back, enabling raw text to be processed by transformer models, which operate on numerical data. It uses a predefined vocabulary mapping (word2idx) and its inverse (idx2word), along with special tokens such as [PAD] for padding, [SOS] to mark the start of a summary and [EOS] to signal its end. The tokenise function transforms input text into a list of token IDs, and the detokenise function reconstructs readable text from model predictions, stopping at the end-of-sequence token. This mechanism is essential for bridging human language and a machine-readable format.

Greedy decoding for summary generation
def generate_summary(model, src_sentence, max_len=10, device='cpu'):
    model.eval()
    src_tokens = torch.tensor([tokenize(src_sentence)], dtype=torch.long, device=device)
    # Assume [PAD]=0, no padding needed here for single sentence
    tgt_tokens = torch.tensor([[word2idx['[SOS]']]], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(src_tokens, tgt_tokens)
        next_token_logits = output[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(1)
        tgt_tokens = torch.cat((tgt_tokens, next_token), dim=1)
        if next_token.item() == word2idx['[EOS]']:
            break
    summary = tgt_tokens[0,1:].cpu().tolist()  # remove [SOS]
    return detokenize(summary)
This function generates a summary by tokenising the input sentence, initialising with a [SOS] token and repeatedly passing tokens through the model to predict the next word using greedy decoding (argmax). It appends each predicted token until it reaches [EOS] or hits the maximum length and then detokenises the result to produce readable output. This loop transforms model predictions into a coherent summary, bridging raw text and generated output.

Running the model
input_sentence = "My name …”
summary = generate_summary(model, input_sentence, max_len=5, device=device)
input_word_count = len(input_text.split())
summary_word_count = len(summary.split())
This code performs a demo run of the summarisation model by feeding it an input sentence and generating a short summary using greedy decoding. Since the model is untrained, the output won't be meaningful yet, but the process confirms that the full pipeline – from tokenisation to prediction to detokenisation – is functioning correctly. Word counts at the end help compare input and output lengths, offering a quick check on summary compression. In the required assignment following this mini-lesson, the same content is summarised using a trained BERT model to show the improvement in the content of the summary.