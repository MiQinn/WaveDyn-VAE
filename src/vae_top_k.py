import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch import optim
import lightning as L
from src.vae.datamodule import MolDataModule
import re

NOP = "<NOP>"  # 用于非法 token 替换

class VAE(L.LightningModule):
    def __init__(
        self,
        max_len: int = None,
        vocab_size: int = None,
        latent_dim: int = 256,
        embedding_dim: int = 128,
        learning_rate: float = 3e-4,
        hidden_sizes: list[int] = None,
        p: float = 0.75,
        padding_idx: int = 0,
        dm: MolDataModule = None,   # ✅ 新增参数
    ):
        if hidden_sizes is None:
            hidden_sizes = [1024, 512]

        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.p = p
        self.dm = dm

        self.dropout = 0.15
        self.beta_start = 0.0
        self.beta_end = p
        self.anneal_epochs = 60

        self.save_hyperparameters()

        def _block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(self.dropout))  # 新增 dropout
            return layers

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        encoder_layers = _block(
            max_len * embedding_dim, hidden_sizes[0], normalize=False
        )
        for i in range(len(hidden_sizes) - 1):
            encoder_layers.extend(_block(hidden_sizes[i], hidden_sizes[i + 1]))
        encoder_layers.append(nn.Linear(hidden_sizes[-1], latent_dim * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = _block(latent_dim, hidden_sizes[-1])
        for i in range(len(hidden_sizes) - 1, 0, -1):
            decoder_layers.extend(_block(hidden_sizes[i], hidden_sizes[i - 1]))
        decoder_layers.append(nn.Linear(hidden_sizes[0], max_len * vocab_size))
        self.decoder = nn.Sequential(*decoder_layers)

    # ===== 新增：KL annealing函数 =====
    def get_beta(self):
        if self.current_epoch >= self.anneal_epochs:
            return self.beta_end
        else:
            return self.beta_start + (self.beta_end - self.beta_start) * (
                self.current_epoch / self.anneal_epochs
            )

    # ===== 新版 encode_to_latent =====
    @torch.no_grad()
    def encode_to_latent(self, smiles: str, dm: MolDataModule):
        """
        将单个 SMILES 编码为潜空间向量
        兼容现有 MolDataModule，不依赖 dataset.smiles_to_tensor
        """
        # 使用 MolDataModule.encode() 得到 token tensor
        tensor = dm.encode(smiles).unsqueeze(0).to(next(self.parameters()).device)

        # 编码得到 mu 和 log_var
        enc_out = self.encoder(self.embedding(tensor).view(1, -1)).view(1, 2, self.latent_dim)
        mu, log_var = enc_out[:, 0, :], enc_out[:, 1, :]

        # 重参数化采样
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    # ===== encode/decode 原方法 =====
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.encoder(self.embedding(x).view((len(x), -1))).view(
            (-1, 2, self.latent_dim)
        )
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z).view(-1, self.max_len, self.vocab_size)

    # ===== decode_to_tokens =====
    def decode_to_tokens(self, z: Tensor, dm: MolDataModule = None, top_k: int = 5, device=None) -> Tensor:
        if dm is None:
            if self.dm is None:
                raise ValueError("MolDataModule (dm) must be provided for decode_to_tokens")
            dm = self.dm
        if device is None:
            device = z.device

        B = z.shape[0]
        max_len = self.max_len
        vocab_size = self.vocab_size

        generated = torch.full((B, max_len), dm.dataset.token_to_id.get(NOP, 0),
                               dtype=torch.long, device=device)

        ring_stacks = [[] for _ in range(B)]
        open_parens = [0 for _ in range(B)]
        ring_counts = [{} for _ in range(B)]
        last_tokens = [None for _ in range(B)]
        ring_tokens = set(str(i) for i in range(1, 10))

        for pos in range(max_len):
            logits = self.decode(z)[:, pos, :]  # (B, vocab_size)
            probs = torch.softmax(logits, dim=-1)

            if top_k > 0:
                topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                sampled = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
                next_tokens = topk_indices.gather(1, sampled.unsqueeze(1)).squeeze(-1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            for b in range(B):
                prev_token = dm.dataset.id_to_token[generated[b, pos-1].item()] if pos > 0 else None
                cur_token = dm.dataset.id_to_token[next_tokens[b].item()]

                if prev_token == "*" and cur_token == "*":
                    next_tokens[b] = dm.dataset.token_to_id.get(NOP, 0)
                    cur_token = NOP

                nums = re.findall(r"\d+", cur_token)
                if nums:
                    if re.fullmatch(r"\d+", cur_token):
                        if prev_token is not None and re.match(r"^[A-Za-z\*\[]", prev_token):
                            merged_tok = prev_token + cur_token
                            merged_id = dm.dataset.token_to_id.get(merged_tok, None)
                            if merged_id is not None:
                                generated[b, pos-1] = merged_id
                                next_tokens[b] = dm.dataset.token_to_id.get(NOP, 0)
                                cur_token = NOP
                            else:
                                next_tokens[b] = dm.dataset.token_to_id.get(NOP, 0)
                                cur_token = NOP
                        else:
                            next_tokens[b] = dm.dataset.token_to_id.get(NOP, 0)
                            cur_token = NOP

                if cur_token == "(":
                    open_parens[b] += 1
                elif cur_token == ")":
                    if open_parens[b] > 0:
                        open_parens[b] -= 1
                    else:
                        next_tokens[b] = dm.dataset.token_to_id.get(NOP, 0)
                        cur_token = NOP

                if cur_token != NOP:
                    nums = re.findall(r"\d+", cur_token)
                    illegal_due_to_ring = False
                    for n in nums:
                        if n in ring_stacks[b]:
                            ring_stacks[b].remove(n)
                        else:
                            if len(ring_stacks[b]) < 3:
                                ring_stacks[b].append(n)
                            else:
                                illegal_due_to_ring = True
                                break
                        count = ring_counts[b].get(n, 0)
                        if count >= 2:
                            illegal_due_to_ring = True
                            break
                    if illegal_due_to_ring:
                        next_tokens[b] = dm.dataset.token_to_id.get(NOP, 0)
                        cur_token = NOP
                    else:
                        for n in nums:
                            ring_counts[b][n] = ring_counts[b].get(n, 0) + 1

                if cur_token == '=' and last_tokens[b] == '=':
                    next_tokens[b] = dm.dataset.token_to_id.get(NOP, 0)
                    cur_token = NOP

                last_tokens[b] = cur_token
                generated[b, pos] = next_tokens[b]

        return generated

    # ===== decode_to_smiles =====
    def decode_to_smiles(self, z: Tensor, dm: MolDataModule = None, top_k: int = 5, device=None) -> list[str]:
        if dm is None:
            if self.dm is None:
                raise ValueError("MolDataModule (dm) must be provided for decode_to_smiles")
            dm = self.dm

        token_ids = self.decode_to_tokens(z, dm=dm, top_k=top_k, device=device).tolist()
        smiles_list = []
        for seq in token_ids:
            tokens = [dm.dataset.id_to_token[idx] for idx in seq if idx in dm.dataset.id_to_token]

            if "<eos>" in tokens:
                tokens = tokens[:tokens.index("<eos>")]

            tokens = [t for t in tokens if t not in ("<NOP>", "[nop]")]
            smiles = "".join(tokens)
            smiles_list.append(smiles)
        return smiles_list

    # ===== forward / optim / loss =====
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)   # 新增 weight_decay
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def loss_function(self, pred, target, mu, log_var, batch_size):
        nll = F.cross_entropy(pred, target, reduction='mean')
        kld_sum = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        denom = float(pred.shape[0]) if pred.shape[0] > 0 else float(batch_size)
        kld = kld_sum / denom
        beta = self.get_beta()  # 使用 annealing
        loss = (1 - beta) * nll + beta * kld

        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=-1e6)
        if not torch.isfinite(nll):
            nll = torch.nan_to_num(nll, nan=1e6, posinf=1e6, neginf=-1e6)
        if not torch.isfinite(kld):
            kld = torch.nan_to_num(kld, nan=1e6, posinf=1e6, neginf=-1e6)

        return loss, nll, kld

    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        pred = out.reshape((-1, self.vocab_size))
        target = train_batch.flatten().long()
        loss, nll, kld = self.loss_function(pred, target, mu, log_var, len(train_batch))

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_nll", nll, sync_dist=True)
        self.log("train_kld", kld, sync_dist=True)
        self.log("beta", self.get_beta(), sync_dist=True)  # 新增
        return loss

    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        pred = out.reshape((-1, self.vocab_size))
        target = val_batch.flatten().long()
        loss, nll, kld = self.loss_function(pred, target, mu, log_var, len(val_batch))

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_nll", nll, sync_dist=True)
        self.log("val_kld", kld, sync_dist=True)
        self.log("val_mu", torch.mean(mu), sync_dist=True)
        self.log("val_logvar", torch.mean(log_var), sync_dist=True)
        self.log("beta", self.get_beta(), sync_dist=True)  # 新增
        return loss
