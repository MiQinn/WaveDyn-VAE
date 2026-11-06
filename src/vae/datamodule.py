import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L

import os
import re
from pathlib import Path
from attrs import define, field

NOP = "[nop]"


@define
class MolDataset(Dataset):
    dataset: list[list[int]]
    smiles: list[str]
    vocab: np.ndarray[str]
    max_len: int

    token_to_id: dict[str, int] = field(factory=dict)
    id_to_token: dict[int, str] = field(factory=dict)

    def __attrs_post_init__(self):
        if not self.token_to_id:
            self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        if not self.id_to_token:
            self.id_to_token = {i: t for i, t in enumerate(self.vocab)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.int64)

    def size(self, idx: int):
        return len(self.dataset) if idx == 0 else self.max_len

    def __getstate__(self):
        return {
            'dataset': self.dataset,
            'smiles': self.smiles,
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'max_len': self.max_len,
        }

    def __setstate__(self, state):
        self.dataset = state['dataset']
        self.smiles = state['smiles']
        self.vocab = state['vocab']
        self.token_to_id = state['token_to_id']
        self.id_to_token = state['id_to_token']
        self.max_len = state['max_len']


def tokenize_smiles(s: str) -> list[str]:
    """
    SMILES tokenizer：
    - 保留 *, 数字与前一个原子结合 (环标记)
    - 支持多字符元素（Cl, Br, Si, Se, Na, Ca）
    - 保留括号、键符号、+, - 等
    - 增加约束：* 去除非法开头/结尾   * 平衡括号   * 过滤孤立数字   * 检查环标号闭合性
    """
    # 1. 基础 token 化
    pattern = r"\[[^\[\]]+?\]|Br|Cl|Si|Se|Na|Ca|@@?|=|#|\(|\)|\.|\+|\-|\d|[A-Za-z*]"
    raw_tokens = re.findall(pattern, s)

    if not raw_tokens:
        return []

    # 2. 数字与前一个原子或 * 合并；孤立数字丢弃
    merged = []
    for token in raw_tokens:
        if token.isdigit():
            if merged and re.match(r"^[A-Za-z*]$|^\[.*\]$", merged[-1]):
                merged[-1] = merged[-1] + token
            else:
                continue  # 孤立数字直接丢弃
        else:
            merged.append(token)

    if not merged:
        return []

    # 3. 去除非法开头/结尾（仅允许原子、*、[原子]、数字组合，不允许 ),(,=,#,-,. 等）
    while merged and re.match(r"^[\)\(\=\#\-\.\+]+$", merged[0]):
        merged.pop(0)
    while merged and re.match(r"^[\)\(\=\#\-\.\+]+$", merged[-1]):
        merged.pop()

    if not merged:
        return []

    # 4. 检查括号平衡
    balanced = []
    stack = 0
    for tok in merged:
        if tok == "(":
            stack += 1
            balanced.append(tok)
        elif tok == ")":
            if stack > 0:
                stack -= 1
                balanced.append(tok)
            else:
                # 多余的右括号丢弃
                continue
        else:
            balanced.append(tok)
    # 去掉多余的左括号（无法配对的）
    while stack > 0 and "(" in balanced:
        # 从右向左删除第一个未配对的 "("
        idx = len(balanced) - 1 - balanced[::-1].index("(")
        balanced.pop(idx)
        stack -= 1

    if not balanced:
        return []

    # 5. 检查环标号闭合：每个数字必须出现偶数次
    ring_numbers = [re.findall(r"\d+", t) for t in balanced]
    flat_numbers = [n for sub in ring_numbers for n in sub]
    from collections import Counter
    cnt = Counter(flat_numbers)
    if any(v % 2 != 0 for v in cnt.values()):
        return []  # 有未闭合环则直接丢弃

    return balanced


def make_dataset(file: Path, length_limit: int = 128) -> MolDataset:
    smiles_df = pd.read_csv(file, header=None, names=["smiles"])
    print(f"Loaded {len(smiles_df)} SMILES strings from {file}")
    print(f"Example SMILES: {smiles_df['smiles'].sample(5).tolist()}")

    dataset = []
    smiles = []
    vocab_set = set()

    for s in smiles_df["smiles"]:
        try:
            tokens = tokenize_smiles(s)
            if len(tokens) <= length_limit:
                vocab_set.update(tokens)
                smiles.append(s)
                dataset.append(tokens)
        except Exception:
            continue

    max_len = max(len(t) for t in dataset)
    print(f"Max SMILES length: {max_len}")

    vocab = np.array([NOP] + sorted(vocab_set))
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for t, i in token_to_id.items()}

    dataset_ids = np.full((len(dataset), max_len), token_to_id[NOP], dtype=np.int64)
    for i, tok_list in enumerate(dataset):
        dataset_ids[i, :len(tok_list)] = [token_to_id[t] for t in tok_list]

    return MolDataset(
        dataset=dataset_ids.tolist(),
        smiles=smiles,
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        max_len=max_len,
    )


class MolDataModule(L.LightningDataModule):
    dataset: MolDataset
    train_data: MolDataset
    test_data: MolDataset

    max_len: int
    vocab_size: int

    def __init__(
        self,
        file: str = "data/processed/osimertinib/osimertinib_sca_screened.smi",
        batch_size: int = 64,
        length_limit: int = 72,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.file = Path(file)
        self.name = self.file.stem
        self.length_limit = length_limit
        self.path = Path("data/processed/osimertinib") / f"{self.name}_data.pt"

    def prepare_data(self) -> None:
        try:
            dataset = torch.load(self.path)
            print(f"Loaded tokenized dataset from {self.path}")
        except Exception as err:
            print(f"Failed to load tokenized dataset from {self.path}: {err}")
            print(f"Tokenizing dataset and saving to {self.path}")
            dataset = make_dataset(self.file, self.length_limit)
            torch.save(dataset, self.path)

    def setup(self, stage: str = None) -> None:
        self.dataset = torch.load(self.path)
        self.max_len = self.dataset.max_len
        self.vocab_size = len(self.dataset.vocab)

        self.train_data, self.test_data = random_split(
            self.dataset,
            [int(len(self.dataset) * 0.9), len(self.dataset) - int(len(self.dataset) * 0.9)],
            generator=torch.Generator().manual_seed(42),
        )

    def __len__(self):
        return len(self.dataset)

    def encode(self, x: str | list[str]) -> Tensor:
        if isinstance(x, str):
            x = [x]
        results = torch.zeros((len(x), self.max_len), dtype=torch.int32)
        for i, s in enumerate(x):
            tokens = [self.dataset.token_to_id.get(t, 0) for t in tokenize_smiles(s)]
            results[i, : len(tokens)] = torch.tensor(tokens)
        return results if len(x) > 1 else results.squeeze(0)

    def decode(self, x: Tensor, progress_bar: bool = False) -> str | list[str]:
        token_ids = x.argmax(-1) if x.ndim == 3 else x
        smiles = []
        for row in token_ids:
            chars = [self.dataset.id_to_token.get(i.item(), NOP) for i in row if i.item() != 0]
            smiles.append("".join(chars))
        return smiles if x.ndim > 1 else smiles[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=4,
        )


if __name__ == "__main__":
    from cd2root import cd2root

    cd2root()

    dm = MolDataModule(file="data/processed/osimertinib/osimertinib_sca_screened.smi")
    dm.prepare_data()
    dm.setup()

    print("Vocab size:", dm.vocab_size)
    print("Max length:", dm.max_len)

    for x in dm.train_dataloader():
        print(x)
        break

    print("Encode:", dm.encode("CCO"))
    print("Decode:", dm.decode(dm.encode("CCO")))

    x = torch.randn(dm.max_len * dm.vocab_size)
    print(dm.decode(x))

    print(len(dm))
    print(dm.encode(["C", "CC", "CCC"]))
