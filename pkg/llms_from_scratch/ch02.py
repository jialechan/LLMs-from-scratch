# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# 一个用于批处理输入和目标的数据集类
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Args:
            txt (str): 需要被tokenize的文本
            tokenizer (tiktoken.Encoding): 用于tokenize文本的tokenizer
            max_length (int): 每个token sequence的最大长度。简单来说比如一个文章有1000个token，max_length=256，那么就会有4个token sequence，每个sequence有256个token
            stride (int): 滑动窗口的步长
        """
        self.tokenizer = tokenizer

        # input_ids 是将txt词元化之后，按max_length和stride分后的token sequence
        self.input_ids = []
        # target_ids 是input_ids向右滑动一个token后的结果
        self.target_ids = []

        # 将整个文本词元化
        # 允许特殊token <|endoftext|>
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分成重叠的max_length长度的token sequence
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        返回数据集的长度，即token sequence的数量
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        返回数据集的指定行
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):

    # 初始化tokenizer为gpt2
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, # 如果drop_last为True，那么最后一个batch如果不足batch_size个样本，就会被丢弃，以防止在训练期间出现损失剧增
        num_workers=num_workers # 用于加载数据的子进程数量，0表示在主进程中加载数据
    )

    return dataloader
