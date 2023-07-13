from torch.utils.data import Dataset, DataLoader
import torch


class GPReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        #
        # encoding = self.tokenizer.encode_plus(
        #     review,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     return_token_type_ids=False,
        #     pad_to_max_length=True,
        #     return_attention_mask=True,
        #     return_tensors='pt',
        # )

        return review, torch.tensor(target, dtype=torch.long)
