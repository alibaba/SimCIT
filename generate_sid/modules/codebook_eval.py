import torch


class codebook_evaluator:

    @staticmethod
    def compute_perplexity(quantized_indices, codebook_size=128):
        bsz, n_codebooks = quantized_indices.shape
        perplexity = 0.
        for i in range(n_codebooks):
            encoding_indices = quantized_indices[:, i:i+1]
            encodings = torch.zeros(encoding_indices.shape[0], codebook_size).to(quantized_indices.device)
            encodings.scatter_(1, encoding_indices, 1)
            avg_probs = torch.mean(encodings, dim=0)
            perplexity += torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    @staticmethod
    def compute_collision(quantized_indices):
        total_items = quantized_indices.shape[0]
        total_unique_items = torch.unique(quantized_indices, dim=0).shape[0]
        print(f"total_items: {total_items}, total_unique_items: {total_unique_items}")
        return total_unique_items / total_items
