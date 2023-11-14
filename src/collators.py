import torch


class TrainingCollator:
    def __init__(self, num_classes: int, max_length: int = 150):
        self.num_classes = num_classes
        self.max_length = max_length

    def __call__(self, batch):
        """
        x - [T, D]
        labels - [num_classes]
        """
        N = len(batch)
        T = min(self.max_length, max(x[0].shape[0] for x in batch))
        D = batch[0][0].shape[1]

        x = torch.zeros((N, T, D))
        y = torch.zeros((N, self.num_classes))

        mask = torch.zeros((N, T)).long()
        for i in range(N):
            xi, yi = batch[i]
            t = min(self.max_length, xi.shape[0])
            x[i, :t, :] = torch.tensor(xi[:t])
            y[i, yi] = 1
            mask[i, :t] = 1

        return {"x": x, "mask": mask, "labels": y}


class InferenceCollator:
    def __init__(self, max_length: int = 150):
        self.max_length = max_length

    def __call__(self, batch):
        """
        x - [T, D]
        """
        N = len(batch)
        T = min(self.max_length, max(x.shape[0] for x in batch))
        D = batch[0].shape[1]

        x = torch.zeros((N, T, D))

        mask = torch.zeros((N, T)).long()
        for i in range(N):
            xi = batch[i]
            t = min(self.max_length, xi.shape[0])
            x[i, :t, :] = torch.tensor(xi[:t])
            mask[i, :t] = 1

        return {"x": x, "mask": mask}
