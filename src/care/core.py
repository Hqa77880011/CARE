from itertools import combinations, product
import numpy as np


def joint_probability(policy):
    shape = tuple(len(p) for p in policy)
    result = np.ones(shape)
    for i, p in enumerate(policy):
        view = [1] * len(policy)
        view[i] = len(p)
        result *= np.asarray(p).reshape(view)
    return result


def risk(policy, risks):
    return np.sum(
        np.asarray(risks) * joint_probability(policy),
        axis=tuple(range(1, np.asarray(risks).ndim)),
    )


def certificate(policy, risks, allowed, alpha, point=False):
    models = np.asarray(risks[:1] if point else risks)
    n = len(policy)
    checks = []
    for size in range(n + 1):
        for subset in combinations(range(n), size):
            for action in product(*(allowed[i] for i in subset)):
                candidate = [np.asarray(p).copy() for p in policy]
                for i, a in zip(subset, action):
                    candidate[i] = np.eye(len(policy[i]))[a]
                values = risk(candidate, models)
                checks.append(
                    dict(
                        subset=list(subset), actions=list(action), risk=values.tolist()
                    )
                )
                if max(values) <= alpha:
                    return dict(
                        subset=list(subset),
                        actions=list(action),
                        worst_risk=float(max(values)),
                        alpha=alpha,
                        model_ids=list(range(len(models))),
                        cardinality_lower_bound=size,
                        verification=checks,
                    ), candidate
    return None, None


def update(policy, subset, witness, risks, steps, lr):
    import torch

    logits = [
        torch.tensor(
            np.log(np.maximum(p, 1e-12)), dtype=torch.float64, requires_grad=i in subset
        )
        for i, p in enumerate(policy)
    ]
    if not subset:
        return [p.copy() for p in policy]
    optimizer = torch.optim.Adam([logits[i] for i in subset], lr=lr)
    rr = torch.tensor(risks, dtype=torch.float64)
    for _ in range(steps):
        joint = torch.ones(rr.shape[1:], dtype=torch.float64)
        for i, z in enumerate(logits):
            view = [1] * len(logits)
            view[i] = len(z)
            joint = joint * z.softmax(0).view(view)
        loss = (rr * joint).flatten(1).sum(1).max()
        if witness is not None:
            loss += 0.1 * sum(
                -(torch.tensor(witness[i]) * logits[i].log_softmax(0)).sum()
                for i in subset
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return [z.detach().softmax(0).numpy() for z in logits]
