import heapq

import numpy as np
from scipy.special import softmax


METHODS = ["eventgrad", "ipw", "plugin_only", "uncorrected", "current_gradient"]


def coordinate_jacobian(x, basis, classes):
    return np.einsum("kdr,d->kr", basis.reshape(classes, len(x), -1), x)


def make_features(logits, jacobian):
    return np.r_[1.0, logits, jacobian.ravel()]


def generate(rng, count):
    for index in range(count):
        d, classes, rank = 8, 3, 6
        weights = np.array([[rng.gauss(0, 1) for _ in range(d + 1)] for _ in range(classes)])

        def draw(n):
            x = np.array([[rng.gauss(0, 1) for _ in range(d)] + [1.0] for _ in range(n)])
            probability = softmax(x @ weights.T, axis=-1)
            y = [rng.choices(range(classes), weights=p, k=1)[0] for p in probability]
            return x, y

        warm_x, warm_y = draw(100)
        x, y = draw(200)
        propensity = [0.2 + 0.6 / (1 + np.exp(-row[0])) for row in x]
        yield {
            "id": str(index),
            "family": f"delayed-stream-{index}",
            "x": x.tolist(),
            "y": y,
            "warm_x": warm_x.tolist(),
            "warm_y": warm_y,
            "classes": classes,
            "rank": rank,
            "propensity": propensity,
            "delay": [rng.randrange(21) for _ in x],
            "selected": [rng.random() < p for p in propensity],
            "seed": rng.randrange(2**31),
            "learning_rate": 0.01,
        }


def run_case(case, method):
    x = np.asarray(case["x"])
    y = np.asarray(case["y"], dtype=int)
    warm_x, warm_y = np.asarray(case["warm_x"]), np.asarray(case["warm_y"], dtype=int)
    classes, rank = case["classes"], case["rank"]
    parameter_count = classes * x.shape[1]
    if not 0 < rank <= parameter_count:
        raise ValueError("Invalid logged subspace rank")
    pi = np.asarray(case["propensity"])
    if np.any((pi <= 0) | (pi > 1)):
        raise ValueError("Selection correction requires 0 < propensity <= 1")
    rng = np.random.default_rng(case["seed"])
    basis = np.linalg.qr(rng.normal(size=(parameter_count, rank)))[0]
    theta = np.linalg.solve(
        warm_x.T @ warm_x + np.eye(x.shape[1]), warm_x.T @ np.eye(classes)[warm_y]
    ).T
    warm_features, warm_gradients = [], []
    for row, label in zip(warm_x, warm_y):
        logits = theta @ row
        J = coordinate_jacobian(row, basis, classes)
        derivative = softmax(logits) - np.eye(classes)[label]
        warm_features.append(make_features(logits, J))
        warm_gradients.append(J.T @ derivative)
    features = np.asarray(warm_features)
    mu_model = np.linalg.solve(
        features.T @ features + np.eye(features.shape[1]), features.T @ np.asarray(warm_gradients)
    )
    pending = []
    estimate = np.zeros(rank)
    records = []
    outputs = []
    parameters_at_event = []
    max_pending_bytes = 0

    def arrive(time):
        nonlocal theta, estimate
        while pending and pending[0][0] <= time:
            _, event, record = heapq.heappop(pending)
            logits, J, propensity = record["logits"], record["jacobian"], record["propensity"]
            mu = make_features(logits, J) @ mu_model
            exact = J.T @ (softmax(logits) - np.eye(classes)[y[event]])
            if method == "eventgrad":
                correction = (exact - mu) / propensity
            elif method == "ipw":
                correction = exact / propensity
            elif method == "plugin_only":
                correction = np.zeros(rank)
            elif method == "uncorrected":
                correction = exact
            else:
                current = theta @ x[event]
                correction = coordinate_jacobian(x[event], basis, classes).T @ (
                    softmax(current) - np.eye(classes)[y[event]]
                )
            estimate += correction
            theta = (theta.ravel() - case["learning_rate"] * basis @ correction).reshape(
                theta.shape
            )
            records.append({"event": event, "arrival": time, "gradient": exact.tolist()})

    for t, row in enumerate(x):
        arrive(t - 1e-9)
        logits = theta @ row
        J = coordinate_jacobian(row, basis, classes)
        parameters_at_event.append(theta.tolist())
        outputs.append(softmax(logits).tolist())
        mu = make_features(logits, J) @ mu_model
        if method in ("eventgrad", "plugin_only"):
            estimate += mu
            theta = (theta.ravel() - case["learning_rate"] * basis @ mu).reshape(theta.shape)
        if case["selected"][t]:
            record = {"logits": logits, "jacobian": J, "propensity": pi[t]}
            heapq.heappush(pending, (t + case["delay"][t], t, record))
        max_pending_bytes = max(
            max_pending_bytes,
            sum(item[2]["logits"].nbytes + item[2]["jacobian"].nbytes + 24 for item in pending),
        )
        arrive(t)
    arrive(float("inf"))
    oracle = []
    captures = []
    reconstruction = []
    for row, label, parameter in zip(x, y, parameters_at_event):
        full = np.outer(softmax(np.asarray(parameter) @ row) - np.eye(classes)[label], row).ravel()
        J = coordinate_jacobian(row, basis, classes)
        reconstructed = J.T @ (softmax(np.asarray(parameter) @ row) - np.eye(classes)[label])
        oracle.append(basis.T @ full)
        captures.append(np.linalg.norm(basis @ reconstructed) / max(np.linalg.norm(full), 1e-12))
        reconstruction.append(np.linalg.norm(reconstructed - basis.T @ full))
    return {
        "estimated_population_gradient": (estimate / len(x)).tolist(),
        "oracle_population_gradient": np.mean(oracle, axis=0).tolist(),
        "event_probabilities": outputs,
        "record_scalars": classes * (rank + 1) + 3,
        "peak_pending_bytes": max_pending_bytes,
        "capture": float(np.mean(captures)),
        "reconstruction_error": float(np.mean(reconstruction)),
        "basis_bytes": basis.nbytes,
        "learned_parameters": theta.tolist(),
        "arrived_records": len(records),
    }


def score(case, result):
    estimate = np.asarray(result["estimated_population_gradient"])
    truth = np.asarray(result["oracle_population_gradient"])
    probability = np.asarray(result["event_probabilities"])
    labels = np.asarray(case["y"])
    return {
        "gradient_error": float(np.linalg.norm(estimate - truth)),
        "relative_gradient_error": float(
            np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1e-12)
        ),
        "gradient_cosine": float(
            estimate @ truth / max(np.linalg.norm(estimate) * np.linalg.norm(truth), 1e-12)
        ),
        "reconstruction_error": result["reconstruction_error"],
        "full_gradient_capture": result["capture"],
        "record_scalars": result["record_scalars"],
        "peak_pending_bytes": result["peak_pending_bytes"],
        "basis_bytes": result["basis_bytes"],
        "event_accuracy": float(np.mean(probability.argmax(-1) == labels)),
        "event_nll": float(
            -np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-12)).mean()
        ),
        "legal_effects": len(labels),
    }
