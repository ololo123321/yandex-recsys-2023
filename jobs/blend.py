import os
import argparse
import tqdm
import numpy as np


def prob2score(x):
    """
    перевод вероятностей в логиты
    """
    eps = 1e-6
    p = np.clip(x, eps, 1 - eps)
    return np.log(p / (1 - p))


def score2prob(x):
    return 1 / (1 + np.exp(-x))


def main(args):
    # load tracks
    tracks = []
    with open(args.data_path) as f:
        next(f)  # skip header
        for line in f:
            tracks.append(line.strip())
    tracks = np.array([int(x) for x in tracks])

    # load preds
    y_pred = np.zeros((len(tracks), 256))
    num_models = 0
    with open(args.config_path) as f:
        next(f)  # skip header
        for line in tqdm.tqdm(f):
            row = line.strip().split()  # name step1 step2 step3 weight
            for i in [1, 2, 3]:
                x = np.load(os.path.join(args.predictions_dir, f"fold{i}", f"{row[0]}__step{row[i]}.npz"))
                assert np.all(x["tracks"] == tracks)  # sanity check
                y_pred += prob2score(x["probs"]) * float(row[4]) * 1/3
            num_models += 1
    # * это лишняя операция, если предикты идут в sklearn.metrics.average_precision_score.
    #   но по правилам турнира нужны были именно вероятности, поэтому для порядка сделал.
    # * деление на num_models для numerical stability
    y_pred = score2prob(y_pred / num_models)
    assert not np.isinf(y_pred).any()
    assert not np.isnan(y_pred).any()

    # save preds
    with open(args.output_path, "w") as f:
        f.write("track,prediction\n")
        for i in tqdm.trange(len(tracks)):
            f.write(str(tracks[i]) + "," + ",".join(map(str, y_pred[i])) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")  # .csv file with track ids. has header
    parser.add_argument("--config_path")  # .tsv file. header=name step1 step2 step3 weight
    parser.add_argument("--predictions_dir")  # .npz files with keys "tracks", "probs" grouped by fold
    parser.add_argument("--output_path")
    args_ = parser.parse_args()
    main(args_)
