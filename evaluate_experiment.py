#sources: 
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2
# https://huggingface.co/laion/clap-htsat-unfused

# imports for files, json data, hashing, math, plots, and models
import argparse
import json
import hashlib
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import ClapModel, ClapProcessor


# set the fixed models and evaluation settings used in the experiment
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
CLAP_MODEL = "laion/clap-htsat-unfused"
SEGMENT_SECONDS = 10.0
AUDIO_BATCH_SIZE = 4
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 2026


# make a file hash so two audio files can be checked for exact equality
def file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    # read the file in chunks so large audio files do not need to be loaded at once
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# load a json file from disk
def read_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


# split the full artist dna string into lyrical dna and acoustic dna
def split_dna(dna: str) -> tuple[str, str]:
    try:
        lyrical, acoustic = dna.split("[LYRICAL DNA]", 1)[1].split("[ACOUSTIC DNA]", 1)
    except ValueError as e:
        raise ValueError("Artist DNA needs [LYRICAL DNA] and [ACOUSTIC DNA].") from e

    return lyrical.strip(), acoustic.strip()


# find an audio file from a stored path or from the local song folder
def audio_path(raw: str | None, folder: Path, name: str | None = None) -> Path:
    candidates = []

    # try the stored path, the current working directory, and the local folder
    if raw:
        path = Path(raw)
        candidates += [path, Path.cwd() / path, folder / path.name]

    # also try the final audio file name if it is given separately
    if name:
        candidates.append(folder / name)

    # return the first path that really exists
    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"Audio not found: raw={raw!r}, name={name!r}")


# load all packages and their single prompt and mas result files and check existence
def load_experiment(exp: Path):
    package_file = exp / "initial_packages.json"

    # stop if the main package file is missing
    if not package_file.is_file():
        raise FileNotFoundError(package_file)

    # keep the original packages by package id for later validation
    packages = {p["package_id"]: p for p in read_json(package_file)}
    records = []

    # collect one record for each package and condition
    for package_id, package in packages.items():
        for condition in ("single_prompt", "mas"):
            folder = exp / condition / package_id
            result_file = folder / "result.json"

            # stop if one result file is missing
            if not result_file.is_file():
                raise FileNotFoundError(result_file)

            # load the final output fields and store the needed metadata
            result = read_json(result_file)
            final = result["final_output"]
            records.append({
                "package_id": package_id,
                "condition": condition,
                "topic": package["topic"],
                "seed": package["seed"],
                "final_lyrics": final["lyrics"],
                "final_audio": audio_path(None, folder, final["audio_file_name"]),
                "lyrics_critic": result.get("lyrics_critic", {}),
                "audio_critic": result.get("audio_critic", {}),
                "folder": folder,
                "song_revision_count": result.get("song_revision_count", 0),
            })

    return records, packages


# make one sentence transformer embedding and reuse cached results
def text_embedding(text, model, cache):
    text = text.strip()

    # only encode the same text once, already normalized
    if text not in cache:
        cache[text] = np.asarray(model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)

    return cache[text]


# get the tensor from possible clap output formats
def extract_clap_tensor(output):
    # some model calls already return a tensor
    if torch.is_tensor(output):
        return output

    # some model calls return an object with one of these attributes
    for key in ("pooler_output", "text_embeds", "audio_embeds"):
        value = getattr(output, key, None)
        if torch.is_tensor(value):
            return value

    # some model calls return a tuple or list
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]

    raise TypeError(f"Unsupported CLAP output: {type(output)!r}")


# normalize one vector to length one
def normalize_vector(vector):
    
    # calculate the L2 norm of the vector
    norm = np.linalg.norm(vector)

    # avoid division by zero
    if norm == 0:
        return vector

    return vector / norm


# make a clap text embedding for the acoustic dna
def clap_text_embedding(text, processor, model, dev, cache):
    text = text.strip()

    # only encode the same acoustic dna text once
    if text not in cache:
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(dev)

        # no gradients are needed because this is only evaluation
        with torch.no_grad():
            vector = extract_clap_tensor(model.get_text_features(**inputs))[0]

        cache[text] = normalize_vector(vector.cpu().float().numpy())

    return cache[text]


# make one song-level clap audio embedding
def clap_audio_embedding(path, processor, model, dev, cache):
    key = str(path.resolve())

    # only encode the same audio file once
    if key in cache:
        return cache[key]

    # load audio as mono 48 khz signal
    audio, _ = librosa.load(path, sr=48000, mono=True)
    size = max(1, int(SEGMENT_SECONDS * 48000))

    # split the song into fixed-length segments and ignore very short ending pieces
    segments = [audio[i:i + size] for i in range(0, len(audio), size) if len(audio[i:i + size]) >= 48000]

    # keep very short files evaluable
    if not segments:
        segments = [audio]

    vectors = []

    # encode the audio segments in small batches
    for i in range(0, len(segments), AUDIO_BATCH_SIZE):
        inputs = processor(audio=segments[i:i + AUDIO_BATCH_SIZE], sampling_rate=48000, return_tensors="pt", padding=True).to(dev)

        # no gradients are needed because this is only evaluation
        with torch.no_grad():
            vector = extract_clap_tensor(model.get_audio_features(**inputs))

        vectors.append(vector.cpu().float().numpy())

    # normalize segment vectors, average them, and normalize the final song vector
    vectors = np.concatenate(vectors)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    cache[key] = normalize_vector(vectors.mean(axis=0))

    return cache[key]


# check that the experiment data matches the controlled paired design
def validate(records, packages, config):
    # check that the number of packages matches the config
    if len(packages) != config["song_count"]:
        raise ValueError(f"Expected {config['song_count']} packages, found {len(packages)}.")
    pairs = {}

    # group records by package id so each package can be checked as a pair
    for r in records:
        pairs.setdefault(r["package_id"], {})[r["condition"]] = r

    # check every single prompt and mas pair
    for pid, pair in pairs.items():
        if set(pair) != {"single_prompt", "mas"}:
            raise ValueError(f"Incomplete pair: {pid}")

        package, single, mas = packages[pid], pair["single_prompt"], pair["mas"]

        # single prompt lyrics must still be the frozen initial lyrics
        if single["final_lyrics"].strip() != package["lyrics"].strip():
            raise ValueError(f"Single lyrics differ from initial package: {pid}")

        # both conditions must use the same package seed
        if single["seed"] != mas["seed"] or single["seed"] != package["seed"]:
            raise ValueError(f"Seed mismatch: {pid}")

        # the mas lyrics history should start from version 0 lyrics
        lyrics_history = mas["lyrics_critic"].get("history", [])
        if not lyrics_history:
            raise ValueError(f"Missing lyrics history: {pid}")
        if lyrics_history[0]["lyrics"].strip() != package["lyrics"].strip():
            raise ValueError(f"Lyrics history does not start at version 0: {pid}")

        # the mas audio history should start from the original audio prompt
        audio_history = mas["audio_critic"].get("history", [])
        if not audio_history:
            raise ValueError(f"Missing audio history: {pid}")

        first = " ".join(audio_history[0]["audio_prompt"].split())
        initial = " ".join(package["audio_prompt"].split())

        if first != initial:
            raise ValueError(f"Audio history does not start at version 0: {pid}")

        mas_initial_audio = audio_path(audio_history[0].get("local_path"), mas["folder"])

        if file_hash(single["final_audio"]) != file_hash(mas_initial_audio):
            raise ValueError(f"Single Prompt audio and MAS version 0 audio are not identical: {pid}")

    print(f"Validated {len(packages)} paired song packages.")


# compute cosine similarity for already normalized vectors with L2 norm of 1
def cosine(a, b):
    return float(np.dot(a, b))


# compute final lyrics and audio dna similarity for both conditions
def final_metrics(records, lyrical_dna, acoustic_dna, text_model, text_cache, clap_processor, clap_model, dev, clap_text_cache, clap_audio_cache):
    # encode the dna target texts
    lyric_dna = text_embedding(lyrical_dna, text_model, text_cache)
    audio_dna = clap_text_embedding(acoustic_dna, clap_processor, clap_model, dev, clap_text_cache)

    rows, lyric_vectors, audio_vectors = [], {}, {}

    # score every final output against the matching dna part
    for r in records:
        key = (r["package_id"], r["condition"])
        lyric_vector = text_embedding(r["final_lyrics"], text_model, text_cache)
        audio_vector = clap_audio_embedding(r["final_audio"], clap_processor, clap_model, dev, clap_audio_cache)

        # store vectors for later intra-group similarity
        lyric_vectors[key] = lyric_vector
        audio_vectors[key] = audio_vector

        rows.append({
            "package_id": r["package_id"],
            "condition": r["condition"],
            "topic": r["topic"],
            "lyrics_dna_cosine": cosine(lyric_vector, lyric_dna),
            "audio_dna_cosine": cosine(audio_vector, audio_dna),
            "lyrics_revisions": r["lyrics_critic"].get("revision_count", 0),
            "audio_revisions": r["audio_critic"].get("revision_count", 0),
        })

    # reshape from long format to one paired row per song package
    long = pd.DataFrame(rows)
    wide = long.pivot(index=["package_id", "topic"], columns="condition").reset_index()
    wide.columns = [a if not b else f"{a}_{b}" for a, b in wide.columns]
    wide = wide.rename(columns={"lyrics_dna_cosine_single_prompt": "lyrics_dna_cosine_single", "audio_dna_cosine_single_prompt": "audio_dna_cosine_single"})

    # compute the paired mas minus single prompt differences
    wide["lyrics_delta"] = wide["lyrics_dna_cosine_mas"] - wide["lyrics_dna_cosine_single"]
    wide["audio_delta"] = wide["audio_dna_cosine_mas"] - wide["audio_dna_cosine_single"]

    return wide, lyric_vectors, audio_vectors


# compute version-level similarity scores for the mas refinement histories
def loop_metrics(records, lyrical_dna, acoustic_dna, text_model, text_cache, clap_processor, clap_model, dev, clap_text_cache, clap_audio_cache):
    # encode the dna target texts
    lyric_dna = text_embedding(lyrical_dna, text_model, text_cache)
    audio_dna = clap_text_embedding(acoustic_dna, clap_processor, clap_model, dev, clap_text_cache)

    lyrics, audios = [], []

    # process only mas as only mas records have revision histories
    for r in records:
        if r["condition"] != "mas":
            continue

        # score every stored lyrics version and its step change
        previous = None
        for e in r["lyrics_critic"].get("history", []):
            score = cosine(text_embedding(e["lyrics"], text_model, text_cache), lyric_dna)

            lyrics.append({
                "package_id": r["package_id"],
                "version": int(e["version"]),
                "decision": e["decision"],
                "dna_similarity": score,
                "delta_from_previous": np.nan if previous is None else score - previous,
            })

            previous = score

        # score every stored audio version and its step change
        previous_score = None
        for e in r["audio_critic"].get("history", []):
            path = audio_path(e.get("local_path"), r["folder"])
            score = cosine(clap_audio_embedding(path, clap_processor, clap_model, dev, clap_audio_cache), audio_dna)

            audios.append({
                "package_id": r["package_id"],
                "version": int(e["version"]),
                "decision": e["decision"],
                "dna_similarity": score,
                "delta_from_previous": np.nan if previous_score is None else score - previous_score,
            })

            previous_score = score

    return pd.DataFrame(lyrics), pd.DataFrame(audios)


# estimate a bootstrap confidence interval for the mean
def bootstrap(values, n, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    # bootstrap the mean by sampling with same size and replacement, computing the mean of each sample
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n)]
    # return 95% confidence interval for the mean (2.5th and 97.5th percentiles of the bootstrap means)
    return np.quantile(means, [0.025, 0.975])


# summarize final mas versus single prompt scores for one metric
def paired_summary(metrics, name, n_boot):
    single = metrics[f"{name}_single"].to_numpy(float)
    mas = metrics[f"{name}_mas"].to_numpy(float)

    # use paired differences because both conditions share the same starting package
    diff = mas - single
    low, high = bootstrap(diff, n_boot)

    return {
        "metric": name,
        "n": len(diff),
        "single_mean": single.mean(),
        "mas_mean": mas.mean(),
        "mean_difference": diff.mean(),
        "median_difference": np.median(diff),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "mas_better_count": int((diff > 0).sum()),
        "single_better_count": int((diff < 0).sum()),
        "tie_count": int(np.isclose(diff, 0).sum()),
    }


# summarize the mas revision process for lyrics and audio
def revision_summary(records, loops, config):
    mas = [r for r in records if r["condition"] == "mas"]
    rows = []

    for modality, loop in loops.items():
        critic = f"{modality}_critic"

        # collect final critic decisions and revision counts
        decisions = [r[critic].get("decision") for r in mas]
        revisions = np.array([r[critic].get("revision_count", 0) for r in mas])

        # check whether the whole mas system ended with both critics approving
        system_approved = np.array([r["lyrics_critic"].get("decision") == "APPROVED" and r["audio_critic"].get("decision") == "APPROVED" for r in mas])

        # check whether the shared song revision limit was reached
        limit_reached = np.array([r.get("song_revision_count", 0) >= config["max_song_revisions"] for r in mas])

        gains, final_best, regret = [], [], []

        # compare the first, final, and best observed version per song package
        for _, group in loop.groupby("package_id"):
            group = group.sort_values("version")
            initial = group.iloc[0].dna_similarity
            final = group.iloc[-1].dna_similarity
            best = group.dna_similarity.max()

            gains.append(final - initial)
            final_best.append(np.isclose(final, best))
            regret.append(best - final)

        # collect all step changes between consecutive versions
        steps = loop.delta_from_previous.dropna().to_numpy(float)

        rows.append({
            "modality": modality,
            "songs": len(mas),
            "approval_rate": np.mean([d == "APPROVED" for d in decisions]),
            "song_limit_reached_rate": np.mean(limit_reached & ~system_approved),
            "mean_revisions": revisions.mean(),
            "mean_initial_to_final_gain": np.mean(gains),
            "songs_improved_fraction": np.mean(np.array(gains) > 0),
            "positive_step_fraction": np.mean(steps > 0) if len(steps) else np.nan,
            "negative_step_fraction": np.mean(steps < 0) if len(steps) else np.nan,
            "final_is_best_fraction": np.mean(final_best),
            "mean_best_minus_final": np.mean(regret),
        })

    return pd.DataFrame(rows)


# compute mean pairwise similarity inside each condition group
def intragroup(vectors):
    rows = []

    # compare all final songs within each modality and condition
    for modality, store in vectors.items():
        for condition in ("single_prompt", "mas"):
            selected = [v for (_, current_condition), v in store.items() if current_condition == condition]
            matrix = np.vstack(selected) @ np.vstack(selected).T
            values = matrix[np.triu_indices(len(selected), 1)]

            rows.append({
                "modality": modality,
                "condition": condition,
                "n": len(selected),
                "mean_pairwise_cosine": values.mean() if len(values) else np.nan,
            })

    return pd.DataFrame(rows)


# save the plots used to inspect final differences and revision paths
def plots(metrics, loops, folder):
    folder.mkdir(parents=True, exist_ok=True)

    for modality in ("lyrics", "audio"):
        # plot paired mas minus single prompt differences per package
        plt.figure(figsize=(9, 4.5))
        plt.axhline(0, linewidth=1)
        plt.bar(metrics.package_id, metrics[f"{modality}_delta"])
        plt.title(f"{modality.title()}: MAS − Single Prompt")
        plt.ylabel("Cosine difference")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(folder / f"{modality}_paired_differences.png", dpi=200)
        plt.close()

        # plot all mas version trajectories plus the mean trajectory
        loop = loops[modality]
        plt.figure(figsize=(9, 5))

        for _, group in loop.groupby("package_id"):
            group = group.sort_values("version")
            plt.plot(group.version, group.dna_similarity, marker="o", alpha=0.5)

        mean = loop.groupby("version").dna_similarity.mean()
        plt.plot(mean.index, mean.values, marker="o", linewidth=3, label="Mean")

        plt.title(f"{modality.title()} Persona Development")
        plt.xlabel("Version")
        plt.ylabel("DNA cosine")
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder / f"{modality}_revision_trajectories.png", dpi=200)
        plt.close()


# run the full evaluation pipeline
def main():
    # read the experiment folder from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    exp = parser.parse_args().experiment_dir.resolve()

    if not exp.is_dir():
        raise NotADirectoryError(exp)

    # load the fixed experiment config and split the artist dna
    config = read_json(exp / "config.json")
    lyrical_dna, acoustic_dna = split_dna(config["artist_dna"])

    # load and validate the saved experiment outputs
    records, packages = load_experiment(exp)
    validate(records, packages, config)

    # choose gpu when available, otherwise use cpu
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    # load the text and audio encoders
    text_model = SentenceTransformer(TEXT_MODEL, device=dev)
    text_cache = {}

    clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    clap_model = ClapModel.from_pretrained(CLAP_MODEL).to(dev).eval()
    clap_text_cache = {}
    clap_audio_cache = {}

    # compute final output scores and revision history scores
    metrics, lyric_vectors, audio_vectors = final_metrics(records, lyrical_dna, acoustic_dna, text_model, text_cache, clap_processor, clap_model, dev, clap_text_cache, clap_audio_cache)
    lyrics_loop, audio_loop = loop_metrics(records, lyrical_dna, acoustic_dna, text_model, text_cache, clap_processor, clap_model, dev, clap_text_cache, clap_audio_cache)
    loops = {"lyrics": lyrics_loop, "audio": audio_loop}

    # compute the summary tables used in the thesis
    summaries = pd.DataFrame([paired_summary(metrics, "lyrics_dna_cosine", BOOTSTRAP_SAMPLES), paired_summary(metrics, "audio_dna_cosine", BOOTSTRAP_SAMPLES)])
    revisions = revision_summary(records, loops, config)
    intra = intragroup({"lyrics": lyric_vectors, "audio": audio_vectors})

    # create the output folder for all evaluation files
    out = exp / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    # save the numeric result tables as csv files
    metrics.to_csv(out / "metrics_per_song.csv", index=False)
    summaries.to_csv(out / "paired_summary.csv", index=False)
    lyrics_loop.to_csv(out / "lyrics_loop.csv", index=False)
    audio_loop.to_csv(out / "audio_loop.csv", index=False)
    revisions.to_csv(out / "revision_summary.csv", index=False)
    intra.to_csv(out / "intragroup_similarity.csv", index=False)

    # save the plot images
    plots(metrics, loops, out / "plots")

    # save a compact json summary for quick inspection
    with (out / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump({
            "experiment": exp.name,
            "song_pairs": len(metrics),
            "paired_summary": summaries.to_dict("records"),
            "revision_summary": revisions.to_dict("records"),
            "intragroup_similarity": intra.to_dict("records"),
        }, file, indent=2)

    print(f"Evaluation completed: {out}")


# start the script when it is run directly
if __name__ == "__main__":
    main()