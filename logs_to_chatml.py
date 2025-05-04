#!/usr/bin/env python3
import os
import json
import argparse

from datasets import Dataset


def format_conversation(prompt: str, response: str) -> str:
    """
    Wrap a single prompt/response turn into the OpenAssistant style conversation string:
      <im_start>user
      {prompt}
      <im_end>
      <im_start>assistant
      {response}
      <im_end>
    """
    return (
        "<im_start>user\n"
        f"{prompt}"
        "<im_end>\n"
        "<im_start>assistant\n"
        f"{response}"
        "<im_end>"
    )


# def process_logs(logs_dir: str):
#     werewolf_examples = []
#     villager_examples = []

#     # iterate over every .json file in logs_dir
#     for fn in os.listdir(logs_dir):
#         if not fn.lower().endswith(".json"):
#             continue
#         path = os.path.join(logs_dir, fn)
#         with open(path, "r") as f:
#             games = json.load(f)

#         for game in games:
#             # 1) eliminate → always Tyler/Jacob werewolf turn
#             elim = game.get("eliminate")
#             if elim and "prompt" in elim and "raw_resp" in elim:
#                 txt = format_conversation(elim["prompt"], elim["raw_resp"])
#                 werewolf_examples.append({"text": txt})

#             # 2) bids → look for “the Villager” in prompt
#             for round_actions in game.get("bid", []):
#                 for player_name, entry in round_actions:
#                     prompt = entry.get("prompt", "")
#                     raw = entry.get("raw_resp", "")
#                     if "the Villager" in prompt and prompt and raw:
#                         txt = format_conversation(prompt, raw)
#                         villager_examples.append({"text": txt})

#     return werewolf_examples, villager_examples

def load_games(path: str):
    """
    Load either:
      - a JSON list of objects
      - a single JSON object
      - a JSONL file (one JSON object per line)
    and return a flat List[dict].
    """
    with open(path, "r") as f:
        raw = f.read().strip()

    # Case A: proper JSON list
    if raw.startswith("[") and raw.endswith("]"):
        data = json.loads(raw)
        if isinstance(data, list):
            return data

    # Case B: single JSON object
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Case C: JSONL (one JSON object per line)
    games = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            if isinstance(o, dict):
                games.append(o)
        except json.JSONDecodeError:
            # skip any malformed line
            continue

    return games


def process_logs(logs_dir: str):
    werewolf_examples = []
    villager_examples = []

    for fn in os.listdir(logs_dir):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(logs_dir, fn)
        games = load_games(path)

        for game in games:
            # skip anything that still isn’t a dict
            if not isinstance(game, dict):
                continue

            # --- Werewolf: "eliminate" turn ---
            elim = game.get("eliminate")
            if isinstance(elim, dict) and elim.get("prompt") and elim.get("raw_resp"):
                text = format_conversation(elim["prompt"], elim["raw_resp"])
                werewolf_examples.append({"text": text})

            # --- Villager: any bid mentioning "the Villager" ---
            for round_actions in game.get("bid", []):
                # round_actions might be a list of [player_name, entry] pairs
                for player_name, entry in round_actions:
                    prompt = entry.get("prompt", "")
                    raw   = entry.get("raw_resp", "")
                    if "the Villager" in prompt and prompt and raw:
                        text = format_conversation(prompt, raw)
                        villager_examples.append({"text": text})

    return werewolf_examples, villager_examples

def main():
    parser = argparse.ArgumentParser(
        description="Build Werewolf & Villager finetuning datasets from game logs"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        required=True,
        help="Folder containing one or more game‐logs JSON files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to write `werewolf_dataset/` and `villager_dataset/`",
    )
    args = parser.parse_args()

    wolf_ex, vill_ex = process_logs(args.logs_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    wolf_ds = Dataset.from_list(wolf_ex)
    vill_ds = Dataset.from_list(vill_ex)

    wolf_ds.save_to_disk(os.path.join(args.out_dir, "werewolf_dataset"))
    vill_ds.save_to_disk(os.path.join(args.out_dir, "villager_dataset"))

    print(f"▶️ Saved {len(wolf_ds)} werewolf examples to `{args.out_dir}/werewolf_dataset/`")
    print(f"▶️ Saved {len(vill_ds)} villager examples to `{args.out_dir}/villager_dataset/`")


if __name__ == "__main__":
    main()
