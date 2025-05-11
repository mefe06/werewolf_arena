import os
import json
import argparse
from datasets import Dataset
from huggingface_hub import create_repo
def format_conversation(prompt: str, response: str) -> str:
    return (
        "<im_start>user\n"
        f"{prompt}"
        "<im_end>\n"
        "<im_start>assistant\n"
        f"{response}"
        "<im_end>"
    )

def load_games(path: str):
    """Load JSON list, single object, or JSONL → List[dict]."""
    with open(path, "r") as f:
        raw = f.read().strip()

    # JSON array
    if raw.startswith("[") and raw.endswith("]"):
        data = json.loads(raw)
        if isinstance(data, list):
            return data

    # Single JSON object
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
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
            continue
    return games

# def process_logs(logs_dir: str, filter_by_winner: bool=False):
#     werewolf_examples = []
#     villager_examples = []

#     for entry in os.listdir(logs_dir):
#         entry_path = os.path.join(logs_dir, entry)

#         # if it's a folder, look for game_logs.json inside it
#         if os.path.isdir(entry_path):
#             json_path = os.path.join(entry_path, "game_logs.json")
#             if not os.path.isfile(json_path):
#                 continue
#         # else if it's directly a .json file, use it
#         elif entry.lower().endswith(".json"):
#             json_path = entry_path
#         else:
#             continue

#         games = load_games(json_path)
#         for game in games:
#             if not isinstance(game, dict):
#                 continue

#             winner = game.get("winner")  # "Werewolves" or "Villagers" or None

#             # — Werewolf examples —
#             if (not filter_by_winner) or (winner == "Werewolves"):
#                 elim = game.get("eliminate")
#                 if isinstance(elim, dict) and elim.get("prompt") and elim.get("raw_resp"):
#                     text = format_conversation(elim["prompt"], elim["raw_resp"])
#                     werewolf_examples.append({"text": text})

#             # — Villager examples —
#             if (not filter_by_winner) or (winner == "Villagers"):
#                 for round_actions in game.get("bid", []):
#                     for player_name, entry in round_actions:
#                         prompt = entry.get("prompt", "")
#                         raw    = entry.get("raw_resp", "")
#                         if "the Villager" in prompt and prompt and raw:
#                             text = format_conversation(prompt, raw)
#                             villager_examples.append({"text": text})

#     return werewolf_examples, villager_examples

def process_logs(logs_dir: str, filter_by_winner: bool=False):
    werewolf_examples = []
    villager_examples = []

    for entry in os.listdir(logs_dir):
        folder = os.path.join(logs_dir, entry)
        if not os.path.isdir(folder):
            continue

        logs_path = os.path.join(folder, "game_logs.json")
        if not os.path.isfile(logs_path):
            continue

        # locate metadata file
        complete = os.path.join(folder, "game_complete.json")
        partial  = os.path.join(folder, "game_partial.json")
        meta_path = complete if os.path.isfile(complete) else partial if os.path.isfile(partial) else None

        # if filtering, require a complete game with a real winner
        winner = None
        if filter_by_winner:
            if not meta_path:
                continue
            meta = load_games(meta_path)
            if not meta:
                continue
            winner = meta[0].get("winner", "")
            if winner not in ("Werewolves", "Villagers"):
                continue

        # load the actual game logs
        games = load_games(logs_path)
        for game in games:
            if not isinstance(game, dict):
                continue

            game_winner = winner if filter_by_winner else game.get("winner")

            # — Werewolf examples —
            if (not filter_by_winner) or (game_winner == "Werewolves"):
                elim = game.get("eliminate")
                if isinstance(elim, dict) and elim.get("prompt") and elim.get("raw_resp"):
                    text = format_conversation(elim["prompt"], elim["raw_resp"])
                    werewolf_examples.append({"text": text})

            # — Villager examples —
            if (not filter_by_winner) or (game_winner == "Villagers"):
                for round_actions in game.get("bid", []):
                    for _, entry in round_actions:
                        prompt = entry.get("prompt", "")
                        raw    = entry.get("raw_resp", "")
                        if "the Villager" in prompt and prompt and raw:
                            text = format_conversation(prompt, raw)
                            villager_examples.append({"text": text})

    return werewolf_examples, villager_examples

def main():
    parser = argparse.ArgumentParser(
        description="Build Werewolf & Villager finetuning datasets from game logs"
    )
    parser.add_argument(
        "--logs_dir", required=True,
        help="Folder containing JSON game‐log files"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output folder for `werewolf_dataset/` and `villager_dataset/`"
    )
    parser.add_argument(
        "--filter_by_winner", action="store_true",
        help=(
            "If set, only emit Werewolf turns from games where "
            "werewolves actually won, and Villager turns from games "
            "where villagers actually won."
        )
    )
    args = parser.parse_args()

    wolf_ex, vill_ex = process_logs(
        args.logs_dir,
        filter_by_winner=args.filter_by_winner
    )

    os.makedirs(args.out_dir, exist_ok=True)

    wolf_ds = Dataset.from_list(wolf_ex)
    vill_ds = Dataset.from_list(vill_ex)

    wolf_ds.save_to_disk(os.path.join(args.out_dir, "werewolf_dataset"))
    vill_ds.save_to_disk(os.path.join(args.out_dir, "villager_dataset"))

    print(f"▶️ Saved {len(wolf_ds)} werewolf examples → {args.out_dir}/werewolf_dataset")
    print(f"▶️ Saved {len(vill_ds)} villager examples → {args.out_dir}/villager_dataset")

    # — automatically create & push HF repos named after out_dir —
    base = os.path.basename(os.path.abspath(args.out_dir))
    wolf_repo = f"{base}-werewolf-dataset"
    vill_repo = f"{base}-villager-dataset"

    print(f"🔄 Creating & pushing {wolf_repo} …")
    create_repo(wolf_repo, exist_ok=True, repo_type="dataset")
    wolf_ds.push_to_hub(repo_id=wolf_repo)
    print("✅ Done.")

    print(f"🔄 Creating & pushing {vill_repo} …")
    create_repo(vill_repo, exist_ok=True, repo_type="dataset")
    vill_ds.push_to_hub(repo_id=vill_repo)
    print("✅ Done.")

if __name__ == "__main__":
    main()
