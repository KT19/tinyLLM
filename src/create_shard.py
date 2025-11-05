import os
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm


def create_main(target_dir: str, stream: Any):
    os.makedirs(target_dir, exist_ok=True)

    def dir_size_gb(path: str) -> float:
        p = Path(path)
        total = 0
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except FileNotFoundError:
                    pass
        return total / (1024**3)

    chunk_size = 100000
    max_size_gb = 5
    chunk_paths = []
    examples = []
    chunk_idx = 0

    for i, ex in enumerate(tqdm(stream)):
        txt = ex.get("text")
        if txt:
            examples.append({"text": txt})

        # write one chunk
        if len(examples) >= chunk_size:
            ds_chunk = Dataset.from_list(examples)

            out_path = f"{target_dir}/chunk_{chunk_idx:02d}"
            ds_chunk.save_to_disk(out_path)

            chunk_paths.append(out_path)

            examples.clear()
            chunk_idx += 1

            used_gb = dir_size_gb(target_dir)
            print(f"\nsave new chunk: {out_path}")
            if used_gb >= max_size_gb:
                print(f"Reached roughly {used_gb:.2f} GB limit.")
                break

    # final chunk
    if examples:
        ds_chunk = Dataset.from_list(examples)
        out_path = f"{target_dir}/chunk_{chunk_idx:02d}"
        ds_chunk.save_to_disk(out_path)
        chunk_paths.append(out_path)

    print(f"Saved {len(chunk_paths)} chunks.")

    # Merge all chunks
    print("Merging chunks...")
    datasets = [Dataset.load_from_disk(p) for p in chunk_paths]
    merged = concatenate_datasets(datasets)

    # Save as single dataset
    final_dir = f"{target_dir}/final"
    merged.save_to_disk(final_dir)
    print(f"Dataset saved to {final_dir}")


if __name__ == "__main__":
    ##  fineweb
    stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True, revision="main")
    create_main(target_dir="dataset_source/fineweb_5gb", stream=stream)

    ## OpenMath
    stream = load_dataset("open-web-math/open-web-math", split="train", streaming=True, revision="main")
    create_main(target_dir="dataset_source/open-web-math_5gb", stream=stream)
