import os
import shutil
import argparse

def clean_cache(all=False, embeds=False, checkpoints=False, dataset=False):
    files_to_remove = []
    dirs_to_remove = []

    if embeds or all:
        files_to_remove.append("software/data/esc50_embeds_v2.pth")
        files_to_remove.append("software/data/esc50_embeds.pth")
    
    if checkpoints or all:
        files_to_remove.append("software/training_checkpoint.pth")
        files_to_remove.append("software/mapper_grounded.pth")
        files_to_remove.append("software/mapper_final.pth")
        # remove any epoch-based saves if they exist
        import glob
        files_to_remove.extend(glob.glob("software/tiny_tcn_epoch_*.pth"))
        files_to_remove.extend(glob.glob("software/mapper_epoch_*.pth"))

    if dataset or all:
        dirs_to_remove.append("software/data/esc50")

    print("--- Cache Cleaning Utility ---")
    
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed file: {f}")
        else:
            print(f"Skipping (not found): {f}")

    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed directory: {d}")
        else:
            print(f"Skipping (not found): {d}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean training cache and checkpoints")
    parser.add_argument("--all", action="store_true", help="Delete everything (dataset, embeds, checkpoints)")
    parser.add_argument("--embeds", action="store_true", help="Delete pre-computed CLAP embeddings")
    parser.add_argument("--checkpoints", action="store_true", help="Delete model checkpoints")
    parser.add_argument("--dataset", action="store_true", help="Delete raw ESC-50 dataset")
    
    args = parser.parse_args()
    
    if not (args.all or args.embeds or args.checkpoints or args.dataset):
        print("Please specify what to clean (e.g., --embeds or --all)")
    else:
        clean_cache(all=args.all, embeds=args.embeds, checkpoints=args.checkpoints, dataset=args.dataset)
