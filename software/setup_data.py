import musdb
import os
import argparse

def setup_musdb(root_dir, download=False):
    """
    Sets up the MUSDB18 dataset. 
    If download is True, it will attempt to download the 7-second sample version.
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print(f"Created directory: {root_dir}")

    print(f"Initializing musdb in: {root_dir}")
    try:
        # download=True only downloads the 7s sample version (musdb18hq is not auto-downloadable via this API)
        db = musdb.DB(root=root_dir, download=download)
        
        if len(db) > 0:
            print(f"Success! Found {len(db)} tracks.")
            print(f"Example track: {db.tracks[0].name}")
        else:
            print("No tracks found. If you have the full dataset, point to the directory containing the 'train' and 'test' folders.")
            if not download:
                print("Hint: Run with --download to get the 7s sample version.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUSDB18 Setup Script")
    parser.add_argument("--path", type=str, default="software/data/musdb18", help="Path to musdb18 dataset")
    parser.add_argument("--download", action="store_true", help="Download the 7s sample version")
    
    args = parser.parse_args()
    setup_musdb(args.path, download=args.download)
