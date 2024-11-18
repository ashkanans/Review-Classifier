# src/data/data_preprocessing.py
import os
from collections import Counter

from tqdm import tqdm

# Define markers
positive_markers = {'fantastic', 'amazing', 'excellent', 'very good'}
neutral_markers = {'adequate', 'fine but', 'good but', 'ok but'}
negative_markers = {'returning', 'sucks', 'waste'}

# Combine markers and define star to marker mapping
markers = positive_markers | neutral_markers | negative_markers
star2markers = {
    1: negative_markers,
    2: negative_markers,
    3: neutral_markers,
    4: positive_markers,
    5: positive_markers
}

def filter_and_save_dataset(input_path: str, output_path: str):
    """Filters the dataset based on marker words and writes the output to a new file."""
    progress_bar = tqdm()
    star_writes = Counter()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as fi, open(output_path, 'w', encoding='utf-8') as fo:
        for i, line in enumerate(fi):
            try:
                # Parse the star and review, and apply lowercase
                star, review = line.strip().lower().split('\t')
                star = int(float(star))

                # Filter reviews based on length and marker presence
                if len(review) > 20 and len(review) < 100 and any(m in review for m in star2markers[star]):
                    star_writes[star] += 1
                    fo.write(f'{star}\t{review}\n')

            except Exception as e:
                pass

            # Update progress every 1,000 lines
            progress_bar.update()
            if i % 1_000 == 0:
                progress_bar.set_postfix(**{str(k): v for k, v in star_writes.items()})

    progress_bar.close()
    print(f"Filtering completed. Saved filtered data to {output_path}")
