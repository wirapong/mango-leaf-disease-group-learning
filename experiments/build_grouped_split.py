from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass
class ImageRecord:
    filepath: str
    label: str
    clean_stem: str
    ahash: str
    dhash: str


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img_path in sorted(class_dir.glob('*.jpg')):
            rows.append({'filepath': str(img_path), 'label': class_dir.name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f'No JPG files found in {data_dir}')
    return df


def clean_stem(stem: str) -> str:
    stem = re.sub(r'( \(Custom\))+', '', stem)
    stem = re.sub(r'( - Copy)+$', '', stem)
    return stem


def average_hash(path: str, size: int = 8) -> str:
    img = Image.open(path).convert('L').resize((size, size))
    arr = np.array(img)
    mean = float(arr.mean())
    return ''.join('1' if x > mean else '0' for x in arr.flatten())


def difference_hash(path: str, size: int = 8) -> str:
    img = Image.open(path).convert('L').resize((size + 1, size))
    arr = np.array(img)
    diff = arr[:, 1:] > arr[:, :-1]
    return ''.join('1' if x else '0' for x in diff.flatten())


def annotate_records(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in df.itertuples(index=False):
        path = Path(row.filepath)
        records.append(
            {
                'filepath': row.filepath,
                'label': row.label,
                'clean_stem': clean_stem(path.stem),
                'ahash': average_hash(row.filepath),
                'dhash': difference_hash(row.filepath),
            }
        )
    return pd.DataFrame(records)


def assign_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frames = []
    audit = {}
    for label, class_df in df.groupby('label', sort=True):
        class_df = class_df.reset_index(drop=True).copy()
        uf = UnionFind(len(class_df))

        for key in ['clean_stem', 'ahash', 'dhash']:
            bucket = defaultdict(list)
            for idx, value in enumerate(class_df[key].tolist()):
                bucket[value].append(idx)
            for indices in bucket.values():
                if len(indices) > 1:
                    first = indices[0]
                    for other in indices[1:]:
                        uf.union(first, other)

        roots = [uf.find(i) for i in range(len(class_df))]
        root_to_gid = {}
        group_ids = []
        for root in roots:
            if root not in root_to_gid:
                root_to_gid[root] = f'{label}__g{len(root_to_gid):04d}'
            group_ids.append(root_to_gid[root])
        class_df['group_id'] = group_ids
        group_sizes = class_df['group_id'].value_counts().to_dict()
        audit[label] = {
            'num_images': int(len(class_df)),
            'num_groups': int(class_df['group_id'].nunique()),
            'max_group_size': int(max(group_sizes.values())),
            'groups_gt1': int(sum(size > 1 for size in group_sizes.values())),
        }
        frames.append(class_df)
    return pd.concat(frames, ignore_index=True), audit


def greedy_class_split(class_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(class_df)
    targets = {
        'train': round(total * TRAIN_RATIO),
        'val': round(total * VAL_RATIO),
        'test': total - round(total * TRAIN_RATIO) - round(total * VAL_RATIO),
    }

    grouped = []
    for group_id, gdf in class_df.groupby('group_id'):
        grouped.append((group_id, gdf.copy(), len(gdf)))
    grouped.sort(key=lambda x: (-x[2], x[0]))

    buckets = {'train': [], 'val': [], 'test': []}
    counts = {'train': 0, 'val': 0, 'test': 0}

    for group_id, gdf, size in grouped:
        best_split = None
        best_score = None
        for split_name in ['train', 'val', 'test']:
            new_counts = counts.copy()
            new_counts[split_name] += size
            score = sum(abs(new_counts[name] - targets[name]) for name in ['train', 'val', 'test'])
            overflow_penalty = sum(max(0, new_counts[name] - targets[name]) for name in ['train', 'val', 'test'])
            objective = (overflow_penalty, score, counts[split_name])
            if best_score is None or objective < best_score:
                best_score = objective
                best_split = split_name
        buckets[best_split].append(gdf)
        counts[best_split] += size

    train_df = pd.concat(buckets['train'], ignore_index=True) if buckets['train'] else pd.DataFrame(columns=class_df.columns)
    val_df = pd.concat(buckets['val'], ignore_index=True) if buckets['val'] else pd.DataFrame(columns=class_df.columns)
    test_df = pd.concat(buckets['test'], ignore_index=True) if buckets['test'] else pd.DataFrame(columns=class_df.columns)
    return train_df, val_df, test_df


def build_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    trains, vals, tests = [], [], []
    split_audit = {}
    for label, class_df in df.groupby('label', sort=True):
        train_df, val_df, test_df = greedy_class_split(class_df)
        trains.append(train_df)
        vals.append(val_df)
        tests.append(test_df)
        split_audit[label] = {
            'train': int(len(train_df)),
            'val': int(len(val_df)),
            'test': int(len(test_df)),
            'train_groups': int(train_df['group_id'].nunique()),
            'val_groups': int(val_df['group_id'].nunique()),
            'test_groups': int(test_df['group_id'].nunique()),
        }
    train = pd.concat(trains, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val = pd.concat(vals, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test = pd.concat(tests, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return train, val, test, split_audit


def audit_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    def keyed(df: pd.DataFrame, key: str) -> set[tuple[str, str]]:
        return set(zip(df['label'].tolist(), df[key].tolist()))

    audit = {}
    for key in ['group_id', 'clean_stem', 'ahash', 'dhash']:
        train_set = keyed(train, key)
        val_set = keyed(val, key)
        test_set = keyed(test, key)
        audit[key] = {
            'train_val_overlap': len(train_set & val_set),
            'train_test_overlap': len(train_set & test_set),
            'val_test_overlap': len(val_set & test_set),
        }
    return audit


def save_split(df: pd.DataFrame, path: Path) -> None:
    df[['filepath', 'label', 'group_id', 'clean_stem', 'ahash', 'dhash']].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build leakage-resistant grouped train/val/test split for MangoLeafBD.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = build_dataframe(data_dir)
    annotated_df = annotate_records(raw_df)
    grouped_df, group_audit = assign_groups(annotated_df)
    train_df, val_df, test_df, split_audit = build_split(grouped_df)
    overlap_audit = audit_overlap(train_df, val_df, test_df)

    save_split(train_df, output_dir / 'train_split.csv')
    save_split(val_df, output_dir / 'val_split.csv')
    save_split(test_df, output_dir / 'test_split.csv')

    summary = {
        'group_audit': group_audit,
        'split_audit': split_audit,
        'overlap_audit': overlap_audit,
        'num_train': int(len(train_df)),
        'num_val': int(len(val_df)),
        'num_test': int(len(test_df)),
    }
    (output_dir / 'group_split_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
