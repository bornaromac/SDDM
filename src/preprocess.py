import yaml
from src.data import DataConfig, build_datasets

def main():
    cfg = yaml.safe_load(open("configs/ddpm_ham_64.yaml", "r", encoding="utf-8"))
    seed = int(cfg.get("seed", 42))

    data_cfg = DataConfig(**cfg["data"])
    train_ds, val_ds, stats = build_datasets(data_cfg, seed=seed)

    print("\nPreprocessing done.")
    print(f"- Metadata rows total:        {stats['metadata_rows_total']}")
    print(f"- Rows after label map:       {stats['rows_after_label_map']}")
    print(f"- Images indexed:             {stats['images_indexed']}")
    print(f"- Rows with existing files:   {stats['rows_with_existing_files']}")
    print(f"- Train size:                 {stats['train_size']}")
    print(f"- Val size:                   {stats['val_size']}")
    print(f"- Splits:                     data/splits/train.csv, data/splits/val.csv")
    print(f"- Extracted:                  data/extracted/ (ako prije nije postojalo)")

    print("\nImage dirs found:")
    for d in stats["image_dirs_found"]:
        print("  -", d)

    print("\nImage dirs used for indexing:")
    for d in stats["image_dirs_used_for_index"]:
        print("  -", d)

if __name__ == "__main__":
    main()
