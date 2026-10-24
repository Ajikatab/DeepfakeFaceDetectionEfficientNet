import json, os

for var in ['B0', 'B3', 'B7']:
    print(f"\n{'='*60}")
    print(f"  {var} — Best Fold per Optimizer")
    print(f"{'='*60}")
    for opt in ['baseline', 'adam', 'sgd', 'adamw']:
        # Check kfold_summary
        summary_path = f"{var}/kfold_summary_{opt}.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
            print(f"\n  [{opt.upper()}] kfold_summary:")
            if 'best_fold' in data:
                print(f"    best_fold = {data['best_fold']}")
            # Print all keys to understand structure
            for k in sorted(data.keys()):
                if isinstance(data[k], (int, float, str)):
                    print(f"    {k} = {data[k]}")
        
        # Also check from raw history files
        print(f"  [{opt.upper()}] from history files:")
        best_fold = -1
        best_acc = 0
        best_loss = 999
        for fold in range(1, 6):
            hp = f"{var}/history_{opt}_fold{fold}.json"
            if os.path.exists(hp):
                with open(hp) as f:
                    h = json.load(f)
                va = max(h.get('val_accuracy', [0]))
                vl = min(h.get('val_loss', [999]))
                print(f"    Fold {fold}: best_val_acc={va:.4f}, best_val_loss={vl:.4f}")
                if va > best_acc:
                    best_acc = va
                    best_fold = fold
                    best_loss = vl
        print(f"    => BEST Fold = {best_fold} (val_acc={best_acc:.4f})")
