# Batch B locked final-test collection

Place the anonymized, labelled Batch B export here as `final_chat.csv`
with columns `match_id,message_order,text,label,notes`.

Keep labels sealed until `artifacts/frozen_improved_hybrid_config.json` exists
(or the improved screen aborts and you intentionally evaluate the original
freeze). Then:

```powershell
.venv\Scripts\Activate.ps1
python predict.py --batch-b
python evaluate_final_test.py --batch-b
```

Predictions and metrics land under `artifacts/final_test/batch_b/`.
