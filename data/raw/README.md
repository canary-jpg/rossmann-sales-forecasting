# Raw data

Not committed to the repo (see `.gitignore`).

Download from Kaggle:
https://www.kaggle.com/c/rossmann-store-sales/data

Or via the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c rossmann-store-sales -p data/raw
unzip data/raw/rossmann-store-sales.zip -d data/raw
```

Expected files:
- `train.csv` - ~1,017,209 rows: daily sales per store, Jan 2013-Jul 2015
- `store.csv` - 1,115 rows: metadata per store (type assortment, competition distance, promo intervals)
- `test.csv` - Kaggle's own submission test set. **Not used in this project** - we create our own chronological held-out period from `train.csv` instead, since `test.csv` has no `Sales` column (it's for Kaggle leadership submission, not for our own evaluation).

**Known data quirks, confirmed from the dataset's documentation and common issues reported by others who've used it - verify these against your own EDA rather than assuming:**
- `StateHoliday` mixes the string`"0"` and other representations for "not a holiday" - a real data type inconsistency, not a typo on your part if you see it.
- `CompetitionDistance` (in `store.csv`) has missing values.
- Some rows have `Open == 0` (stored closed that day) - `Sales` is trivially 0 on these days, and they typically need explicit handling (usually excluded from training) rather than being treated as normal low-sales days.

Place `train.csv` and `store.csv` at `data/raw/train.csv` and `data/raw/store.csv` before running the notebooks.