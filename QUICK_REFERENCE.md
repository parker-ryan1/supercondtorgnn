# Quick Reference Guide

## 🚀 QUICK START (5 Minutes)

### 1. Train Model
```bash
python train_stable_enhanced.py --max-structures 500 --epochs 40
```
**Output**: `models/stable_enhanced_model.pt` + `results/training_results.json`

### 2. Analyze Results
```bash
python analyze_results.py
```
**Output**: Detailed analysis + recommendations

### 3. Create Visualizations
```bash
python visualize_results.py
```
**Output**: 4 PNG files in `results/`

---

## 📋 COMMAND REFERENCE

### Training

| Command | Purpose | Time |
|---------|---------|------|
| `python train_stable_enhanced.py` | Train with default settings | ~5 min |
| `python train_stable_enhanced.py --max-structures 1000` | Train with more data | ~10 min |
| `python train_stable_enhanced.py --epochs 60` | Train longer | ~8 min |

### Analysis

| Command | Purpose |
|---------|---------|
| `python analyze_results.py` | Full analysis + recommendations |
| `python visualize_results.py` | Generate visualizations |

### Data Fetching

| Command | Purpose | Time |
|---------|---------|------|
| `python fetch_detailed_data.py` | Fetch for first 100 materials | ~5 min |
| `python fetch_for_structures.py` | Fetch for all structures | ~30 min |

---

## 📊 KEY METRICS

### Current Performance (500 materials)

- **R² Score**: 0.2333 ⭐⭐
- **MAE**: 30.86 K ⭐⭐
- **RMSE**: 53.29 K
- **Enhanced Coverage**: 0.4% ⚠️

### Expected Performance (with 80% coverage)

- **R² Score**: 0.45-0.55 ⭐⭐⭐⭐
- **MAE**: 15-20 K ⭐⭐⭐⭐
- **Enhanced Coverage**: 80%+ ✅

---

## 🐛 TROUBLESHOOTING

### NaN Loss During Training

**Solution**: Use `train_stable_enhanced.py` (not original `gnn_model.py`)

### Low Enhanced Coverage

**Solution**:
```bash
python fetch_for_structures.py
```

### Out of Memory

**Solution**: Reduce batch size
```bash
python train_stable_enhanced.py --max-structures 300
```

---

## 📁 KEY FILES

### Scripts (Run These)
- `train_stable_enhanced.py` - Main training
- `analyze_results.py` - Analysis
- `visualize_results.py` - Plots

### Data
- `data/superconductors.csv` - Base data
- `data/enhanced_superconductors.csv` - MP data

### Models
- `models/stable_enhanced_model.pt` - Trained model

### Results
- `results/training_results.json` - Metrics
- `results/*.png` - Visualizations

---

## 🎯 WORKFLOW

```
Step 1: Train
    ↓
python train_stable_enhanced.py
    ↓
Step 2: Analyze
    ↓
python analyze_results.py
    ↓
Step 3: Visualize
    ↓
python visualize_results.py
    ↓
Step 4: Review
    ↓
Check results/*.png
```

---

## 💡 TIPS

### For Better Performance

1. **Fetch more data first**:
   ```bash
   python fetch_for_structures.py
   ```

2. **Then train with larger dataset**:
   ```bash
   python train_stable_enhanced.py --max-structures 1000 --epochs 60
   ```

3. **Compare before/after**:
   - Check R² improvement
   - Review enhanced coverage %

### For Quick Testing

```bash
# Quick test (100 materials, 20 epochs)
python train_stable_enhanced.py --max-structures 100 --epochs 20

# Analyze
python analyze_results.py

# Done in ~2 minutes!
```

---

## ⚙️ ARGUMENTS

### train_stable_enhanced.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-structures` | 500 | Number of materials |
| `--epochs` | 50 | Training epochs |

### fetch_detailed_data.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-materials` | 100 | Materials to fetch |

---

## 📈 PERFORMANCE TARGETS

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| R² Score | 0.23 | 0.50+ | 🔄 In Progress |
| MAE | 31K | 15K | 🔄 In Progress |
| Coverage | 0.4% | 80%+ | ⚠️ Needs Work |
| Dataset | 500 | 1000+ | 🔄 Scalable |

---

## ✅ CHECKLIST

- [x] Fix NaN training issue
- [x] Integrate enhanced data
- [x] Create stable pipeline
- [x] Add analysis tools
- [x] Generate visualizations
- [ ] Fetch full enhanced dataset
- [ ] Re-train with 80%+ coverage
- [ ] Achieve R² > 0.5

---

## 🆘 GETTING HELP

1. **Check analysis**: `python analyze_results.py`
2. **Review logs**: Terminal output
3. **Check visualizations**: `results/*.png`
4. **Read summary**: `INTEGRATED_PIPELINE_SUMMARY.md`

---

**Status**: ✅ WORKING  
**Last Updated**: October 30, 2025


