# bubble-OMR

bubble-OMR is a Python tool to align, visualize, grade, and analyze bubble-sheet exams.  
It supports both **YAML** and **JSON** configuration files, and includes both a CLI and a Streamlit-based GUI.

---

## Installation

```bash
git clone https://github.com/navarrew/bubble-OMR.git
cd bubble-OMR
pip install -e .
```

Make sure you have required dependencies (OpenCV, Typer, Streamlit, Ghostscript for PDF compression).

---

## Usage

### Align scans
```bash
bubble-omr align raw_scans.pdf --template template.pdf --out-pdf aligned_scans.pdf
```

### Visualize configuration
```bash
bubble-omr visualize template.pdf --config config.yaml --out-image config_overlay.png
```

### Grade aligned scans
```bash
bubble-omr grade aligned_scans.pdf --config config.yaml --key-txt key.txt --total-questions 34 --out-csv results.csv --out-annotated-dir annotated --annotate-all-cells
```

### Compute statistics
```bash
bubble-omr stats results.csv --output-csv results_with_item_stats.csv --item-report-csv item_analysis.csv --exam-stats-csv exam_stats.csv
```

### Launch the GUI
```bash
bubble-omr gui
```

---

## Example Config (YAML)

```yaml
answer_layouts:
  - zone: [0.06, 0.59, 0.25, 0.92]
    questions: 16
    questions_per_row: 1
    choices: 5
  - zone: [0.29, 0.59, 0.48, 0.92]
    questions: 16
    questions_per_row: 1
    choices: 5

last_name_layout:
  zone: [0.36, 0.10, 0.76, 0.56]
  rows: 27
  cols: 14
  symbols: " ABCDEFGHIJKLMNOPQRSTUVWXYZ"

first_name_layout:
  zone: [0.79, 0.10, 0.94, 0.56]
  rows: 27
  cols: 5
  symbols: " ABCDEFGHIJKLMNOPQRSTUVWXYZ"

id_layout:
  zone: [0.03, 0.39, 0.32, 0.56]
  rows: 10
  cols: 10
  symbols: "0123456789"

version_layout:
  zone: [0.20, 0.30, 0.31, 0.32]
  rows: 1
  cols: 4
  symbols: "ABCD"
  orientation: "horizontal"

bubble_shape: circle
```

---

## Outputs

- **Aligned scans**: multipage PDF (`aligned_scans.pdf`)
- **Overlay visualization**: PNG preview of config zones
- **Grading results**: `results.csv` per student
- **Annotated sheets**: optional PNGs in `annotated/`
- **Stats**: item analysis, KR-20 reliability, difficulty, point-biserial

---

## License

MIT License.
