# bubble-OMR
This is a set of scripts, written in Python, for a versitile and fast bubble sheet OMR scanner and scoring software.

![lead_image](/images/top.png)

## Dependencies
To use this software you will need Python 3.10 or higher.  These scripts rely on several additional packages.

pip install opencv-contrib-python numpy Pillow PyMuPDF

--

# Bubble-OMR overview
This is a quick summary of how the pipeline works.

## Step 1- Make your bubble sheet
1.	Make your bubble sheet as a pdf file.  I provide templates in the 'bubble forms' folder.  I used Affinity Designer to make these (similar to Adobe-Illustrator), but any software could work as long as it exports your bubble sheet as a pdf. 
2.  It’s a good idea to add landmarks called ARUCO markers to it so the scans, which can be wonky and slightly misaligned/rotated, can be easily aligned with software prior to analysis.
2.	Make a ‘config.json’ file that tells the marking software where the question bubbles are, where the student ID bubbles are, etc.  Use the **zone_visualizer.py** script to see how well the zones described in the config.json file fit to your template bubble sheet so you can make adjustments to get everything lined up properly.
3.	Make a key as a text file.  The answers (as letters like A,B,A,D,C,C,D...) can be comma-separated or separated with newlines.

## Step 2- Scan and align your bubble sheets
1.	Scan your student bubble sheets with a high quality scanner (garbage in, garbage out).  
2.  It's easiest if the scans are provided as a single multi-page pdf at 300dpi.
2.	Align the student pdfs to a blank version of the bubble sheet (the clean, original pdf file is best) using the **scan_align.py** script.  It will give you a new set of pdfs that are all aligned so the software can accurately find the bubbles.

## Step 3- Score 
1.	Run the **bubble_score.py** script on the aligned pdf file in a directory that also has the key.txt file and the config file.  Parameters for threshold darkness and completeness can be set as options.
2.  You will get an output comma-separated file with each student in a row, their answers, and their final score
3.  You can optionally export a pdf of the bubble sheets with circles that visually indicate which bubbles were scored.  This is useful for diagnosing issues.

<i>Future versions will add analysis functionality including mean, mode, and per-question analysis to determine if a question is well-written.</i>

### Example commands:

```
python zone_visualizer.py --config testform_config.json --input blank_template.pdf --page 1 --bubble-shape circle --out zones_overlay.jpg
```

```
python scan_aligner.py --method auto --dpi 300 --fallback-original --save-debug debug_auto --template template.pdf --input-pdf test.pdf --out testalign.pdf
```

```
python bubble_score.py --config config.json --key-txt key.txt --total-questions 16 --out-csv results.csv --out-annotated-dir annotated_pages --annotate-all-cells --name-min-fill 0.70  --label-density testalign.pdf
```

---
# Detailed instructions

## Making your bubble sheet
If you're in a hurry, we provide a few different templates (and their corresponding config files) that should be ready to use right away with no issues.  You can also freely modify these files for your own use.  If you want to make your own bubble sheet from scratch that's fine too.

Suggestions:
1. You should align the bubbles with even spacing between each.  You'll run into problems if the gaps between bubbles are not consistent.   Use the 'align' and 'distribute' features on your graphics software to ensure your bubbles are aligned properly.

2. You may want to decrease the darkness of the bubbles themselves (gray instead of black circles and letters) so the student marks stand out more against the background of the bubbles. (I made my bubbles with hollow black circles and dialed their transparency down to 50% before saving as a pdf.)

---
## Using the zone_visualizer.py script to make the config file for your bubble sheet template

The config file is like a map that tells bubble_score.py where the bubbles are located on the bubble sheet and what type of bubble the are (student name, student ID, test version, or the answer to a question).  This is a very important starting step because if the config file doesn't line up with the bubble sheet then you will get poor results.  

Config files are written in 'JSON' format.

```python zone_visualizer.py --config config.json --input blank_sheet.pdf --page 1 --bubble-shape circle --out zone_overlay.jpg```


---
## Making your test key
Test keys are text files where the answers (A, B, C, D, E...) are separated by spaces, commas, or newlines.

---
## Aligning your scanned documents with scan_aligner.py

After scanning it is typical that the page images will be randomly a bit off-center or askew.  A small bit of rotation in a page is usually tolerated by the scoring software but much better results will be obtained the pages are pre-processed to align almost perfectly with the original template that matches the config map file.

A philosophy of 'garbage in, garbage out' should apply here.  The higher quality scans you provide, the less likely you'll have issues later.  300 dpi inputs are great.  A high quality scanner that isn't prone to warping or scrunching is also great.

Example command:
```
python scan_aligner.py --method auto --dpi 300 --fallback-original --save-debug debug_auto --template template.pdf --input-pdf test.pdf --out aligned_scans.pdf
```
#### Mandatory flags:
--input-pdf {filename (and path) of input pdf file(s)}
--out {output filename including path if necessary}
--template {filename of the template everything will be aligned to}

#### Optional flags:
```
--method {aruco,feature,auto} (default auto)
--dpi 300 (rasterization)
--fallback-original (keeps page scaled to template size if alignment fails)
--save-debug DIR (write per-page PNGs with overlay)
--metrics-csv out.csv (a csv file of per-page stats, good for troubleshooting)
--first-page N / --last-page M (sekect a subset of pages to align 0-based inclusive)
```

---
## Scoring the tests with bubble_score.py


---
## Analyzing the results with bubble_stats.py

The script *bubble_stats.py* processes the results from *bubble_score.py*. It supports a KEY row (with correct answers) and student responses, converts them into a correctness matrix (0/1), and computes per-item and exam-level statistics.

### Usage
Basic command:
python bubble_stats.py -i results.csv -o results_with_item_stats.csv
Optional Flags
•	-i, --input: Path to input CSV (required).
•	-o, --output: Path to output CSV. Default: input name + '_with_item_stats.csv'.
•	--item-pattern: Regex to detect item columns. Default: Q\d+.
•	--percent: Output difficulty as percent (0–100) instead of proportion (0–1).
•	--label-col: Which column to use for placing summary row labels. Default: first non-item column.
•	--exam-stats-csv: Path to write exam-level stats (KR-20, KR-21, mean, SD, etc).
•	--plots-dir: Directory to save item characteristic plots (PNG).
•	--key-row-index: Row index (0-based) of KEY row. Default: auto-detect (looks for 'KEY').
•	--answers-mode: Force interpretation of responses: 'letters', 'binary', or 'auto' (default).
•	--item-report-csv: Path to write per-item distractor analysis table.
•	--key-label: Label string to identify KEY row in non-item column. Default: 'KEY'.
Outputs
1. Main CSV (results_with_item_stats.csv):
- Original data (including KEY row).
- Appends two rows at the bottom:
  * 'Pct correct (0-1)' or '(0-100)': item difficulty.
  * 'Point–biserial': discrimination index of each item.
2. Exam-level stats (exam_stats.csv):
Contains summary statistics:
- k_items: Number of questions.
- mean_total, sd_total, var_total: Distribution of student scores.
- avg_difficulty: Mean proportion correct across items.
- KR-20: Reliability estimate for dichotomous items.
- KR-21: Approximate reliability using mean difficulty.
3. Item report (item_analysis.csv):
One row per item-option combination:
- item, key, option, is_key.
- option_count, option_prop: How often each option was selected.
- option_biserial: Correlation between choosing this option and total score (excl. item).
- item_difficulty, item_point_biserial.
4. Item plots (item_plots/):
One PNG per item, showing a nonparametric Item Characteristic Curve (ICC):
- X-axis: binned total-minus-item score.
- Y-axis: proportion correct in each bin.
Interpretation of Statistics
•	Difficulty (Pct correct):
Proportion of students who answered correctly. Ideal values often 0.3–0.8.
•	Point–biserial:
Correlation between correctness and total score (excluding the item). Higher positive values indicate better discrimination. Negative values are problematic.
•	KR-20:
Reliability coefficient for dichotomous items. Higher values (≥0.7) indicate consistent test performance.
•	KR-21:
Approximation of KR-20 assuming items have similar difficulty. Useful when item-level data is limited.
•	Option biserial:
Correlation between selecting a particular option and student ability. Correct option should be positive; distractors should be negative or near-zero.

