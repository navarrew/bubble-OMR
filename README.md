# bubble-OMR
versitile and fast bubble sheet OMR scanner and scoring software in python

# Bubblesheet Scanner Instructions
These are instructions for a set of bubble-sheet scoring scripts for multiple choice exams.

## General Overview: Make your bubble sheet:
1.	Make your bubble sheet as a pdf file.  It’s best if you add landmarks called ARUCO markers to it so the scans, which can be wonky and slightly misaligned/rotated, can be easily aligned with software prior to analysis.
2.	Make a ‘config.json’ file that tells the marking software where the question bubbles are, where the student ID bubbles are, etc.  Use the ‘zone_visualizer.py’ to see how well the zones described in the json file fit to your scanned bubble sheet so you can adjust the config.json file to get everything lined up properly.
3.	Make a key as a text file.

## General Overview 2: Scan and score your bubble sheets:
1.	Scan your student bubble sheets with a high quality scanner (garbage in, garbage out).  Best if the scans are provided as a single multi-page pdf at 300dpi.
2.	Align the student pdfs to a blank version of the bubble sheet (the clean, original pdf file is best) using the scan_align.py script.  It will give you a new set of pdfs that are all aligned so the software can accurately find the bubbles.
3.	Run the bubble_score.py scoring software on the aligned pdfs and you will get an output csv file.

<i>Future versions will add analysis functionality including mean, mode, and per-question analysis.</i>

### Example commands:

python zone_visualizer.py --config MGYform_config.json --input vector_template.pdf --page 1 --bubble-shape circle --out zones_overlay.jpg

python scan_aligner.py --method auto --dpi 300 --fallback-original --save-debug debug_auto --template template.pdf --input-pdf test.pdf --out testalign.pdf

python bubble_score.py --config config.json --key-txt key.txt --total-questions 16 --out-csv results.csv --out-annotated-dir annotated_pages --annotate-all-cells --name-min-fill 0.70  --label-density testalign.pdf
