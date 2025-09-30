# bubble-OMR
This is a set of scripts, written in Python, for a versitile and fast bubble sheet OMR scanner and scoring software.

![lead_image](/images/top.png)

## Dependencies
To use this software you will need Python 3.10 or higher.  These scripts rely on several additional packages.

pip install opencv-contrib-python numpy Pillow PyMuPDF

--

# Bubble-OMR Instructions
These are instructions for a set of bubble-sheet scoring scripts for multiple choice exams.

## Step 1- Make your bubble sheet
1.	Make your bubble sheet as a pdf file.  I provide templates in the 'bubble forms' folder.  I used Affinity Designer, but any software could work as long as it exports your bubble sheet as a pdf. 
2.  It’s a good idea to add landmarks called ARUCO markers to it so the scans, which can be wonky and slightly misaligned/rotated, can be easily aligned with software prior to analysis.
2.	Make a ‘config.json’ file that tells the marking software where the question bubbles are, where the student ID bubbles are, etc.  Use the ‘zone_visualizer.py’ to see how well the zones described in the json file fit to your scanned bubble sheet so you can adjust the config.json file to get everything lined up properly.
3.	Make a key as a text file.  The answers (as letters like A,B,A,D,C,C,D...) can be comma-separated or separated with newlines.

## Step 2- Scan and align your bubble sheets:
1.	Scan your student bubble sheets with a high quality scanner (garbage in, garbage out).  
2.  It's easiest if the scans are provided as a single multi-page pdf at 300dpi.
2.	Align the student pdfs to a blank version of the bubble sheet (the clean, original pdf file is best) using the scan_align.py script.  It will give you a new set of pdfs that are all aligned so the software can accurately find the bubbles.

## Step 3- Score 
1.	Run the bubble_score.py script on the aligned pdf file in a directory that also has the key.txt file and the config file.  Parameters for threshold darkness and completeness can be set as options.
2.  You will get an output comma-separated file with each student in a row, their answers, and their final score
3.  You can optionally export a pdf of the bubble sheets with circles that visually indicate which bubbles were scored.  This is useful for diagnosing issues.

<i>Future versions will add analysis functionality including mean, mode, and per-question analysis to determine if a question is well-written.</i>

### Example commands:

> python zone_visualizer.py --config testform_config.json --input blank_template.pdf --page 1 --bubble-shape circle --out zones_overlay.jpg

> python scan_aligner.py --method auto --dpi 300 --fallback-original --save-debug debug_auto --template template.pdf --input-pdf test.pdf --out testalign.pdf

> python bubble_score.py --config config.json --key-txt key.txt --total-questions 16 --out-csv results.csv --out-annotated-dir annotated_pages --annotate-all-cells --name-min-fill 0.70  --label-density testalign.pdf

---
## Making your bubble sheet.
If you're in a hurry, we provide a few different templates (and their corresponding config files) that should be ready to use right away with no issues.  You can also freely modify these files for your own use.  If you want to make your own bubble sheet from scratch that's fine too.

Suggestions:
1. You should align the bubbles with even spacing between each.  You'll run into problems if the gaps between bubbles are not consistent.   Use the 'align' and 'distribute' features on your graphics software to ensure your bubbles are aligned properly.

2. You may want to decrease the darkness of the bubbles themselves (gray instead of black circles and letters) so the student marks stand out more against the background of the bubbles. (I made my bubbles with hollow black circles and dialed their transparency down to 50% before saving as a pdf.)


## Using the zone_visualizer.py script to make the config file for your bubble sheet template.

The config file is like a map that tells bubble_score.py where the bubbles are located on the bubble sheet and what type of bubble the are (student name, student ID, test version, or the answer to a question).  This is a very important starting step because if the config file doesn't line up with the bubble sheet then you will get poor results.  

Config files are written in 'JSON' format.

> python zone_visualizer.py --config config.json --input blank_sheet.pdf --page 1 --bubble-shape circle --out zone_overlay.jpg


## Making your test key.
Test keys are text files where the answers (A, B, C, D, E...) are separated by spaces, commas, or newlines.

## Aligning your scanned documents with scan_aligner.py.

> python scan_aligner.py --method auto --dpi 300 --fallback-original --save-debug debug_auto --template template.pdf --input-pdf test.pdf --out testalign.pdf

scan_aligner.py chooses one of two methods to align your student scans.

OPTIONAL FLAGS:
```
--method {aruco,feature,auto} (default auto)
--dpi 300 (rasterization)
--fallback-original (keeps page scaled to template size if alignment fails)
--save-debug DIR (write per-page PNGs with overlay)
--first-page N / --last-page M (0-based inclusive)
--metrics-csv out.csv (per-page stats)
```
