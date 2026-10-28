# Pothole Detection Project: Presentation Guide

## Suggested 4–5 minute explanation

### 1. Aim and scope

“My project explores object detection for potholes using pretrained YOLO models. I wanted to see how model family, training duration, optimizer choice, batch size, and dataset size affect validation performance. I organized the run outputs into a dashboard so I can compare metrics and inspect the training and evaluation plots together.”

“The project report, `project_report_ieee.pdf`, describes my project. The published reference paper I use for context is Fortin and Llantos’s *Performance Analysis of YOLO Versions for Real-time Pothole Detection*. They are different documents: my report is not an independent benchmark against my own results.”

### 2. Data and experimental variables

“The project documentation describes an original dataset of 13,767 images, with 11,068 for training, 1,808 for validation, and 891 for testing. It also describes a reduced 3,000-image version with 2,400 training, 300 validation, and 300 test images. I ran experiments using both full and reduced data.”

“Across 15 saved runs, I compared YOLOv5nu, YOLOv8n, YOLOv11n and YOLOv11s. Within YOLOv11, I varied training length, early-stopping patience, optimizer, dataset size, and in one run the batch size. The saved run folders and `args.yaml` files record those settings. These are configuration experiments; because several variables change together across runs, they are exploratory comparisons rather than a fully controlled ablation study.”

### 3. What I measured

“For each epoch, the training logs record precision, recall, mAP at IoU 0.5, mAP from IoU 0.5 to 0.95, training and validation losses, learning rates, elapsed training time, and epoch number. I calculate F1 from the logged validation precision and recall using 2 × precision × recall divided by precision plus recall. The paper does not report F1 in its final metrics table, so I calculate its F1 from its published precision and recall only for an indicative comparison.”

### 4. Dashboard walkthrough

1. **Overview & final metrics:** select the runs to compare. Point out final precision, recall, calculated F1, mAP50, mAP50-95 and losses. Explain that these values come from the final logged epoch; best mAP50-95 is shown separately.
2. **Training curves:** show mAP50 and box/class/DFL loss over epochs. Discuss convergence and whether validation loss follows training loss.
3. **Saved graphs:** show a confusion matrix, F1 curve, precision curve, recall curve, PR curve, and one validation prediction image. Explain that these are per-run diagnostic plots.
4. **Paper comparison:** show the published table, the selected run’s `args.yaml` settings, and the coverage table. Use this view to explain which paper figures/metrics are available and where setups differ.

### 5. Main result

“Among the current saved runs, `yolov11n_250_epochs_10_patience` has the strongest final mAP50 and mAP50-95: 0.86083 and 0.62280. Its final validation precision is 0.89294 and recall is 0.79363, giving a calculated F1 of about 0.8403. Its final training box, class, and DFL losses are 0.85064, 0.49572, and 1.01885. These are results for this run’s validation setup, not a matched test-set comparison with the paper.”

### 6. Context against the published reference

“The reference paper compares YOLOv9-tiny, YOLOv10-nano, and YOLOv11-nano. It reports final YOLOv11-nano precision of 0.90713, recall of 0.82107, and mAP50 of 0.86888. My strongest YOLO11n run is lower by about 1.42 percentage points in precision, 2.75 points in recall, and 0.81 points in mAP50. The paper’s strongest reported mAP50 is 0.88365 for YOLOv9-tiny, also above my best mAP50.”

“My run’s mAP50-95 is 0.62280, which gives a useful stricter-IoU result for my experiments, but the reference paper does not report mAP50-95 in its final comparison table. I therefore cannot use that metric to claim an improvement over the paper.”

“The setups also differ. The paper reports 16,054 images, 320-by-320 input, batch size 16, and AdamW. My run configuration is YOLO11n, 640-by-640 input, batch size 16, optimizer `auto`, and 250 maximum epochs with patience 10. The model variants, data splits and evaluation protocols are not matched.”

### 7. Conclusion and limitations

“My strongest contribution so far is the systematic exploration and consolidation of multiple model and training configurations, including full and reduced data runs, and an interactive view of their curves and diagnostics. The results identify a strong configuration within my own experiments. They do not show that I outperform the published study overall.”

“To make a fairer head-to-head claim, I would need to reproduce the paper’s model, data split, input size, optimizer and evaluation protocol, then repeat runs with multiple random seeds. I would also report test-set results and measure inference speed and memory on the same hardware.”

## Questions you may be asked

**Why show F1 if the paper does not report it?**  
“F1 summarizes the balance between precision and recall. I calculate it consistently from the reported precision and recall, and label the paper’s value as derived rather than as a number reported by the authors.”

**Did your experiments beat the paper?**  
“I cannot claim that from the current evidence. My best YOLO11n validation metrics are close to, but below, the paper’s YOLO11-nano precision, recall and mAP50. The experiments also use different data and training settings.”

**What does the dashboard add?**  
“It reads the per-run CSVs and plot images, calculates final F1, compares metrics across selected runs, and makes it easier to inspect convergence, confusion matrices, confidence curves and validation predictions. It makes the evidence easier to review; it does not change the underlying results.”

**What is the next experiment?**  
“Run YOLO11n on the same dataset split and evaluation set with matched 320-pixel inputs, batch 16 and AdamW, then compare the same metrics. Repeat with several seeds and report mean and standard deviation.”

## Reference

Fortin, L. V., & Llantos, O. E. (2025). *Performance Analysis of YOLO versions for Real-time Pothole Detection*. Procedia Computer Science, 257, 77–84. https://doi.org/10.1016/j.procs.2025.03.013.

Project documentation: `project_report_ieee.pdf`, `README.md`, saved `args.yaml` and `results.csv` files under `results/`.
