\# Tools



This directory contains command-line scripts grouped by task type.



\## data



Data preprocessing and dataset conversion scripts.



\- `build\_deepfashion2\_landmark\_dataset.py`

\- `export\_deepfashion2\_to\_yolo.py` 后续新增



\## train



Model training scripts.



\- `train\_landmark\_predictor.py`

\- `train\_yolov8\_detector.py` 后续可选



\## infer



Main inference pipelines.



\- `run\_sam\_deepfashion2\_box\_prompt.py`

\- `run\_sam\_single\_image.py`

\- `run\_local\_region\_baseline.py`

\- `batch\_run\_local\_region\_baseline.py`

\- `infer\_landmarks\_for\_predictions.py`

\- `attach\_deepfashion2\_landmarks\_to\_predictions.py`

\- `run\_instance\_pipeline.py` 后续新增



\## eval



Evaluation scripts.



\- `eval\_sam\_deepfashion2.py`



\## analysis



Analysis and inspection scripts.



\- `summarize\_deepfashion2\_landmarks.py`

\- `export\_landmark\_inspection\_by\_category.py`



\## visualize



Visualization utilities.



\- `visualize\_deepfashion2\_annotations.py`

\- `visualize\_deepfashion2\_landmarks.py`

\- `visualize\_landmark\_dataset\_samples.py`

\- `visualize\_landmark\_predictions.py`



\## experiments



Exploratory scripts. These are not part of the main production pipeline.



\- `test\_owlvit\_minimal.py`

\- `batch\_test\_owlvit\_validation.py`



