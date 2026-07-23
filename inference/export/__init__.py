"""
ONNX export and TensorRT engine build scripts.

One script per model. These are NOT run during docker build — they are
run once per model version change. The resulting .engine files are stored
in a model registry (S3/Git LFS).

    export_yolo.py
    export_mobilesam.py
    export_landmark.py
    export_dino.py
    export_fashionpedia.py
    export_attributes.py

Status: Pre-implementation. No export scripts exist yet.
To be built during Week 3-4 of the optimization roadmap.
"""
