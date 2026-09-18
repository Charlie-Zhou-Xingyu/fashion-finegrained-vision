"""
Docker deployment and monitoring configuration.

    Dockerfile
    docker-compose.yml
    prometheus.yml
    dashboard.json  (Grafana)
    Dockerfile.trtllm   -- GPU-server-only, serves the Qwen-VL TensorRT-LLM
                           engine via inference/serving/trtllm_server.py.
                           See docs/tensorrt_llm_integration.md.

Status: Pre-implementation except Dockerfile.trtllm. The rest of this
directory (Dockerfile, docker-compose.yml, monitoring) is still to be
built during Week 7-8 of the optimization roadmap.
"""
