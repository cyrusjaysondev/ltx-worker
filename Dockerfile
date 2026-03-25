FROM thankfulcarp/wan2gp-docker:runpod-latest

# Install flash-attn at build time (survives restarts, no runtime compile)
RUN pip install flash-attn --no-build-isolation

# Install worker dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy handler
COPY handler.py /workspace/handler.py

# Ensure output dir exists
RUN mkdir -p /workspace/wan2gp/outputs

CMD ["python3", "-u", "/workspace/handler.py"]
