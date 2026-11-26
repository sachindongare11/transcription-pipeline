FROM python:3.11-slim
WORKDIR /app
# PyAV wheels bundle FFmpeg's libraries, so no apt-get ffmpeg is needed.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
# Bake the model into the image: no cold-start download, works without egress.
ARG MODEL=base
ENV TRANSCRIBER_MODEL_SIZE=${MODEL} HF_HOME=/opt/models
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL}', device='cpu', compute_type='int8')" \
 && chmod -R a+rX /opt/models
RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "--factory", "transcriber.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
