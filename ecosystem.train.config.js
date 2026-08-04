// PM2 entry for LoRA training (train_lora.py).
//
// One-shot job: runs once, exits when training finishes. Does not autorestart.
//
// Start:  pm2 start ecosystem.train.config.js
// Logs:   pm2 logs teutonic-train-lora
// Stop:   pm2 stop teutonic-train-lora

module.exports = {
  apps: [{
    name: "teutonic-train-lora",
    script: "train_lora.py",
    interpreter: "/workspace/teu_lora/.venv/bin/python",
    cwd: "/workspace/teu_lora",
    exec_mode: "fork",
    instances: 1,
    autorestart: false,
    max_restarts: 0,
    kill_timeout: 30000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: {
      PYTHONUNBUFFERED: "1",
      HF_HOME: "/workspace/.hf_home",
      PYTHONPATH: "/workspace/teu_lora/newking",
      PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True",
      TOKENIZERS_PARALLELISM: "false",
      // Active run config — swap to a different config file to change experiments.
      TEUTONIC_TRAIN_CONFIG: "/workspace/teu_lora/configs/v7_sampling.yaml",
      TEUTONIC_LORA_BATCH: "1",
      TEUTONIC_LORA_GRAD_ACCUM: "16",
      TEUTONIC_LORA_SEQ_LEN: "2048",
    },
  }],
};
