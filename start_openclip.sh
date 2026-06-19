#!/bin/bash
# OpenClip 高光时刻系统启动脚本

export PATH="/data/home/jiangjinghao/.local/bin:$PATH"
cd /data/home/jiangjinghao/openclip

# LLM API Keys
export DOUBAO_API_KEY=ed0df4a7-c765-4c78-8576-ce3701f4dca1
export CUSTOM_OPENAI_API_KEY=sk-Ene1N3ONJDf2y504A104DdAe626140FfBa3848Ff0dEa6112
export QWEN_API_KEY=sk-b3c67260859e45deb54989a1dca58c95

# Editor 服务固定端口
export OPENCLIP_EDITOR_HOST=172.18.140.100
export OPENCLIP_EDITOR_PORT=5402

# 默认 provider 已在 config.py 设置为 custom_openai
# 默认 base_url: https://oneapi-comate.baidu-int.com/v1/chat/completions
# 默认 model: gpt-5.5

# 启动 Streamlit（监听所有接口）
exec uv run python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 5401
