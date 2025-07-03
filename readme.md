# ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge)
[![Research Paper](https://img.shields.io/badge/Paper-brightgreen.svg?style=for-the-badge)](https://likaixin2000.github.io/papers/ScreenSpot_Pro.pdf)
[![Huggingface Dataset](https://img.shields.io/badge/Dataset-blue.svg?style=for-the-badge)](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-8A2BE2?style=for-the-badge)](https://gui-agent.github.io/grounding-leaderboard)

## 📢 Updates
(May 19, 2025) 🔥🔥🔥 We're excited to introduce our new model, [SE-GUI](https://github.com/YXB-NKU/SE-GUI)!  It achieves **47.2%** accuracy with a **7B model** and **35.9%** with a **3B model** — trained on just **3k open-source samples**. Check out [the arxiv paper](https://arxiv.org/pdf/2505.12370)!

(Feb 21, 2025) We’re excited to see our work acknowledged and used as a benchmark in several great projects: [Omniparser v2](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/), [Qwen2.5-VL](https://arxiv.org/pdf/2502.13923), [UI-TARS](https://arxiv.org/pdf/2501.12326), [UGround](https://x.com/ysu_nlp/status/1882618596863717879), [AGUVIS](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/issues/2), ...

(Jan 4, 2025) The paper and dataset are released. Please also check out [ScreenSpot-v2-variants](https://huggingface.co/datasets/likaixin/ScreenSpot-v2-variants) which contains more instruction styles (original instruction, action, target UI description, and negative instructions).

## Set Up

Before you begin, ensure your environment variables are set:

- `OPENAI_API_KEY`: Your OpenAI API key.

## Evaluation
Use the shell scripts to launch the evaluation. 
```bash 
bash run_ss_pro.sh
```
or
```bash 
bash run_ss_pro_cn.sh
```

# Citation
Please consider citing if you find our work useful:
```plain
@inproceedings{
    li2025screenspotpro,
    title={ScreenSpot-Pro: {GUI} Grounding for Professional High-Resolution Computer Use},
    author={Kaixin Li and Meng Ziyang and Hongzhan Lin and Ziyang Luo and Yuchen Tian and Jing Ma and Zhiyong Huang and Tat-Seng Chua},
    booktitle={Workshop on Reasoning and Planning for Large Language Models},
    year={2025},
    url={https://openreview.net/forum?id=XaKNDIAHas}
}
```
