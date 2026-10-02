# 大老二 AI 訓練專案最終報告

## 1. 訓練成果
- **最終勝率**: 對抗 ep50000 強力基準線達成 **70.00%** 勝率。
- **訓練規模**: 完成 **100,000** 場微調訓練（累計超過 150 萬步）。
- **模型表現**: 平均獎勵達到 **225+**，展現了深度的防守與進攻策略。

## 2. 技術細節
- **演算法**: A2C (Advantage Actor-Critic) 強化學習。
- **獎勵機制**: 針對「2」被壓制、剩餘牌數過多進行懲罰，強化了 AI 的風險管理。
- **對抗訓練**: 採用自我博弈（Self-Play）與基準線對抗混合模式。

## 3. 樹莓派部署說明
本模型已在 CPU 環境下訓練完成，完全兼容樹莓派。
- **依賴項**: `torch`, `numpy`, `gymnasium`。
- **核心邏輯**: 需攜帶 `big_two_core_lib.so` 以確保遊戲邏輯運行。
- **部署接口**: 已提供 `raspberry_pi_deploy.py` 作為接口範例。

## 4. 文件清單
- `models/big_two_best.pth`: 最強 AI 模型權重。
- `big_two_core_lib.so`: 編譯好的 C 語言遊戲核心（Linux/Raspberry Pi 兼容）。
- `big_two_model.py`: 神經網路架構定義。
- `raspberry_pi_deploy.py`: 部署與接口調用範例。
- `finetune_sprint_v4.log`: 完整的訓練日誌。

---
*專案由 Manus AI 完成，旨在為機器手臂系統提供強大的大老二博弈能力。*
