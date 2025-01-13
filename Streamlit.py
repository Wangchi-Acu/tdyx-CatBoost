import streamlit as st
import catboost as cb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 加载模型
CB_model = joblib.load('CatBoost_best_auc_model.pkl')

# 创建SHAP解释器
explainer = shap.TreeExplainer(CB_model)

# Streamlit 用户界面
# 添加团队logo
st.image("jsszyylogo.png", width=500)  # 更改url_to_your_logo.png为你的logo图片链接，调整width为适当的大小

# 使用Markdown来定制标题的字体大小
st.markdown('<h1 style="font-size:42px;">“通督养心”针刺组方治疗失眠症疗效预测</h1>', unsafe_allow_html=True)

best_threshold = 0.40  # 这是你确定的最佳阈值

# 创建列布局
col1, col2 = st.columns(2)
with col1:
    DUR = st.number_input("病程（月）:", min_value=0.0, max_value=500.0, value=1.0)
    N1P = st.number_input("N1期占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
    N2P = st.number_input("N2期占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
    
    
with col2:
    PSQI = st.number_input("PSQI总分（分）:", min_value=0.0, max_value=50.0, value=1.0)
    PD = st.number_input("PSQI睡眠效率得分（分）:", min_value=0.0, max_value=10.0, value=1.0)
    HPRDS = st.number_input("睡眠期间最高脉率（次/分钟）:", min_value=0.0, max_value=200.0, value=1.0)
    

# 进行预测
if st.button("预测"):
    feature_values = [DUR, HPRDS, N1P, N2P, PSQI,PD]
    feature_names = ["DUR", "HPRDS", "N1P", "N2P", "PSQI", "PD"]
    prediction_proba = CB_model.predict_proba([feature_values])[0, 1]
    st.write(f"该患者经“通督养心”针刺组方治疗后PSQI减分率≥50%的概率: {prediction_proba:.2%}")
    
    if prediction_proba >= best_threshold:
        st.write("该患者可能经“通督养心”针刺组方治疗后显效")
    else:
        st.write("该患者可能经“通督养心”针刺组方治疗后不显效")

    # 计算 SHAP 值并生成力图
    shap_values = explainer.shap_values([feature_values])
    plt.figure(figsize=(12, 5))  # 调整图的尺寸
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True, show=False)
    plt.savefig("shap_force_plot.png", bbox_inches='tight')
    st.image("shap_force_plot.png")