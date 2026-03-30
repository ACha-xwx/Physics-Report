import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APP_TITLE = "自由落体实验数据集成分析与多维对标平台"
M = 0.15  # 质量 (kg)
G = 9.8  # 重力加速度
H = 5.0  # 初始高度

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

st.sidebar.header("导航与控制面板")
st.sidebar.markdown("---")

view_mode = st.sidebar.radio("模式选择", ["物理全景对标 (3D)", "数据深度分析 (2D)"])

selected_exp_id = st.sidebar.selectbox(
    "选择实验组次",
    [i for i in range(1, 11)],
    index=0,
    disabled=(view_mode == "物理全景对标 (3D)")
)

st.sidebar.markdown("---")
k_val = st.sidebar.slider("理论平方阻力系数 k", 0.0000, 0.0050, 0.0015, 0.0001, format="%.4f")

@st.cache_data
def load_and_fix_data(file_path, window=21):
    df = pd.read_csv(file_path)
    df['y_smooth'] = df['物理位移y(m)(PhysicalY)'].rolling(window=window, center=True, min_periods=1).mean()
    step = 10
    df['v_fixed'] = df['y_smooth'].diff(periods=step) / df['时间(s)(Time)'].diff(periods=step)
    df['v_fixed'] = df['v_fixed'].fillna(0)
    df['Ek_fixed'] = 0.5 * M * (df['v_fixed'] ** 2)
    df['Ep_fixed'] = M * G * (H - df['y_smooth'])
    df['E_total_fixed'] = df['Ek_fixed'] + df['Ep_fixed']
    E0 = df['E_total_fixed'].iloc[15] if len(df) > 15 else df['E_total_fixed'].iloc[0]
    df['Delta_E_measured'] = E0 - df['E_total_fixed']
    return df

st.title(APP_TITLE)
st.markdown("---")

if view_mode == "物理全景对标 (3D)":
    st.markdown("### 多实验组机械能损耗 3D 对标全景图")
    st.markdown("此视图展示全部 10 组实验数据的时空一致性。通过观察三维空间内机械能耗散轨迹的重合度，验证系统可复现性。")

    fig_3d = go.Figure()

    colors_3d = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(1, 11):
        file_path = f"{i}_data.csv"
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)

        df = df.iloc[::4].copy()
        df['y_smooth'] = df['物理位移y(m)(PhysicalY)'].rolling(window=7, center=True, min_periods=1).mean()
        step = 3
        df['v_fixed'] = df['y_smooth'].diff(periods=step) / df['时间(s)(Time)'].diff(periods=step)
        df['v_fixed'] = df['v_fixed'].fillna(0)
        df['Ek_fixed'] = 0.5 * M * (df['v_fixed'] ** 2)
        df['Ep_fixed'] = M * G * (H - df['y_smooth'])
        df['E_total_fixed'] = df['Ek_fixed'] + df['Ep_fixed']
        E0 = df['E_total_fixed'].iloc[4]
        df['Delta_E'] = E0 - df['E_total_fixed']

        fig_3d.add_trace(go.Scatter3d(
            x=df['时间(s)(Time)'],
            y=[i] * len(df),
            z=df['Delta_E'],
            mode='lines', name=f'实验组 {i}',
            line=dict(color=colors_3d[i - 1], width=4)
        ))

    fig_3d.update_layout(
        height=650, template="plotly_dark", paper_bgcolor="black", margin=dict(l=0, r=0, b=0, t=20),
        scene=dict(
            xaxis_title='下落时间 (s)', yaxis_title='实验组次', zaxis_title='耗散能量 ΔE (J)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(tickvals=list(range(1, 11)), autorange="reversed", showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.3))
        ),
        hovermode=False
    )
    st.plotly_chart(fig_3d, use_container_width=True)

else:
    file_path = f"{selected_exp_id}_data.csv"
    st.markdown(f"### [实验组 {selected_exp_id}] 数据深度分析与验证")
    st.markdown(f"此视图专注于特定实验组数据的动力学与机械能守恒验证。通过调节理论 k 值，可进行多维对标。")

    if not os.path.exists(file_path):
        st.error(f"找不到对应数据文件: {file_path}")
    else:
        df_fixed = load_and_fix_data(file_path)
        t = df_fixed['时间(s)(Time)'].values

        vt = np.sqrt((M * G) / k_val) if k_val > 0 else 9999
        x_theory = (M / k_val) * np.log(np.cosh(np.sqrt(k_val * G / M) * t)) if k_val > 0 else 0.5 * G * t ** 2
        v_theory = vt * np.tanh(G * t / vt) if k_val > 0 else G * t
        E_theory = 0.5 * M * (v_theory ** 2) + M * G * (H - x_theory)
        Delta_E_theory = (M * G * H) - E_theory

        col1, col2, col3 = st.columns(3)
        col1.metric("当前理论极限速度 (vt)", f"{vt:.2f} m/s" if k_val > 0 else "∞")
        col2.metric("当前理论耗散做功总和", f"{Delta_E_theory[-1]:.4f} J")
        energy_loss_real = df_fixed['Delta_E_measured'].iloc[-1]
        col3.metric("落地实测机械能损耗", f"{energy_loss_real:.4f} J")
        st.markdown("<br>", unsafe_allow_html=True)

        from plotly.subplots import make_subplots

        fig_2d = make_subplots(
            rows=1, cols=3,
            subplot_titles=("动力学验证 (v-t)", "机械能衰减验证 (E-t)", "耗散能量对标 (ΔE-t)")
        )

        fig_2d.add_trace(go.Scatter(x=t, y=df_fixed['v_fixed'], name='实测速度', line=dict(color='blue', width=2)),
                         row=1, col=1)
        fig_2d.add_trace(go.Scatter(x=t, y=v_theory, name='平方阻力理论曲线', line=dict(color='red', dash='dash')),
                         row=1, col=1)

        fig_2d.add_trace(go.Scatter(x=t, y=df_fixed['Ek_fixed'], name='动能', line=dict(color='blue', width=1)), row=1,
                         col=2)
        fig_2d.add_trace(go.Scatter(x=t, y=df_fixed['Ep_fixed'], name='势能', line=dict(color='green', width=1)), row=1,
                         col=2)
        fig_2d.add_trace(
            go.Scatter(x=t, y=df_fixed['E_total_fixed'], name='实测总机械能', line=dict(color='orange', width=2)),
            row=1, col=2)
        fig_2d.add_trace(go.Scatter(x=t, y=E_theory, name='理论总机械能', line=dict(color='black', dash='dot')), row=1,
                         col=2)

        fig_2d.add_trace(
            go.Scatter(x=t, y=df_fixed['Delta_E_measured'], name='实测损耗量 (ΔE)', line=dict(color='purple', width=2)),
            row=1, col=3)
        fig_2d.add_trace(
            go.Scatter(x=t, y=Delta_E_theory, name='理论做功预测 (W_f)', line=dict(color='black', dash='dash')), row=1,
            col=3)

        fig_2d.update_layout(template="plotly_white", hovermode="x unified", height=450, margin=dict(t=30))
        fig_2d.update_yaxes(title_text="速度 (m/s)", row=1, col=1)
        fig_2d.update_yaxes(title_text="绝对能量 (J)", row=1, col=2)
        fig_2d.update_yaxes(title_text="耗散能量 (J)", row=1, col=3)

        st.plotly_chart(fig_2d, use_container_width=True)