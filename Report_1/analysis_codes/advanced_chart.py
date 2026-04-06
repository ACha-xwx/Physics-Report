import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 负号支持
M = 0.15
G = 9.8
H = 5.0

print("🚀 正在生成高阶学术分析图表矩阵 (10组均值版)...")

all_data = []
delta_e_list = []

for i in range(1, 11):
    file_path = f"{i}_data.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gbk')
            except:
                df = pd.read_csv(file_path, encoding='gb18030')

        df = df.dropna(subset=['时间(s)(Time)', '物理位移y(m)(PhysicalY)'])

        df['y_smooth_temp'] = df['物理位移y(m)(PhysicalY)'].rolling(window=21, center=True, min_periods=1).mean()
        df['v_fixed_temp'] = df['y_smooth_temp'].diff(periods=10) / df['时间(s)(Time)'].diff(periods=10)
        df['v_fixed_temp'] = df['v_fixed_temp'].bfill()
        df['E_total_temp'] = 0.5 * M * (df['v_fixed_temp'] ** 2) + M * G * (H - df['y_smooth_temp'])

        E0 = df['E_total_temp'].iloc[15] if len(df) > 15 else df['E_total_temp'].iloc[0]
        final_loss = E0 - df['E_total_temp'].iloc[-5:].mean()
        delta_e_list.append(final_loss)

        all_data.append(df)

fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
sns.boxplot(y=delta_e_list, width=0.4, color="#8172B3", ax=ax1, boxprops=dict(alpha=0.7))
sns.swarmplot(y=delta_e_list, color=".2", size=8, ax=ax1)
ax1.set_title("10组重复实验最终机械能耗散 ($\Delta E$) 稳定性分析", fontsize=15, pad=15)
ax1.set_ylabel("最终耗散能量 (J)", fontsize=12)
ax1.set_xticks([0])
ax1.set_xticklabels(["10次独立苹果坠落实验"], fontsize=12)
plt.tight_layout()
fig1.savefig("Chart1_Boxplot.png")
plt.close(fig1)
print("✅ 图 1：稳定性箱线图生成完毕！")

df_combined = pd.concat(all_data)
df_combined['Time_Round'] = df_combined['时间(s)(Time)'].round(4)
df_avg = df_combined.groupby('Time_Round').mean().reset_index()
df_avg = df_avg.dropna(subset=['物理位移y(m)(PhysicalY)'])

t = df_avg['Time_Round']

df_avg['y_smooth'] = df_avg['物理位移y(m)(PhysicalY)'].rolling(window=21, center=True, min_periods=1).mean()
df_avg['v_fixed'] = df_avg['y_smooth'].diff(periods=10) / t.diff(periods=10)
df_avg['v_fixed'] = df_avg['v_fixed'].bfill()

fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=300)
ax2.plot(t, df_avg['瞬时速度v(m/s)(Velocity)'], color='gray', alpha=0.4, label='CV组分析原始速度')
ax2.plot(t, df_avg['v_fixed'], color='#C44E52', linewidth=2.5, label='修正后速度（去除了误差）')

ax2.set_title("原始速度与修正速度的对比", fontsize=15, pad=15)
ax2.set_xlabel("时间 (s)", fontsize=12)
ax2.set_ylabel("速度 (m/s)", fontsize=12)
ax2.legend(loc="upper left", frameon=True)
plt.tight_layout()
fig2.savefig("Chart2_Denoise.png")
plt.close(fig2)
print("✅ 图 2：滤波降噪对比图 (均值版) 生成完毕！")

df_avg['a_fixed'] = df_avg['v_fixed'].diff(periods=5) / t.diff(periods=5)
df_avg['a_fixed'] = df_avg['a_fixed'].rolling(window=10, center=True).mean().fillna(G)

mask = (t > 0.2) & (t < 0.9)
v_slice = df_avg.loc[mask, 'v_fixed']
a_slice = df_avg.loc[mask, 'a_fixed']

v_square = v_slice ** 2
f_drag = M * (G - a_slice)

v_square_matrix = v_square.values[:, np.newaxis]
k_regression, _, _, _ = np.linalg.lstsq(v_square_matrix, f_drag.values, rcond=None)
k_calculated = k_regression[0]

fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=300)
sns.scatterplot(x=v_square, y=f_drag, color='#4C72B0', alpha=0.6, label='10组均值的离散数据点', ax=ax3)

x_line = np.linspace(0, max(v_square), 100)
y_line = k_calculated * x_line
ax3.plot(x_line, y_line, color='black', linestyle='--', linewidth=2,
         label=f'全局最优回归线\n(拟合斜率 $k$ = {k_calculated:.4f})')

ax3.set_title("平方阻力模型的数据验证 ($f$ vs $v^2$)", fontsize=15, pad=15)
ax3.set_xlabel("速度的平方 $v^2$ ($m^2/s^2$)", fontsize=12)
ax3.set_ylabel("瞬时空气阻力 $f$ (N)", fontsize=12)
ax3.legend(loc="upper left", frameon=True, fontsize=11)
plt.tight_layout()
fig3.savefig("Chart3_Regression.png")
plt.close(fig3)
print("✅ 图 3：回归拟合反推图 (均值版) 生成完毕！")

print("🎉 全部三张高阶学术图表 (10组均值升级版) 已导出到当前文件夹！")