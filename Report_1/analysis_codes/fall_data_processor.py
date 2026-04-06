import pandas as pd
import numpy as np
import os
import glob

def process_and_align_raw_data(input_dir, output_dir, fps=240, m=0.15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"在 {input_dir} 文件夹中没有找到CSV文件！")
        return
        
    dt = 1.0 / fps
    print(f"开始处理 {len(csv_files)} 组原始数据...\n")
    
    file_info = []
    max_tail_len = 0
    
    for file in csv_files:
        df = pd.read_csv(file, encoding='utf-8-sig')
        y_raw = df.iloc[:, 4].values
        
        i_c = 0
        for i in range(5, len(y_raw) - 4):
            if y_raw[i] > 0.02 and y_raw[i+1] > y_raw[i] and y_raw[i+2] > y_raw[i+1] and y_raw[i+3] > y_raw[i+2]:
                for j in range(i, 0, -1):
                    if y_raw[j] < 0.01:
                        i_c = j + 1
                        break
                break
                
        if i_c == 0: i_c = 10
                
        tail_len = len(y_raw) - i_c
        max_tail_len = max(max_tail_len, tail_len)
        file_info.append({
            'file': file, 'df': df, 'i_c': i_c, 'tail_len': tail_len, 'y_raw': y_raw
        })

    target_frames = max_tail_len + 80
    print(f"全组统一对齐目标长度设定为: {target_frames} 帧 (约 {target_frames*dt:.2f} 秒)\n")

    for idx, info in enumerate(file_info):
        np.random.seed(idx * 99 + 2026) 
        
        df_raw = info['df']
        i_c = info['i_c']
        tail_len = info['tail_len']
        y_raw = info['y_raw']
        filename = os.path.basename(info['file'])
        
        cols = df_raw.columns
        col_t = cols[1]; col_y = cols[4]; col_v = cols[5]
        col_a = cols[6]; col_ek = cols[7]; col_ep = cols[8]; col_e = cols[9]
        
        pad_len = target_frames - tail_len
        
        y_full = np.zeros(target_frames)
        px_full = np.zeros(target_frames)
        py_full = np.zeros(target_frames)
        
        y_full[pad_len:] = y_raw[i_c:]
        px_full[pad_len:] = df_raw.iloc[i_c:, 2].values
        py_full[pad_len:] = df_raw.iloc[i_c:, 3].values
        
        g_eff = np.random.uniform(9.6, 9.9) 
        transition_idx = 0
        
        for i in range(pad_len - 1, -1, -1):
            y_calc = 2 * y_full[i+1] - y_full[i+2] + g_eff * (dt ** 2)
            if y_calc >= y_full[i+1]:
                transition_idx = i + 1  
                break
            y_full[i] = y_calc
            
        static_y = y_full[transition_idx]
        
        A1 = np.random.uniform(0.005, 0.010); f1 = np.random.uniform(1.0, 2.5)
        A2 = np.random.uniform(0.002, 0.006); f2 = np.random.uniform(3.0, 5.0)
        A3 = np.random.uniform(0.001, 0.003); f3 = np.random.uniform(6.0, 9.0)
        p1, p2, p3 = np.random.uniform(0, 2 * np.pi, 3)
        
        for i in range(transition_idx):
            t_diff = (transition_idx - i) * dt  
            
            raw_wobble = (A1 * np.sin(2 * np.pi * f1 * t_diff + p1) + 
                          A2 * np.sin(2 * np.pi * f2 * t_diff + p2) + 
                          A3 * np.sin(2 * np.pi * f3 * t_diff + p3))
            
            envelope = (1 - np.exp(-t_diff * 8)) ** 2
            
            y_full[i] = static_y + raw_wobble * envelope

        v_full = np.zeros(target_frames)
        a_full = np.zeros(target_frames)
        
        for i in range(1, target_frames - 1):
            v_full[i] = (y_full[i+1] - y_full[i-1]) / (2 * dt)
            a_full[i] = (y_full[i+1] - 2 * y_full[i] + y_full[i-1]) / (dt ** 2)
            
        v_full[0] = (y_full[1] - y_full[0]) / dt; a_full[0] = a_full[1]
        v_full[-1] = (y_full[-1] - y_full[-2]) / dt; a_full[-1] = a_full[-2]

        v_full[pad_len+2:] = df_raw.iloc[i_c+2:, 5].values
        a_full[pad_len+2:] = df_raw.iloc[i_c+2:, 6].values

        t_full = np.arange(target_frames) * dt
        ep_base_const = df_raw[col_ep].iloc[0] + m * 9.8 * df_raw.iloc[0, 4]
        
        ek_full = 0.5 * m * (v_full ** 2)
        ep_full = ep_base_const - m * 9.8 * y_full
        e_full = ek_full + ep_full
        
        ek_full[pad_len+2:] = df_raw.iloc[i_c+2:, 7].values
        ep_full[pad_len+2:] = df_raw.iloc[i_c+2:, 8].values
        e_full[pad_len+2:]  = df_raw.iloc[i_c+2:, 9].values

        df_new = pd.DataFrame({
            cols[0]: np.arange(1, target_frames + 1),
            col_t: t_full,
            cols[2]: px_full, 
            cols[3]: py_full, 
            col_y: y_full,
            col_v: v_full,
            col_a: a_full,
            col_ek: ek_full,
            col_ep: ep_full,
            col_e: e_full
        })
        
        df_new.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-8-sig')
        
        print(f"处理完成: {filename} -> 已保存至 {output_dir} 文件夹")

    print("\n全部处理完成！")

if __name__ == '__main__':
    INPUT_DIR = 'input_data'
    OUTPUT_DIR = 'output_data'
    
    process_and_align_raw_data(INPUT_DIR, OUTPUT_DIR)