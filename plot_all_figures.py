# plot_all_figures.py
import os, subprocess, sys

os.environ["MPLBACKEND"] = "Agg"   # <— 关键：改用非交互后端

fig_dir = "./Learning Figures"
os.makedirs(fig_dir, exist_ok=True)

for fig in range(4, 13):
    print(f"▶️  正在绘制 Figure {fig} ...")
    cmd = [sys.executable, "reproduce.py", "--figure_num", str(fig)]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅  Figure {fig} 完成")
    except subprocess.CalledProcessError:
        print(f"❌  Figure {fig} 失败，可能缺少数据")
