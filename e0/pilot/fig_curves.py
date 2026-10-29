import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
S1, S2 = json.load(open('scan.json')), json.load(open('scan2.json'))
def get(scen, alpha, src):
    for r in src:
        if r['scen'] == scen and abs(r['alpha'] - alpha) < 1e-9: return r
panels = [(get('rot_sym', 1.0, S1), 'rot90, h=256, α=1（惰性）'),
          (get('rot_sym', 0.05, S1), 'rot90, h=256, α=0.05（富特征）'),
          (get('rot_h64', 0.05, S2), 'rot90, h=64, α=0.05（窄网络 + 富特征）')]
BLUE, ORANGE, INK, INK2, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURF)
for ax, (res, title) in zip(axes, panels):
    ax.set_facecolor(SURF)
    rs = [float(k) for k in res['true']]
    tr = [res['true'][k] for k in res['true']]; pr = [res['lin'][k] for k in res['lin']]
    xt = [r for r, t in zip(rs, tr) if t >= 0]; yt = [t for t in tr if t >= 0]
    xp = [r for r, t in zip(rs, pr) if t >= 0]; yp = [t for t in pr if t >= 0]
    ax.plot(xt, yt, color=BLUE, lw=2, marker='o', ms=5, label='真实训练 T(r)', zorder=3)
    ax.plot(xp, yp, color=ORANGE, lw=2, marker='s', ms=5, ls='--', label='P_lin2 预测', zorder=2)
    bt = xt[yt.index(min(yt))]; bp = xp[yp.index(min(yp))]
    ax.axvline(bt, color=BLUE, lw=1, alpha=0.5); ax.axvline(bp, color=ORANGE, lw=1, alpha=0.5, ls='--')
    ax.set_xlabel('重放比例 r', color=INK2); ax.set_yscale('log')
    ax.tick_params(colors=INK2); ax.grid(True, color=GRID, lw=0.8); [s.set_color(GRID) for s in ax.spines.values()]
    ax.set_title(f'{title}\n最优 r：真实 {bt}，预测 {bp}', color=INK, fontsize=11)
axes[0].set_ylabel('进入容差带的步数（对数轴）', color=INK2)
axes[0].legend(frameon=False, loc='upper center')
fig.suptitle('线性化预测（P_lin2）给出的 T(r) 与真实训练对比：步数尺度会偏，最优比例基本不偏', color=INK, fontsize=12)
fig.tight_layout()
fig.savefig('fig_T_curves.png', dpi=150, facecolor=SURF)
print('saved')
