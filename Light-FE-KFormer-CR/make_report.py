from pathlib import Path
import json,os
PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT.parent/'result'
os.environ['MPLCONFIGDIR']=str(ROOT/'cache'/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
v=json.loads((ROOT/'verification.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4))
for name,color in [('baseline_no_openfe','#64748b'),('full_openfe','#0891b2')]:
    hist=json.loads((ROOT/name/'history.json').read_text())
    axes[0].plot(range(1,len(hist['train_loss'])+1),hist['train_loss'],label=name,color=color)
    axes[1].plot(range(1,len(hist['val_loss'])+1),hist['val_loss'],label=name,color=color)
for ax,title in zip(axes,['Training loss','Validation loss']):
    ax.set(title=title,xlabel='Epoch',ylabel='Loss');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'training_curves.png',dpi=160);plt.close(fig)
lines=['# Light-FE-KFormer-CR 虚拟数据试跑报告','','## 结论','','两条链路均已完成训练、测试及模型重载复验。原项目源码未修改，适配和兼容处理集中于 项目代码目录/run_trail.py。','','- 不含 OpenFE：AE → 五组 Token → KFormer → CI/CR。','- 包含 OpenFE：OpenFE → AE → 五组 Token → KFormer → CI/CR。','- 原项目测试：15 passed（logs/pytest_final.log）。','- 重载模型及预处理器，重新生成测试 Token 与预测，与保存结果一致（容差 1e-5）。','','## 数据适配','','- 原始数据 288 行、48 块绿地；每块绿地的 3 次 Normal 观测取均值，与每次 Heatwave 观测配对，得到 144 对。','- 沿用数据包 splits：训练 34 块/102 对，验证 7 块/21 对，测试 7 块/21 对。三个集合的绿地 ID 无交叉。','- area 使用原始 area_ha；temperature 使用 Tmax_C；building 使用 built_frac；green 使用 green_frac；NDVI 使用对应尺度 NDVI。','- 缺失 green_ratio 和三个尺度 NDBI，明确省略这四个输入，不以其他指标冒充。五个语义组均保留。','- LST、CI_normal_mean_C、CR_ratio 等标签或标签派生量不作为输入特征。CI 用作监督目标，CR 重新按 ci_heat / ci_normal 计算。','- 详细字段映射及数据哈希保存在各运行目录 config.json。','','## 运行设置','','CPU，seed=42；AE 最多 150 轮；主模型最多 200 轮，两次均实际完成 200 轮，模型使用验证损失最佳权重。OpenFE 仅在训练集拟合，每组本次筛选出 1 个增强特征，详见 full_openfe/selected_features.json。','','## 测试结果','','|链路|CR MAE|CR RMSE|CR R²|CI normal MAE (°C)|CI heat MAE (°C)|','|---|---:|---:|---:|---:|---:|']
for name in v:
    m=v[name]['summary']['metrics']
    lines.append(f"|{name}|{m['cr_mae']:.6f}|{m['cr_rmse']:.6f}|{m['cr_r2']:.6f}|{m['ci_normal_mae']:.6f}|{m['ci_heat_mae']:.6f}|")
lines += ['','训练集均值预测基线：CR MAE=0.066279，CI normal MAE=0.455681°C，CI heat MAE=0.483748°C。含 OpenFE 的 CR 误差优于此基线，但两项 CI 误差未优于基线。样本量小、数据为虚拟数据，本结果用于流程验证，不支持真实环境预测精度结论。','','## 环境兼容处理','','- Python 3.12 独立环境在 ../environment/.venv312；由 uv 安装依赖；精确版本见 ../environment/requirements-lock.txt。','- OpenFE 额外需要 matplotlib，已补齐。scikit-learn 固定 1.5.2，保留 OpenFE 依赖的 squared 参数；pandas=2.2.3，numpy=1.26.4。','- 当前受限 Windows 会话无法创建 OpenFE 多进程管道。脚本改用单工作线程，并在任务提交时深拷贝调用对象和参数以保留隔离。此方式已成功运行，但不等同于验证默认多进程模式。','- 日志中的 LightGBM 无正增益分裂及 sklearn 弃用提示不妨碍本次完成。早期失败日志尚待清理，最终日志为 baseline_compatible.log、openfe_compatible.log。','- OpenFE 自身部分特征使用批次统计量；本次未对其进行面向生产的泄漏和分布稳定性审计。','','## 文件导航','','- full_openfe/：完整链路的配对数据、拆分、Token、模型、预处理器、指标、预测、训练历史、所选特征。','- baseline_no_openfe/：基础链路的同类文件。','- token_pipeline.pkl 与 light_fe_kformer_cr.pt 必须配套使用，加载前应用 config.json 中的特征组配置；示例见 verify_trial.py。','- verification.json：重载及一致性验证结果。','- training_curves.png：训练与验证损失曲线。','- logs/：安装、测试、训练和验证日志。','- ../environment/.venv312：有效运行环境；旧环境和安装缓存因删除权限限制尚未清理。运行缓存仍在 result/tmp 和 result/cache。','','## 复跑（PowerShell）','','```powershell',f"Set-Location '{PROJECT}'","$env:PYTHONDONTWRITEBYTECODE='1'","& '../environment/.venv312/Scripts/python.exe' -B -u './run_trail.py' --openfe","& '../environment/.venv312/Scripts/python.exe' -B -u './run_trail.py'","& '../environment/.venv312/Scripts/python.exe' -B './verify_trial.py'",'```','','复跑会更新对应目录中的同名结果。']
(ROOT/'试跑报告.md').write_text('\n'.join(lines),encoding='utf-8')
print('Report and training curves saved.')



