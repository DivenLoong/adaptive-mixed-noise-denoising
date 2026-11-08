# 混合噪声图像自适应去噪

研究生图像处理课程项目的代码仓库。方法结合**选择性脉冲修复、Haar MAD 噪声估计与自适应非局部均值滤波（NLM）**，包含 Streamlit 交互界面、命令行和可复现实验。

本仓库保存源码、配置、数据来源、测试，以及复现所需的冻结凭据和开发选参记录。报告和视频单独提交；样例图像、虚拟环境和大型实验结果由脚本在本地生成。

## 运行环境

已在 Windows 11、Python 3.13 上验证，算法使用 CPU，无需训练模型或下载模型权重。依赖版本见 `requirements.txt`。

## 快速开始

```powershell
git clone https://github.com/DivenLoong/adaptive-mixed-noise-denoising.git
cd adaptive-mixed-noise-denoising
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m experiments.prepare_data
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

也可安装 Python 3.13 后双击 `start.bat`。默认仅绑定本机，访问 <http://127.0.0.1:8501>；端口占用时可改为 8502。

界面支持灰度／RGB 上传、噪声模拟、算法比较、局部放大、污染掩膜、决策记录及下载。关闭噪声模拟后，输入作为无参考图片处理，不计算 PSNR／SSIM。手动参数探索不修改正式冻结配置。

## 命令行推理

```powershell
.\.venv\Scripts\python.exe -m src.cli "输入图像.png" "输出图像.png"
```

同时生成 JSON 决策记录，支持中文路径。默认最长边 256，可用 `--max-side 512` 探索更大的图片；正式实验仍使用最长边 256。

## 重现实验

```powershell
.\.venv\Scripts\python.exe -m experiments.prepare_data
.\.venv\Scripts\python.exe -m experiments.benchmark --smoke
.\.venv\Scripts\python.exe -m experiments.benchmark
.\.venv\Scripts\python.exe -m experiments.analyze
.\.venv\Scripts\python.exe -m pytest -q
```

也可双击 `run_experiments.bat`。已附带冻结参数，复现时无需重新选参。正式实验会生成逐项 CSV、图表和精确浮点输入／输出；完成分析后可执行全部 16 项测试。

新克隆的仓库不含正式结果，结果审计测试会跳过，核心算法测试仍可运行。`results/raw/tuning.csv` 与 `freeze_receipt.json` 是复现冻结实验所必需的轻量记录，因此保留在仓库中。

`experiments.tune` 仅用于最初开发选参，会拒绝覆盖现有冻结文件。研究新方法时应另建实验目录，避免混入本次冻结结果。正式评估校验协议、参数和原算法源文件的 SHA256；`.gitattributes` 保留冻结 YAML 的原始字节，避免 Git 换行转换使校验失效。

## 算法与消融

1. 结合极值、局部中值偏差、局部 MAD 和邻域支持度计算污染置信度。
2. 只对高置信度位置执行中值替换，RGB 使用共享空间掩膜。
3. 分别计算原图和修复图的逐通道 Haar HH1 居中 MAD，RGB 取算术平均。
4. 以 `h = clip(k * sigma_hat, 1/255, 0.25)` 控制 NLM 强度。

所有 NLM 显式使用 `patch_size=5`、`patch_distance=6`、`fast_mode=True`、`sigma=0.0`、`preserve_range=True`。真实参数、干净参考和污染掩膜只进入离线评价，不进入推理接口。

| 配置 | 预处理 | NLM 强度来源 |
|---|---|---|
| A0 | 无 | 开发集统一固定值 |
| A1 | 全图中值 | 同 A0 |
| A2 | 选择性修复 | 同 A0 |
| A3 | 无 | 原始输入估计 |
| A4 | 选择性修复 | 修复后估计 |
| A5 | 与 A4 相同的预处理数组 | 原始输入估计 |

A4／A5 严格复用同一预处理数组并检查哈希，仅改变估计路径。另有不经过 NLM 的全图中值与选择性修复直接对照。

## 冻结协议

- 开发：`camera`、`coffee`，种子 `17`。测试：`astronaut`、`chelsea`、`coins`、`moon`、`page`、`text`，种子 `42`、`2026`、`1108`。
- 先抗混叠双线性缩放至最长边 256，再生成噪声；灰度／RGB，float32，范围 `[0,1]`。
- G10／G25／G50：高斯 σ=10／25／50；SP05／SP15：椒盐 5%／15%；MIX-L：σ=15、p=5%；MIX-H：σ=35、p=15%。σ 按 255 灰度级计量。
- 先加入逐通道独立高斯扰动并裁剪，再按空间像素椒盐覆盖；RGB 同步设为全黑或全白，盐椒概率各半。
- 污染真值表示“被选中覆盖”，即使像素值恰好不变也计为污染。
- 各基线独立进行开发集有限网格选参，正式测试前冻结，测试结果不用于重新调参。
- 统计先平均图像／条件内三个种子，再跨图像汇总，不把重复测量当作独立自然图像。

## 已完成实验的结果

本地正式实验含 126 组输入、1764 条评价记录，全部经过浮点输出复算。14 个记录标签包含 A0／固定 NLM 别名和两种预处理直接输出，不代表 14 个独立算法。

| 方法 | 平均 PSNR / dB | 平均 SSIM |
|---|---:|---:|
| 固定 NLM | 24.4466 | 0.5931 |
| 完整方法 A4 | 28.5569 | 0.7991 |

- 完整方法相对固定 NLM 平均 +4.1103 dB；42 个图像×条件配对有 36 个为正。
- H1 整体未获支持：纯椒盐与混合条件的结构保持表现不同。
- H2 在混合条件获支持：生成参数恢复 MAE 从 11.5049 降至 1.6991 灰度级；该指标不是修复后残余误差标准差。
- H3 有条件成立：A3−A0 为 −1.4346 dB，A4−A2 为 +3.0524 dB。
- A4−A5 总体只有 +0.1819 dB，MIX-H 还有反例；估计生成参数更准确不保证重建质量更高。

结论仅限于六张测试图像和指定合成条件。固定 NLM 使用统一强度，不是逐条件最优参数；未与 BM3D 或现代深度方法作全面比较，也不宣称基础算子原创。

## 目录与来源

```text
app.py                 Streamlit 界面
src/                   算法、图像 I/O、指标和 CLI
experiments/           数据准备、选参、评估和分析
configs/               实验协议与冻结参数
tests/                 算法及真实结果一致性检查
data/                  来源和图像尺寸清单
results/raw/           冻结凭据与开发选参记录
start.bat              Windows 界面启动脚本
run_experiments.bat     Windows 复现脚本
```

数据来自 scikit-image 示例集，逐图来源与使用条件见 [DATA_SOURCES.md](data/DATA_SOURCES.md)。原始图像不随仓库发布，由 `prepare_data` 从安装包示例数据重建。

基础算子来自 NumPy、SciPy、OpenCV、scikit-image 和 PyWavelets，贡献为处理策略、模块对照及复现流程。

参考：[NLM 原论文](https://www.ipol.im/pub/art/2011/bcm_nlm/)、[Restoration API](https://scikit-image.org/docs/stable/api/skimage.restoration.html)、[PyWavelets DWT](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html)。
