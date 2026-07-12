# 反射坐标、参考场表示与 POP/Zemax 对比验证报告

## 引言

本报告验证反射光路中 POP API 与 Zemax ZBF 结果不一致的根因修正。重点
不是让某一个相位残差数值变小，而是保证物理光程、局部坐标、参考相位和
参考相对复场的定义彼此一致。验证对象包括 S7→S8 自由空间段，以及完整
的 1024×1024 POP/Zemax 基准仿真。

## 方法

Zemax ZBF 的参考相对场先转换到 POP 的复场约定。端面物理场写为

\[
U_{\mathrm{ZBF}}=\operatorname{conj}(E_x)\exp(i\sigma\Phi_{\mathrm{ZBF}}),
\]

其中 \(\sigma=\operatorname{sign}(\hat k\cdot\hat z_{\mathrm{local}})\)。POP
状态同时保存原生参考相对场 \(\chi\) 和与它配套的提升相位 \(\phi_\chi\)，
物理场由 \(U_{\mathrm{POP}}=\chi\exp(i\phi_\chi)\) 得到；不把任意
时刻的 `wfarr` 与 ZBF 参考相位直接相乘。

自由空间传播距离使用沿实际光线累加的正物理光程。局部坐标轴符号只用于
ZBF 参考相位和 ZBF 束腰参数的坐标转换。理想平面镜的横向场使用
Householder 矩阵

\[
H=I-2vv^{\mathsf T},\quad
v=\frac{\hat k_s-\hat k_t}{\lVert\hat k_s-\hat k_t\rVert},
\]

并以 \(T=E_{t,\perp}^{\mathsf T}HE_{s,\perp}\) 进行网格坐标变换。

## 结果

### S7→S8 路径隔离

使用相同的 S7 ZBF 残差场、采样网格和 PROPER 传播器进行 A/B 验证：

| 传播距离 | 相位 RMS | 强度 RMS |
|---:|---:|---:|
| $+368.600912\,\mathrm{mm}$ | $2.03888\times10^{-4}$ 波 | $7.41144\times10^{-4}\%$ |
| $-368.600912\,\mathrm{mm}$ | $5.81608\times10^{-4}$ 波 | $2.06337\%$ |

最终代码采用正物理光程。S7、S8 两端的局部轴符号均为 −1；该符号没有
再次乘入 PROPER 距离。

在最终代码和 ZBF 参考场配对规则下，15 面路径收据给出：

- S7→S8：相位 RMS $2.03888\times10^{-4}$ 波，强度 RMS $7.41144\times10^{-4}\%$；
- S12→S13：相位 RMS $6.16443\times10^{-5}$ 波，强度 RMS $1.73903\times10^{-3}\%$；
- S13→S14：相位 RMS $6.40211\times10^{-6}$ 波，强度 RMS $2.64687\times10^{-3}\%$；
- S14→S15：相位 RMS $2.27422\times10^{-10}$ 波，强度 RMS $1.16570\times10^{-13}\%$。

### 完整 1024×1024 基准

运行命令：

```text
python -B -m sandbox.zemax_pop_benchmark.run_biconic_zemax_pop_benchmark --mode gaussian --zmx-path D:\BTS\sandbox\Zemax_baseline\biconic_focus_test_expand_validation.zmx --output-dir D:\w\zfsi\output\zemax_compare_20260713_reference_paired --grid-size 1024 --physical-size-mm 348
```

结果文件：

- [summary.json](D:/w/zfsi/output/zemax_compare_20260713_reference_paired/summary.json)
- Zemax 各面 ZBF：`C:\Users\liyitong\Documents\Zemax\POP\BEAMFILES\biconic_gaussian_direct_*.ZBF`

共匹配 17 个 ZBF 面。采用成对物理场比较后，关键面结果为：

| 面 | 相位 RMS | 强度 RMS |
|---:|---:|---:|
| S8 | $1.04009\times10^{-4}$ 波 | $1.98848\times10^{-1}\%$ |
| S9 | $4.79932\times10^{-3}$ 波 | $1.23791\times10^{-1}\%$ |
| S20（VM0） | $5.20603\times10^{-3}$ 波 | $1.12681\%$ |
| S23（VM1） | $2.28486\times10^{-2}$ 波 | $5.57162\times10^{-1}\%$ |

绝对能量比约为 8.65，来自 Gaussian 直接输入与 Zemax ZBF 峰值归一化的
幅值标度差异；表中的强度 RMS 使用各自峰值归一化，专门衡量空间分布，
不把幅值标定差误判为传播相位误差。

## 讨论

A/B 结果说明，反射后的局部 z 轴反向是真实的坐标事实，但它不是把 PROPER
传播距离取负的理由。负距离会把传播方向再次反转，导致同一反射符号被重复
计数。正确的实现是：物理光程始终正向累加，局部轴符号只在 ZBF 参考相位、
ZBF q 参数和端面物理场转换处使用。

状态边界显式保存参考相对场及其配套提升相位后，S7→S8 的公共参考比较闭合，
说明问题不在 ZBF 端点采样面或自由空间传播核。对于多次反射，符号由主光线
方向与原始局部 z 轴的内积计算，不依赖反射次数；横向场则由几何反射矩阵
变换，因而不依赖轴对称场的数值巧合。
