# S7/S8 Phase Interpretation Audit

## 审计结论

本审计不接受“RMS 很小，所以物理相位一定算对了”这种结论。也不把参考相位当作最终评价目标。真正要评价的是同一物理平面上的物理总复场：

```text
U(x,y) = A(x,y) * exp(i * Phi_phys(x,y))
```

因此，最终问题应表述为：

> POP/PROPER 传播后重构出的 `U_S8(x,y)`，是否真的等于 Zemax 在同一平面、同一坐标、同一符号约定下的物理总场 `U_Zemax,S8(x,y)`？

S7/S8 目前能够成立的结论应当更窄：

> 如果接受当前 ZBF 端点参考相位公式为从 ZBF 残余复场恢复物理总场的正确公式，那么 S7 到 S8 的残余复场传播及端点物理总场重构是局部自洽的。

这句话有三个限制。

第一，它说的是 S7/S8 局部段，不包括 VM1、S8 到 S12，也不包括整条光路。第二，它仍然依赖一个被接受的物理总场重构公式，尤其是 `Phi_ZBF` 的解析表达式和符号；它不是与参考相位公式无关的绝对证明。第三，`0.0319913 waves` 不是应当被消去的自由空间传播错误，而是一个混合参考相位比较得到的残差；它仍然有诊断价值，但不能被解释为传播核失败。

因此，原文中 “representation-ledger problem” 的写法不适合作为学术表述。这里的 `ledger` 应改写为：

- 场表示；
- 参考相位约定；
- 参考相位去除后的残余复场；
- 端点物理总场重构。

## 术语替换

建议完全避免把 `ledger` 当作正式术语。更规范的变量定义如下：

| 原笔记口语 | 推荐术语 | 含义 |
|---|---|---|
| residual ledger | reference-relative field representation | 从总场中去掉某个参考相位后得到的残余复场表示 |
| reference ledger | reference phase convention | 被去除或重新乘回去的参考相位定义 |
| physical ledger | physical total field reconstruction | 用残余复场和参考相位重构总物理场的过程 |
| mixed-ledger residual | mixed-reference residual | 两边使用不同参考相位约定重构后得到的表观残差 |

以后报告中应直接说“mixed-reference residual”，不要说“mixed-ledger residual”。如果历史日志保留 `ledger`，必须立即解释它只是“场表示与参考相位约定”的口语标签，不是光学理论术语。

## 场表示与参考相位

标准的复场分解应写成：

```text
U(x,y) = u(x,y) * exp(i * Phi_ref(x,y))
```

其中：

- `U(x,y)` 是物理总复场；
- `u(x,y)` 是相对于某个参考相位的残余复场；
- `Phi_ref(x,y)` 是参考相位。

同一个物理场可以有不同表示：

```text
U = u_A * exp(i * Phi_A)
U = u_B * exp(i * Phi_B)
```

这不是说参考相位是最终物理量。参考相位只是表示变换的一部分；真正的物理量仍然是 `U`。但是在当前数据链中，ZBF 原始 `Ex` 和 PROPER 的 `wfarr` 都不是直接给出的 `U`，而是参考相位去除后的表示。要比较物理全相位，必须先把它们恢复成 `U`。

因此，如果参考相位公式或符号错了，错的不是“中间过程”而已，最终重构出的 `U` 也会错。数学上：

```text
u * exp(i * Phi_ref_wrong) != u * exp(i * Phi_ref_true)
```

所以，参考相位不会改变真实物理场本身，但会改变我们从文件和内部数组中重构出来的物理总场。审计参考相位，是为了判断最终物理全相位是否真的被正确恢复，而不是把中间参考相位当成目标。

## ZBF 到 POP/PROPER 的变换链条

Zemax ZBF 原始 `Ex` 与 POP/PROPER 内部相位符号约定相反。当前代码中的边界转换是：

```text
u_ZBF = conj(Ex_Zemax)
```

如果要在 POP/PROPER 约定下重构 ZBF 端点的物理总场，应写成：

```text
U_ZBF,POP = conj(Ex_Zemax) * exp(i * axis_sign * Phi_ZBF)
```

其中 `axis_sign` 表示 ZBF 表面局部 +z 方向与当前 POP 传播方向是否同向。这个符号不能省略，因为反向表面会改变参考相位在传播坐标中的符号。

对 S7/S8 当前这种已经验证的等半径情形，ZBF 参考相位使用的是精确球面参考：

```text
Phi_ZBF = k * sign(R) * (sqrt(abs(R)^2 + x^2 + y^2) - abs(R))
k = 2*pi*n/lambda
```

这里的 `R` 来自 ZBF header 中通过 Rayleigh gate 后的 `zx/zy` 有效参考半径。它不应被混同为高斯光束曲率半径 `R(z)`，也不应被替换成简单二次近似：

```text
Phi_quad ~= k * (x^2 + y^2) / (2R)
```

PROPER 的 `wfarr` 则应理解为 PROPER 内部传播使用的参考相位去除后的复场。报告中若使用：

```text
U_PROPER = wfarr * exp(i * Phi_PROPER)
```

就必须说明 `Phi_PROPER` 是 PROPER pilot/qphase 参考相位，而不是上面的 `Phi_ZBF` 精确球面参考。把 `Phi_PROPER` 和 `Phi_ZBF` 混用，会直接制造混合参考残差。

## 证据分层

当前证据应分成三层，不应混写。

| 层级 | 问题 | 当前 S7/S8 证据 | 审计判断 |
|---|---|---:|---|
| 残余复场传播闭合 | `u_S7` 传播到 S8 后是否接近 ZBF 的 `u_S8` | S7 `2.28346e-17 waves`；S8 `0.000203888 waves` | 支持自由空间传播对慢变残余场是自洽的 |
| 物理总场重构闭合 | 用明确的端点 ZBF 参考相位把 `u` 重构为 `U` 后，是否接近 Zemax 端点物理总场重构 | S7 `2.3866e-17 waves`；S8 `0.000203888 waves` | 支持“在当前 ZBF 参考相位公式正确”的条件下，S7/S8 物理总场重构局部闭合 |
| 全局物理正确性 | 同一规则是否通过 VM1、S8 到 S12、下游强度 | 当前 S7/S8 证据不覆盖 | 未证明 |

强度证据也支持局部一致性：S7/S8 最大强度 RMS 为 `0.000741144 %`。这说明该局部比较不是只有相位通过、强度完全不对的数值假象。

但是这个强度结果仍然只覆盖 S7/S8 段。它不能证明元素面传递、VM1 后续传播或所有端点的物理总相位都正确。

## 对 `0.0319913 waves` 的解释

`0.0319913 waves` 应被命名为：

```text
mixed-reference S8 residual
```

更直白地说，它是“从内部残余场重构物理总场时，使用了与比较对象不同的参考相位公式”造成的残差。它的主要成分可以理解为：

```text
Delta Phi_ref = Phi_PROPER - Phi_ZBF
```

所以它不是一个可直接归因于自由空间传播核错误的物理残差。当前审计支持如下解释：

- 若两边都在 ZBF reference-relative 表示下比较，S8 相位 RMS 为 `0.000203888 waves`；
- 若用 ZBF 端点参考相位重构物理总场，S8 相位 RMS 仍为 `0.000203888 waves`；
- 若把 PROPER/qphase 参考相位重构结果拿去和 ZBF 精确端点物理场比较，会得到 `0.0319913 waves` 量级的残差。

因此，`0.0319913 waves` 是参考相位约定不一致的诊断信号，而不是自由空间传播失败的直接证据。

## 怀疑性审查

### 1. 这是不是“虚假的数值凑对”？

不能简单说是。原因是：残余复场传播闭合、端点 ZBF 参考相位重构后的物理总场闭合、强度也闭合，这三者同时成立，说明 S7/S8 局部段并非只靠任意相位平移把 RMS 做小。

但也不能说已经证明“物理真实相位完全正确”。原因是：Zemax ZBF 文件并没有直接以 POP/PROPER 约定给出 `U`，当前所谓物理总相位是由 `conj(Ex)` 和 `Phi_ZBF` 重构出来的。如果 `Phi_ZBF` 的解析表达式或符号选择本身错了，那么小 RMS 可能只是“用同一个错误公式定义并比较物理相位”的自洽结果。

因此，更严格的说法是：S7/S8 当前通过了“ZBF 表示内部自洽 + 残余场传播 + 端点重构”的局部检查；它还没有通过“独立物理全相位真值”的检查。

### 2. 当前证据排除了什么？

支持排除 S7/S8 局部段中的若干主导错误：

- 不是明显的采样间隔错误：最大采样间隔差异约 `0.00398782 %`；
- 不是主导的 pilot waist/Rayleigh 错误：最大 waist 差异约 `0.00309296 %`，最大 Rayleigh 差异约 `0.00831229 %`；
- 不是简单 q 增量错误：最大 q 增量差异约 `0.000910865 mm`，远小于拟合出的 `0.444739 mm` 等效离焦相位生成量；
- 不是明显旋转/倾斜主导项：`B/sqrt(A*C) ~= 2.0e-16`，倾斜移除量远小于二次项移除量。

这些排除项支持“`0.0319913 waves` 主要来自参考相位解释方式”，但它们仍然不是全系统物理正确性的证明。

### 3. 当前证据没有排除什么？

没有排除以下可能：

- ZBF 端点参考相位规则只在当前 S7/S8 几何中局部有效；
- 当前 `Phi_ZBF` 公式与 Zemax 真正用于物理全相位显示/导出的参考公式存在高阶差异；
- 后续元素面存在新的参考相位转换项，尤其是 VM1 / S8 到 S12；
- 某些端点的物理总场重构需要额外几何项，而不是简单复用 S7/S8 的端点规则；
- 全路径报告中某些低 RMS 可能仍然来自选择了有利的参考相位，而不是同一个物理场在所有边界都一致。

因此，审计结论必须保守。

## 规范改写文本

下面这段可用于替换或前置到 `s7_s8_phase_root_cause.md` 中的 S7/S8 当前结论：

```markdown
### Audited S7/S8 interpretation

The final quantity of interest is the physical total field, not the reference
phase itself. However, neither raw Zemax ZBF `Ex` nor PROPER `wfarr` is the
physical total field in POP/PROPER convention; each must first be lifted from a
reference-relative representation. In POP/PROPER phasor convention, the ZBF
residual field is `conj(Ex)`, and the ZBF endpoint physical field is
reconstructed as `conj(Ex) * exp(i * axis_sign * Phi_ZBF)`, where the validated
equal-radius ZBF reference phase is
`Phi_ZBF = k * sign(R) * (sqrt(abs(R)^2 + x^2 + y^2) - abs(R))`.

Under the condition that this ZBF endpoint reconstruction formula is physically
correct for S7 and S8, the S7-to-S8 free-space segment is locally consistent:
the propagated residual field can be lifted back to a physical total field that
matches the corresponding Zemax endpoint reconstruction.

Under this explicitly stated convention, the S7 reconstructed physical RMS is
`2.3866e-17 waves`, the S8 reconstructed physical RMS is `0.000203888 waves`,
and the maximum S7/S8 intensity RMS is `0.000741144 %`.

The larger S8 value `0.0319913 waves` is retained as a mixed-reference residual:
it is obtained when a PROPER/qphase-lifted field is compared against a
ZBF-exact endpoint reconstruction. It is therefore diagnostic evidence of a
reference-phase convention mismatch, not direct evidence of a free-space
propagation-kernel error.

This local S7/S8 result does not independently prove absolute physical total
phase correctness, because the endpoint physical phase is still reconstructed
through the accepted `Phi_ZBF` formula. It also does not prove VM1, S8-to-S12,
or whole-path physical phase closure. Later-surface conclusions require separate
tests showing that the same physical field, residual field, and reference phase
conversion remain consistent without diagnostic splicing or fitted scale
factors.
```

## 后续诊断不得越界的规则

1. 报告任何小 RMS 前，必须说明比较的是 `U` 还是 `u`。
2. 报告任何物理总相位前，必须写出使用的 `Phi_ref`。
3. ZBF 端点重构必须明确写出 `axis_sign` 和 `Phi_ZBF`，不得把它简写成“physical phase”。
4. PROPER/qphase 参考相位不得与 ZBF 精确球面参考相位混称为同一个物理相位。
5. `0.0319913 waves` 应保留为 mixed-reference residual，不应作为传播核错误或传播已修复的单独判据。
6. S7/S8 局部闭合不得外推为 VM1、S8 到 S12 或全路径闭合。
7. 任何修正如果只改善端点物理相位 RMS，却破坏残余复场传播或下游强度，都应视为数值补偿，而不是物理修正。
8. 任何 fitted scale、q shift、path shift 或 endpoint-only patch，若没有共同几何方程推导，都不能进入核心代码。

## 最终审计判断

S7/S8 的物理全相位并不是在“独立于重构公式”的意义上被证明完全正确。更准确的判断是：

> 当前证据支持：若 `Phi_ZBF` 是 S7/S8 端点物理总场的正确重构参考相位，则残余场传播后重构出的物理总场与 Zemax 端点重构局部一致。当前证据不支持宣称绝对物理总相位已经被独立证明正确。

这不是简单的虚假凑对，因为残余场传播、物理总场重构和强度同时闭合。但它也不是最终物理证明，因为它仍依赖 `Phi_ZBF` 的解析表达式和符号约定，并且该重构规则尚未通过 VM1 / S8 到 S12 及全路径验证。
