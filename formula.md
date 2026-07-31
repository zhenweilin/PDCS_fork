# Reflection 主序列与 Halpern 候选点

## 1. 符号定义

设：

- \(t\) 表示当前 restart 周期；
- \(k\) 表示该周期内的 inner iteration；
- \(z=(x,y)\) 同时表示 primal variable \(x\) 和 dual variable \(y\)；
- \(z_{t,k}\) 表示当前主迭代点；
- \(\widehat z_{t,k+1}\) 表示从 \(z_{t,k}\) 出发完成一次基础 PDHG 更新后得到的点；
- \(z_{t,0}\) 表示当前 restart 周期内固定不变的 restart anchor；
- \(\bar z_{t,k}\) 表示主迭代序列的加权平均点。

新的设计中，主迭代序列只使用 reflection。Halpern 更新只产生一个额外的
restart candidate，不会写回主迭代序列。

## 2. 基础 PDHG 更新

首先，从当前主迭代点执行一次基础 PDHG 更新：

\[
\widehat z_{t,k+1}
=
\operatorname{PDHG}(z_{t,k}).
\]

## 3. Reflection 主迭代

对基础 PDHG 更新结果执行 reflection：

\[
\boxed{
z_{t,k+1}
=
z^{\mathrm{ref}}_{t,k+1}
=
(1+\beta_t)\widehat z_{t,k+1}
-\beta_t z_{t,k}
}
\]

等价地，

\[
z_{t,k+1}
=
\widehat z_{t,k+1}
+
\beta_t\left(\widehat z_{t,k+1}-z_{t,k}\right).
\]

当前实现使用的 reflection 系数为

\[
\beta_t
=
-0.1\log_{10}\!\left(\operatorname{KKT}^{\max}_t\right)+0.2.
\]

这里的 \(z_{t,k+1}\) 是真正的主迭代点。下一次基础 PDHG 更新从
\(z_{t,k+1}\) 出发，而不是从 Halpern candidate 出发。

如果关闭 reflection，则

\[
z_{t,k+1}=\widehat z_{t,k+1}.
\]

## 4. 主序列的加权平均点

主序列的加权平均点按照当前步长 \(\eta_{t,k+1}\) 更新：

\[
\boxed{
\bar z_{t,k+1}
=
\frac{
    \eta^{\mathrm{cum}}_{t,k}\bar z_{t,k}
    +
    \eta_{t,k+1}z_{t,k+1}
}{
    \eta^{\mathrm{cum}}_{t,k}
    +
    \eta_{t,k+1}
}
}
\]

其中

\[
\eta^{\mathrm{cum}}_{t,k+1}
=
\eta^{\mathrm{cum}}_{t,k}
+
\eta_{t,k+1}.
\]

这个平均只包含 reflection 主序列中的点，不包含 Halpern candidate。

## 5. Halpern restart candidate

在不改变主迭代点的情况下，额外计算 Halpern candidate：

\[
\boxed{
z^{\mathrm{hal}}_{t,k+1}
=
\alpha_k z_{t,k+1}
+
(1-\alpha_k)z_{t,0}
}
\]

其中

\[
\alpha_k=\frac{k+1}{k+2},
\qquad
1-\alpha_k=\frac{1}{k+2}.
\]

代入 reflection 主迭代公式，可得

\[
\boxed{
z^{\mathrm{hal}}_{t,k+1}
=
\frac{k+1}{k+2}
\left[
(1+\beta_t)\widehat z_{t,k+1}
-\beta_t z_{t,k}
\right]
+
\frac{1}{k+2}z_{t,0}
}.
\]

Halpern candidate 只用于 restart candidate 的比较：

- 不写回 \(z_{t,k+1}\)；
- 不作为下一次 PDHG 更新的输入；
- 不加入主序列的加权平均点 \(\bar z_{t,k+1}\)；
- 在未发生 restart 时，不改变求解器的主迭代轨迹。

因此，严格地说，它是一个辅助的 Halpern-mixed candidate，而不是一条独立的
Halpern 迭代轨迹。

## 6. Restart 时的三个候选点

每次执行 restart 检查时，比较以下三个候选点：

\[
\boxed{
\mathcal C_{t,k+1}
=
\left\{
z_{t,k+1},
\bar z_{t,k+1},
z^{\mathrm{hal}}_{t,k+1}
\right\}.
}
\]

分别对应：

1. 当前 reflection 主迭代点 \(z_{t,k+1}\)；
2. reflection 主序列的加权平均点 \(\bar z_{t,k+1}\)；
3. Halpern candidate \(z^{\mathrm{hal}}_{t,k+1}\)。

使用相同的 KKT error、normalized duality gap 或现有 restart merit function
评价三个候选点。若 merit function 记为 \(\mathcal M(z)\)，则选择

\[
\boxed{
z^{\mathrm{restart}}_{t+1,0}
=
\underset{z\in\mathcal C_{t,k+1}}{\operatorname{argmin}}
\ \mathcal M(z).
}
\]

当 restart 条件满足时：

\[
z_{t+1,0}
=
z^{\mathrm{restart}}_{t+1,0},
\]

并将该点同时设置为：

- 下一个 restart 周期的 anchor；
- 下一个 restart 周期的主迭代初始点；
- 新的加权平均初始点。

随后重置 inner iteration 计数和累计步长：

\[
k\leftarrow 0,
\qquad
\eta^{\mathrm{cum}}\leftarrow 0.
\]

## 7. 三个候选点之间的区别

| 候选点 | 使用 reflection | 使用 restart anchor | 反馈到下一次主迭代 | 用于 restart 选择 |
|---|---:|---:|---:|---:|
| 当前点 \(z_{t,k+1}\) | 是 | 否 | 是 | 是 |
| 平均点 \(\bar z_{t,k+1}\) | 是 | 否 | 仅在被 restart 选中后 | 是 |
| Halpern candidate \(z^{\mathrm{hal}}_{t,k+1}\) | 是 | 是 | 仅在被 restart 选中后 | 是 |

## 8. GPU 额外内存

若永久保存 Halpern primal 和 dual candidate，需要增加两个 `Float64` GPU
向量：

\[
x^{\mathrm{hal}}\in\mathbb R^n,
\qquad
y^{\mathrm{hal}}\in\mathbb R^m.
\]

额外显存约为

\[
\boxed{
8(n+m)\ \text{bytes}.
}
\]

如果只在 restart 检查时计算 Halpern candidate，并安全复用已有 scratch
buffer，则可能不需要永久增加这两个向量。但复用时必须保证不会覆盖当前点、
平均点、restart anchor 或 termination/restart 检查仍需要的数据。

## 9. 新设计的核心原则

\[
\boxed{
\text{PDHG}
\longrightarrow
\text{Reflection 主迭代}
\longrightarrow
\begin{cases}
\text{更新主序列加权平均点},\\
\text{计算辅助 Halpern candidate}
\end{cases}
\longrightarrow
\text{从三个候选点中选择 restart 点}.
}
\]

这个设计保留 reflection 主序列的速度，同时只在 Halpern candidate 的
restart merit 更好时利用 Halpern 信息，避免 Halpern anchor mixing 在每次
inner iteration 中持续抑制主迭代的前进。
