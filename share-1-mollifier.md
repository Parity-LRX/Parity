#### **目前力约束分子势能模型所用的损失函数 \(L\) 大多数为以下形式：**
\[
L = \frac{\lambda_E}{B} \sum_{b=1}^{B} \left( \hat{E}_b - E_b \right)^2 + \frac{\lambda_F}{3BN} \sum_{i=1}^{B \cdot N} \sum_{\alpha=1}^{3} \left( -\frac{\partial \hat{E}}{\partial r_{i,\alpha}} - F_{i,\alpha} \right)^2
\]
- \(B\)：batch_size大小。
- \(\alpha\)：遍历三维坐标轴。
  
在具体实例中，\(\lambda_E\) 和 \(\lambda_F\) 一般取值是1和1000，因为力的 *\(RMSE_{-Force}\)* 一般都很小, 所以给 \(\lambda_F\) 赋一个较大的值才能获得比较好的约束效果。

但是，这种选择依赖于经验，且不同的体系可能需要不同的超参数设置。如果超参数选择不当，可能会导致模型训练不稳定或收敛困难。
不仅如此，在我的模型测试中发现仅仅对力进行约束，在 *\(RMSE_{-Energy}\)* 到达当前超参数设置下较小的数值，但 *\(RMSE_{-Force}\)* 会一直震荡，继续训练会特别容易会导致过拟合，最终 *\(RMSE_{-Energy}\)*  也无法收敛。

导致这个问题的一部分原因，我认为模型构造的势能面还不够光滑。并且，仅依赖力的损失不足以构造理想光滑的势能面。

#### **然后想到了Mollifier函数：**
\[
\phi_\epsilon(\mathbf{x}) = 
\begin{cases} 
C_\epsilon \exp\left(-\frac{1}{1 - \|\mathbf{x}\|^2 / \epsilon^2}\right) & \text{if } \|\mathbf{x}\| < \epsilon \\
0 & \text{otherwise}
\end{cases}
\]
- \(C_\epsilon\)：归一化系数。
- \(\|\mathbf{x}\|\)：向量 \(\mathbf{x}\) 的范数。
- \(\epsilon\)：平滑参数，控制函数的支撑范围。

不过我粗略地测试了一下，**高斯型Mollifier**会更为稳健一点：
\[
\phi(\mathbf{x}) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{\|\mathbf{x}\|^2}{2\sigma^2}\right)
\]
- \(\sigma\)：平滑参数，控制函数的支撑范围。

接下来，我们利用Mollifier函数对拟合的势能面的一阶导数进行处理，并为**正则化项**：
\[
\text{Reg-M} = \frac{\lambda_M}{B} \sum_{i=1}^{B} \sum_{\alpha=1}^{3} \int \left\| \frac{\partial \hat{E}}{\partial r_{i,\alpha}} \right\|^2 \phi(r_{i,\alpha}) \, dr_{i,\alpha}
\]

最后将这个正则化项加到原始的损失函数 \(L\) 之中:
\[
L = \frac{\lambda_M}{B} \sum_{i=1}^{B} \sum_{\alpha=1}^{3} \int \left\| \frac{\partial \hat{E}}{\partial r_{i,\alpha}} \right\|^2 \phi(r_{i,\alpha}) \, dr_{i,\alpha} + \frac{\lambda_E}{B} \sum_{b=1}^{B} \left( \hat{E}_b - E_b \right)^2 + \frac{\lambda_F}{3BN} \sum_{i=1}^{B \cdot N} \sum_{\alpha=1}^{3} \left( -\frac{\partial \hat{E}}{\partial r_{i,\alpha}} - F_{i,\alpha} \right)^2
\]

实际使用之中，对原始数据合理放缩之后，\(\sigma\) 建议设置为1， \(\lambda_M\)可以尝试先设置为10，此时 \(\lambda_F\) 建议设置为与 \(\lambda_M\) 相同或更小的数量级，比如5 。

#### **在 \(L\) 中加上 \(\text{Reg-M}\) 有什么好处？**

- Mollifier 函数通过约束势能面的一阶导数，强制势能面变得更加平滑。这可以有效减少势能面的异常值震荡,，提高训练的稳定性。
- 引入 \(\text{Reg-M}\)后，势能模型不仅关注力的准确性，还关注势能面的整体平滑性，避免过度拟合局部特征。
- 平滑的势能面意味着力的变化更加连续，从而减少力的预测误差。
- 通过调整 \(\lambda_M\)来控制势能面的平滑性，减少对 \(\lambda_F\)的依赖。
  