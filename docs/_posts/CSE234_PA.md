---
# 文章标题
title: CSE234
# 设置写作时间
date: 2025-02-27
# 一个页面可以有多个分类
category:
  - CSE234
  - Lab
# 一个页面可以有多个标签
tag:
  - 公开课
  - MLSys
  - course
  - Lab
# 此页面会在文章列表置顶
sticky: true
# 此页面会出现在文章收藏中
star: true
# 侧边栏的顺序
# 数字越小越靠前，支持非整数和负数，比如 -10 < -9.5 < 3.2, order 为 -10 的文章会最靠上。
# 个人偏好将非干货或随想短文的 order 设置在 -0.01 到 -0.99，将干货类长文的 order 设置在 -1 到负无穷。每次新增文章都会在上一篇的基础上递减 order 值。
order: -2.1
---

## PA1: Automatic differentiation

### Question 1: Auto Diff Library

#### Part 1: Operators

The list of operators that you will need to implement are:

* `DivOp`
* `DivByConstOp`
* `TransposeOp`
* `ReLUOp`
* `SqrtOp`
* `PowerOp`
* `MeanOp`
* `MatMulOp`
* `SoftmaxOp`
* `LayerNormOp`

##### DivOp

```Python
class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        """TODO: your code here"""
    

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        """TODO: your code here"""
```

