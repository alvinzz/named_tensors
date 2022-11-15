from __future__ import annotations

from typing import List, Union, Optional
from collections.abc import Callable

from named_shape import NamedShape

import torch

from opt_einsum import contract


class NamedTensor:
    """
    Contains a named tensor.

    Pseudo dimensions may be used temporarily for broadcasting. They have a size
    of 1 in the underlying data representation but should be treated as if they
    have undefined size.
    """

    def __init__(
        self,
        data: torch.Tensor,
        shape: Optional[NamedShape] = None,
        dims: Optional[List[str]] = None,
        pseudos: Optional[List[bool]] = None,
        name: Optional[str] = ""
    ):
        """Creates a NamedTensor.

        Args:
            data (torch.Tensor): data for NamedTensor.
            shape (Optional[NamedShape]): NamedShape of NamedTensor. \
                This input is always used if it is provided.
            dims (Optional[List[str]]): Dims of NamedTensor. Used if {shape} \
                is not provided.
            pseudos (Optional[List[bool]]): If dims are pseudo- or not. Used if \
                {shape} is not provided. Defaults to None, or False for all dims
            name (Optional[str]): ID for NamedTensor. Defaults to "".

        Raises:
            ValueError: If neither {shape} nor {dims} is provided.
            ValueError: If {data.shape} and {self.shape} are not compatible.
        """
        self.name = name

        if shape is not None:
            self.data = data
            self.shape = shape
        elif dims is not None:
            self.data = data
            self.shape = NamedShape(dims, list(self.data.shape), pseudos)
        else:
            raise ValueError("neither shape nor dims specified")
        if list(self.data.shape) != self.shape.sizes:
            raise ValueError("data.shape and self.shape not equal")

    @property
    def tensor(self) -> torch.Tensor:
        """Underlying torch.Tensor data.

        Returns:
            torch.Tensor: Underlying torch.Tensor.
        """
        return self.data

    def __getattr__(self, attr):
        # If attr is not in {self}, tries to get it from {self.shape}, then
        # {self.data}.
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            try:
                return getattr(self.shape, attr)
            except AttributeError:
                try:
                    return getattr(self.data, attr)
                except AttributeError:
                    raise AttributeError(f"could not parse attr {attr}")

    def clone(self) -> NamedTensor:
        """Creates a copy of {self}. Does not clone the underlying tensor data.

        Returns:
            NamedTensor: Copy of {self}.
        """
        return NamedTensor(self.data, self.shape.clone())

    def rename(self, old: str, new: str) -> NamedTensor:
        """Rename a dimension.

        Args:
            old (str): Dimension to rename,
            new (str): New name for {old}.

        Raises:
            ValueError: If {new} is already in dims.

        Returns:
            NamedTensor: Copy of {self} with dimension {old} renamed to {new}.
        """
        return NamedTensor(self.data, self.shape.rename(old, new))

    def reduce(
        self,
        reduce_fn: Callable[[torch.Tensor, List[int]], torch.Tensor],
        reduce_dims: Union[str, List[str]]
    ) -> NamedTensor:
        """Reduce (remove) target dimensions via {reduce_fn}.

        Args:
            reduce_fn (Callable[[torch.Tensor, List[int]], torch.Tensor]): \
                Function which reduces {self.tensor} over a list of dimension \
                indices.
            reduce_dims (Union[str, List[str]]): Target dimensions.

        Returns:
            NameTensor: Copy of {self} with target dimensions reduced.
        """
        if isinstance(reduce_dims, str):
            reduce_dims = [reduce_dims]
        reduce_idxs = [self.idx(dim) for dim in reduce_dims]
        res_data = reduce_fn(self.data, reduce_idxs)
        return NamedTensor(res_data, self.shape.reduce(reduce_dims))

    def squeeze(
        self,
        dims: Optional[Union[str, List[str]]] = None
    ) -> NamedTensor:
        """Squeeze (remove) pseudo-dimensions.

        Args:
            dims (Optional[Union[str, List[str]]]): \
                Dimensions to squeeze. Defaults to None, in which case all \
                pseudo-dimensions will be squeezed.

        Raises:
            ValueError: If dims contains a true dimension.

        Returns:
            NamedTensor: Copy of {self} with pseudo-dims removed.
        """
        res_shape = self.shape.squeeze(dims)
        return NamedTensor(self.data.view(res_shape.sizes), res_shape)

    def squeeze_as(self, other: NamedShape) -> NamedTensor:
        """Squeeze dimensions of {self} not in {other}.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedTensor: Copy of {self} with no dimensions not in {other}.
        """
        res = self.clone()
        for dim in self.dims:
            if dim not in other.dims:
                res = res.squeeze(dim)
        return res

    def unsqueeze(
        self,
        dims: Union[str, List[str]],
        idxs: Optional[Union[int, List[int]]] = None
    ) -> NamedTensor:
        """Unsqueeze (add) pseudo-dimensions.

        Args:
            dims (Union[str, List[str]]): Pseudo-dimensions to add.
            idx (Optional[Union[int, List[int]]]): Indices for pseudo-\
                dimensions. Defaults to None, which puts dims at the end.

        Returns:
            NamedTensor: Copy of {self} with pseudo-dimensions added.
        """
        if isinstance(dims, str):
            if idxs is None:
                idxs = -1
            return NamedTensor(self.data.unsqueeze(idxs),
                               self.shape.unsqueeze(dims, idxs))
        else:
            if idxs is None:
                idxs = [-1] * len(dims)
            res = self.clone()
            for (dim, idx) in zip(dims, idxs):
                res = res.unsqueeze(dim, idx)
            return res

    def unsqueeze_as(self, other: NamedShape) -> NamedTensor:
        """Unsqueeze dimensions of {other} not in {self}.

        New dimensions are placed at the end.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedTensor: Copy of {self} with all the dimensions of {other}.
        """
        res = self.clone()
        for dim in other.dims:
            if dim not in self.dims:
                res = res.unsqueeze(dim)
        return res

    def expand(
        self,
        dims: Union[str, List[str]],
        sizes: Union[int, List[int]]
    ) -> NamedTensor:
        """Expand target dimensions to target sizes.

        If target dimensions are pseudo-dimensions, this converts them to true \
            dimensions.

        Args:
            dims (str): Target dimensions.
            sizes (int): Target sizes.

        Returns:
            NamedTensor: Copy of {self} with dimensions expanded to sizes.
        """
        res_shape = self.shape.expand(dims, sizes)
        res_data = self.data.expand(res_shape.sizes)
        return NamedTensor(res_data, res_shape)

    def expand_as(self, other: NamedShape) -> NamedTensor:
        """Expand dimensions in {self} to match their sizes in {other}.

        A dimension which is pseudo- in {self} and true in {other} will be \
            converted to a true dimension.

        A dimension which is true in {self} and pseudo- in {other} will \
            stay a true dimension with its current size.

        Args:
            other (NamedShape): Target shape.

        Raises:
            ValueError: If dims of {self} and {other} are not the same.

        Returns:
            NamedTensor: Copy of {self} with dimensions expanded to \
                sizes in {other}.
        """
        res_shape = self.shape.expand_as(other)
        res_data = self.data.expand(res_shape.sizes)
        return NamedTensor(res_data, res_shape)

    def union(self, other: NamedTensor) -> NamedTensor:
        """Computes the union of two tensors.

        In the returned result, the dimensions of {self} come before \
            dimensions that are only in {other}. For shared dimensions, \
            the size is the maximum of their sizes in {self} and {other}.

        Args:
            other (NamedTensor): Target NamedTensor.

        Returns:
            NamedTensor: Union of {self} and {other}.
        """
        return (self
                .unsqueeze_as(other)
                .expand_as(
                    other
                    .unsqueeze_as(self)))

    def transpose(self, dim0: str, dim1: str) -> NamedTensor:
        """Swap the places of dim0 and dim1.

        Args:
            dim0 (str): Target dimension.
            dim1 (str): Target dimension.

        Returns:
            NamedTensor: Copy of {self} with dim0 and dim1 swapped.
        """
        return NamedTensor(self.data.transpose(dim0, dim1),
                           self.shape.transpose(dim0, dim1))

    def permute(self, dims: List[str]) -> NamedTensor:
        """Permutes the ordering of dims to the target ordering.

        Args:
            dims (List[str]): Target dimension ordering.

        Returns:
            NamedTensor: Copy of {self} with dimensions in target ordering.
        """
        idxs = [self.idx(dim) for dim in dims]
        return NamedTensor(self.data.permute(idxs), self.shape.permute(dims))

    def permute_as(self, other: NamedShape) -> NamedTensor:
        """Permute the ordering of dims to match the ordering in {other}.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedTensor: Copy of {self} with dimensions reordered \
                to match {other}.
        """
        return self.permute(other.dims)

    def with_idx(self, dim: str, idx: int) -> NamedTensor:
        """Move dimension to target index.

        Args:
            dim (str): Target dimension.
            idx (int): Target index.

        Returns:
            NamedTensor: Copy of {self} with dimension moved to target index.
        """
        res_shape = self.shape.with_idx(dim, idx)
        return self.permute(res_shape.dims)

    def with_first(self, dim: str) -> NamedTensor:
        """Move dimension to front.

        Args:
            dim (str): Target dimension.

        Returns:
            NamedTesnor: Copy of {self} with target dimension moved to front.
        """
        return self.with_idx(dim, 0)

    def with_last(self, dim: str) -> NamedTensor:
        """Move dimension to end.

        Args:
            dim (str): Target dimension.

        Returns:
            NamedShape: Copy of {self} with target dimension moved to the end.
        """
        return self.with_idx(dim, -1)

    def join_dims(
        self,
        dims: List[str],
        new_dim: str,
        idx: Optional[int] = None
    ) -> NamedTensor:
        """Join target dimensions into a new dimension.

        Args:
            dims (List[str]): Dimensions to join.
            new_dim (str): Name of new dimension.
            idx (Optional[int]): Index of new dimension. \
                Defaults to None, which places it at the index of \
                {dims[0]}.

        Returns:
            NamedTensor: Copy of {self} with {dims} joined into {new_dim}.
        """
        if idx is None:
            idx = self.idx(dims[0])

        res_shape = self.shape.join_dims(dims, new_dim, idx)
        if idx < 0:
            idx += (self.ndim + 1)

        # first permute dims into correct ordering
        tmp_dims = list(filter(lambda d: d not in dims, self.dims))
        new_size = 1
        dim_idx = idx
        for dim in dims:
            tmp_dims.insert(dim_idx, dim)
            dim_idx += 1
            new_size *= self.size(dim)
        res = self.permute(tmp_dims)

        return NamedTensor(res.data.reshape(res_shape.sizes), res_shape)

    def split_dim(
        self,
        dim: str,
        new_dims: List[str],
        new_sizes: List[int]
    ) -> NamedTensor:
        """Split target dimension into new dims.

        Args:
            dim (str): Dimension to split.
            new_dims (List[str]): Names of new dimensions.
            new_sizes (Optional[List[int]]): Sizes of new dimensions.\
                Defaults to None, which can only be used if the dimension being\
                split is pseudo-.

        Returns:
            NamedTensor: Copy of {self} with {dim} split into {new_dims}.
        """
        res_shape = self.shape.split_dim(dim, new_dims, new_sizes)
        return NamedTensor(self.data.view(res_shape.sizes), res_shape)

    def apply_elemwise(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> NamedTensor:
        """Apply element-wise function to underlying data.

        Does not change shape.

        Args:
            fn (Callable[[torch.Tensor], torch.Tensor]): Element-wise function.

        Raises:
            ValueError: If {fn} changes the shape of {self.data}.

        Returns:
            NamedTensor: Copy with {self} with {data} = fn{self.data}.
        """
        res = fn(self.data)
        if not res.data.shape == self.data.shape:
            raise ValueError("elemwise fn changed size of tensor")
        return NamedTensor(fn(self.data), self.shape)

    def apply_binary(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        other: NamedTensor
    ) -> NamedTensor:
        """Apply element-wise binary function to {self} and {other}.

        Uses union() to compute the output shape.

        Args:
            fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): \
                Element-wise binary function.
            other (NamedTensor): Target NamedTensor.

        Returns:
            NamedTensor: NamedTensor representing fn({self}, {other}).
        """
        union_shape = self.shape.union(other.shape)
        left = self.unsqueeze_as(other).permute_as(union_shape)
        right = other.unsqueeze_as(self).permute_as(union_shape)
        return NamedTensor(fn(left.data, right.data), union_shape)

    def apply_multi(
        fn: Callable[[List[torch.Tensor]], torch.Tensor],
        tensors: NamedTensor
    ) -> NamedTensor:
        """Apply element-wise n-ary function to list of tensors.

        The output shape is equivalent to:
            tensors[0].union(tensors[1]).union(tensors[2])....

        Args:
            fn (Callable[[List[torch.Tensor]], torch.Tensor]): \
                Element-wise n-ary function.
            tensors (NamedTensor): Target NamedTensors.

        Returns:
            NamedTensor: NamedTensor representing fn({tensors}).
        """
        targets = [tensors[-1]]
        for t in reversed(tensors[:-1]):
            targets.insert(0, t.unsqueeze_to(targets[0]))

        for idx in range(1, len(targets)):
            targets[idx] = targets[idx].unsqueeze_to(targets[idx - 1])
            targets[idx] = targets[idx].permute_to(targets[idx - 1])
            targets[idx] = targets[idx].expand_to(targets[idx - 1])

        res_data = fn([t.data for t in targets])
        return NamedTensor(res_data, targets[-1].shape)

    def einsum(
        self,
        other: NamedTensor,
        reduce_dims: Union[str, List[str]]
    ) -> NamedTensor:
        """Einstein summation (generalized batch matrix-multiplication).

        Multiplies {self} and {other} element-wise, \
            then sums over {reduce_dims}.

        Args:
            other (NamedTensor): Right tensor.
            reduce_dims (Union[str, List[str]]): Dimensions to sum over.

        Raises:
            ValueError: If {self.dims} or {other.dims} do not contain all \
                {reduce_dims}.

        Returns:
            NamedTensor: Result of einsum. \
                Has shape {shared} + {left_only} + {right_only}, \
                where {shared} contains non-reduced shared dims between \
                {self} and {other}, \
                {left_only} contains dims in {self} but not {other}, \
                and {right_only} contains dims in {other} but not {self}. \
                Within {shared} and {left_only}, dims are arranged by order \
                of appearance in {self}. \
                Within {right_only}, dims are arranged by order of \
                appearance in {other}.

        ---Usage---

        For example, we have:
            poses  : [B, T, DST_XYZ, SRC_XYZ],
            pts_src: [B, T, SRC_XYZ, R, C].

        Instead of:
            pts_dst = (poses.reshape(B*T, DST_XYZ, SRC_XYZ).bmm(
                           pts_src.reshape(B*T, SRC_XYZ, R*C)
                       ).reshape(B, T, DST_XYZ, R, C)),
        do:
            pts_dst = poses.einsum(pts_src, [SRC_XYZ]).

        ---Details---

        Einsum is a generalized way to expand, multiply, and sum over \
            dimensions. For example, a 2D-convolution needs to perform 3 \
            summations to compute one element (over the (X_IN, K_R, K_C) \
            dimensions). Rather than writing custom functions for each of \
            these summations, we can instead think of these operations as a \
            kind of generalized batched matrix-multiplication.

        Batched matrix multiplication (bmm) takes a tensor of shape (B, N, M) \
            and a tensor of shape (B, M, P) and outputs a tensor of shape \
            (B, N, P). It is equivalent to expanding the left and right input \
            tensors, performing an element-wise multiplication, and summing \
            over the 'M' dimension:
            |______(B, N, M) -> (B, N, M, 1) \n
            |______(B, M, P) -> (B, 1, M, P) \n
            |_x_____________________________ \n
            |________________(B, N, M, P)
            And reduce_sum over (M) to get result (B, N, P).

        Einstein's insight was that a lot of tensor operations have the \
            same basic form as {bmm}, in that they are element-wise \
            multiplications of expanded tensors, followed by a sum-reduction \
            over certain dimensions.

        So for conv2d, we can replicate the performance by gathering chunks \
            from the input image (for stride=1 and padding=same, \
            we will end up with [B, R, C] chunks of size [X_IN, K_R, K_C]). \
            Then we einsum with the [X_OUT, X_IN, K_R, K_C] kernel and \
            reduce over the [X_IN, K_R, K_C] dims to get the result. \
            This is verified in test_einsum() in \
            mt/nn/tests/test_named_tensor.py.
        """
        if isinstance(reduce_dims, str):
            reduce_dims = [reduce_dims]

        for reduce_dim in reduce_dims:
            if (reduce_dim not in self.dims) or (reduce_dim not in other.dims):
                raise ValueError(
                    "all reduce_dims must be in both self and other")

        shared_dims = []
        left_only_dims = []
        right_only_dims = []

        for self_dim in self.dims:
            if self_dim not in reduce_dims:
                if self_dim in other.dims:
                    shared_dims.append(self_dim)
                else:
                    left_only_dims.append(self_dim)

        for other_dim in other.dims:
            if other_dim not in reduce_dims:
                if other_dim not in self.dims:
                    right_only_dims.append(other_dim)

        left_dims = shared_dims + left_only_dims + reduce_dims
        right_dims = shared_dims + reduce_dims + right_only_dims

        left_shape = self.shape.permute(left_dims)
        right_shape = other.shape.permute(right_dims)
        total_shape = left_shape.union(right_shape)
        res_shape = total_shape.reduce(reduce_dims)

        left_shape = total_shape.reduce(right_only_dims)
        right_shape = total_shape.reduce(left_only_dims)
        left = self.permute(left_dims).expand_as(left_shape)
        right = other.permute(right_dims).expand_as(right_shape)
        left_einsum_sig = ''.join(
            [chr(ord('a') + total_shape.idx(dim)) for dim in left.dims])
        right_einsum_sig = ''.join(
            [chr(ord('a') + total_shape.idx(dim)) for dim in right.dims])
        res_einsum_sig = ''.join(
            [chr(ord('a') + total_shape.idx(dim)) for dim in res_shape.dims])
        res_data = contract(
            f"{left_einsum_sig},{right_einsum_sig}->{res_einsum_sig}",
            left.data,
            right.data
        )

        return NamedTensor(res_data, res_shape)

    def __repr__(self):
        return f"{repr(self.shape)}\n{repr(self.tensor)}"

    def __invert__(self):
        return self.apply_elemwise(lambda x: ~x)

    def __and__(self, other):
        return self.apply_binary(lambda x, y: x & y, other)

    def __or__(self, other):
        return self.apply_binary(lambda x, y: x | y, other)

    def __eq__(self, other):
        return self.apply_binary(lambda x, y: x == y, other)

    def __lt__(self, other):
        return self.apply_binary(lambda x, y: x < y, other)

    def __le__(self, other):
        return self.apply_binary(lambda x, y: x <= y, other)

    def __gt__(self, other):
        return self.apply_binary(lambda x, y: x > y, other)

    def __ge__(self, other):
        return self.apply_binary(lambda x, y: x >= y, other)

    def __neg__(self):
        return self.apply_elemwise(lambda x: -x)

    def __abs__(self):
        return self.apply_elemwise(lambda x: abs(x))

    def __add__(self, other):
        return self.apply_binary(lambda x, y: x + y, other)

    def __sub__(self, other):
        return self.apply_binary(lambda x, y: x - y, other)

    def __mul__(self, other):
        return self.apply_binary(lambda x, y: x * y, other)

    def __truediv__(self, other):
        return self.apply_binary(lambda x, y: x / y, other)

    def __pow__(self, other):
        return self.apply_binary(lambda x, y: x ** y, other)
