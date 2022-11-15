from __future__ import annotations

from typing import List, Union, Optional

import numpy as np


class NamedShape:
    """
    Contains a tensor shape with named dimensions.

    Pseudo dimensions may be used temporarily for broadcasting. They have a size
    of 1 in the underlying data representation but should be treated as if they
    have undefined size.
    """

    def __init__(
        self,
        dims: List[str],
        sizes: List[int],
        pseudos: Optional[List[bool]] = None,
        name: Optional[str] = ""
    ):
        """Creates a NamedShape.

        Args:
            dims (List[str]): Dimension names.
            sizes (List[int]): Dimension sizes.
            pseudos (Optional[List[bool]]): If dimensions are pseudo. \
                Defaults to None, or False for all dims.
            name (Optional[str]): ID for NamedShape. Defaults to "".

        Raises:
            ValueError: If there are duplicate dims.
            ValueError: If dims and sizes do not have the same length.
            ValueError: If dims and provided pseudos do not have the same length
            ValueError: If pseudo dims have size != 1.
        """
        self.name = name

        if len(set(dims)) != len(dims):
            raise ValueError("dims cannot contain duplicates")
        self.dims = dims

        if self.ndim != len(sizes):
            raise ValueError("dims and sizes not same length")
        self.sizes = sizes
        # if any([size == 1 for size in self.sizes]):
        #     print("warning: sizes includes a dimension with size one. assuming "
        #           "that this is the true size of the dimension. specify "
        #           "pseudos explicitly to mark it as a pseudo dimension.")

        if pseudos is None:
            self.pseudos = [False] * self.ndim
        else:
            if self.ndim != len(pseudos):
                raise ValueError("dims and pseudos not same length")
            for (pseudo, size) in zip(pseudos, self.sizes):
                if pseudo and size > 1:
                    raise ValueError("size of pseudo dims must be 1")
            self.pseudos = pseudos

    @property
    def ndim(self) -> int:
        """Number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        return len(self.dims)

    def clone(self) -> NamedShape:
        """Creates a copy of {self}.

        Returns:
            NamedShape: Copy of {self}.
        """
        return NamedShape(self.dims[:], self.sizes[:], self.pseudos[:])

    def idx(self, dim: str) -> int:
        """Gets index of target dimension.

        Args:
            dim (str): Target dimension.

        Returns:
            int: Index of target dimension.
        """
        return self.dims.index(dim)

    def size(self, dim: str) -> int:
        """Gets size of target dimension.

        Args:
            dim (str): Target dimension.

        Returns:
            int: Size of target dimension.
        """
        return self.sizes[self.idx(dim)]

    def pseudo(self, dim: str) -> bool:
        """Determine if the target dimension is pseudo.

        Args:
            dim (str): Target dimension.

        Returns:
            bool: If the target dimension is pseudo.
        """
        return self.pseudos[self.idx(dim)]

    def rename(self, old: str, new: str) -> NamedShape:
        """Rename a dimension.

        Args:
            old (str): Dimension to rename,
            new (str): New name for {old}.

        Raises:
            ValueError: If {new} is already in dims.

        Returns:
            NamedShape: Copy of {self} with dimension {old} renamed to {new}.
        """
        if new in self.dims:
            raise ValueError(f"{new} already in dims")
        res = self.clone()
        res.dims[self.idx(old)] = new
        return res

    def select(self, dims: Union[str, List[str]]) -> NamedShape:
        """Select a subset of dimensions.

        Args:
            dims (Union[str, List[str]]): Dimensions to select.

        Returns:
            NamedShape: Copy with {self} with selected dims.
        """
        if isinstance(dims, str):
            dims = [dims]
        res_dims = dims
        res_sizes = [self.size(dim) for dim in dims]
        res_pseudos = [self.pseudo(dim) for dim in dims]
        return NamedShape(res_dims, res_sizes, res_pseudos)

    def reduce(self, reduce_dims: Union[str, List[str]]) -> NamedShape:
        """Reduce (remove) target dimensions.

        Args:
            reduce_dims (Union[str, List[str]]): Target dimensions.

        Raises:
            ValueError: If {self.dims} does not contain all reduce_dims.

        Returns:
            NamedShape: Copy of {self} with target dimensions removed.
        """
        if isinstance(reduce_dims, str):
            reduce_dims = [reduce_dims]

        if not all([dim in self.dims for dim in reduce_dims]):
            raise ValueError("self.dims must contain all reduce_dims")

        res_dims = []
        res_sizes = []
        res_pseudos = []
        for (dim, size, pseudos) \
                in zip(self.dims, self.sizes, self.pseudos):
            if dim not in reduce_dims:
                res_dims.append(dim)
                res_sizes.append(size)
                res_pseudos.append(pseudos)

        return NamedShape(res_dims, res_sizes, res_pseudos)

    def squeeze(
        self,
        dims: Optional[Union[str, List[str]]] = None
    ) -> NamedShape:
        """Squeeze (remove) pseudo-dimensions.

        Args:
            dims (Optional[Union[str, List[str]]]): \
                Dimensions to squeeze. Defaults to None, in which case all \
                pseudo-dimensions will be squeezed.

        Raises:
            ValueError: If dims contains a true dimension.

        Returns:
            NamedShape: Copy of {self} with pseudo-dims removed.
        """
        if dims is None:
            res = self.clone()
            for idx in reversed(self.ndim):
                if self.pseudos[idx]:
                    res.dims.pop(idx)
                    res.sizes.pop(idx)
                    res.pseudos.pop(idx)
            return res

        elif dims == []:
            return self.clone()

        else:
            if isinstance(dims, str):
                dims = [dims]
            res = self.clone()
            idx = res.idx(dims[0])
            if not res.pseudos[idx]:
                raise ValueError(f"cannot squeeze true dim {dims[0]}")
            res.dims.pop(idx)
            res.sizes.pop(idx)
            res.pseudos.pop(idx)
            return res.squeeze(dims[1:])

    def squeeze_as(self, other: NamedShape) -> NamedShape:
        """Squeeze dimensions of {self} not in {other}.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedShape: Copy of {self} with no dimensions not in {other}.
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
    ) -> NamedShape:
        """Unsqueeze (add) pseudo-dimensions.

        Args:
            dims (Union[str, List[str]]): Pseudo-dimensions to add.
            idx (Optional[Union[int, List[int]]]): Indices for pseudo-\
                dimensions. Defaults to None, which puts dims at the end.

        Raises:
            ValueError: If dim already exists in {self}.

        Returns:
            NamedShape: Copy of {self} with pseudo-dimensions added.
        """
        if isinstance(dims, str):
            dim = dims
            idx = idxs
            if dim in self.dims:
                raise ValueError("cannot add dim already in dims")
            if idx is None:
                idx = -1
            if idx < 0:
                idx += (self.ndim + 1)
            res = self.clone()
            res.dims.insert(idx, dim)
            res.sizes.insert(idx, 1)
            res.pseudos.insert(idx, True)
        else:
            if idxs is None:
                idxs = [-1] * len(dims)
            res = self.clone()
            for (dim, idx) in zip(dims, idxs):
                res = res.unsqueeze(dim, idx)
        return res

    def unsqueeze_as(self, other: NamedShape) -> NamedShape:
        """Unsqueeze dimensions of {other} not in {self}.

        New dimensions are placed at the end.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedShape: Copy of {self} with all the dimensions of {other}.
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
    ) -> NamedShape:
        """Expand target dimensions to target sizes.

        If target dimensions are pseudo-dimensions, this converts them to true \
            dimensions.

        Args:
            dims (str): Target dimensions.
            sizes (int): Target sizes.

        Raises:
            ValueError: If target dimension is a true dimension and it is \
                being expanded into a size that is not equal to its \
                existing size.

        Returns:
            NamedShape: Copy of {self} with dimensions expanded to sizes.
        """
        if isinstance(dims, str):
            dim = dims
            size = sizes
            if not self.pseudo(dim):
                if self.size(dim) == size:
                    return self.clone()
                else:
                    raise ValueError(f"cannot expand true dim {dim} with size "
                                     f"{self.size(dim)} into size {size}")
            else:
                res = self.clone()
                idx = res.idx(dim)
                res.sizes[idx] = size
                res.pseudos[idx] = False
                return res
        else:
            res = self.clone()
            for (dim, size) in zip(dims, sizes):
                res = res.expand(dim, size)
            return res

    def expand_as(self, other: NamedShape) -> NamedShape:
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
            NamedShape: Copy of {self} with dimensions expanded to \
                sizes in {other}.
        """
        if not set(other.dims) == set(self.dims):
            raise ValueError("self.dims not 1:1 with other.dims")

        res = self.clone()
        for (dim, size, pseudo) in zip(other.dims, other.sizes, other.pseudos):
            if not pseudo:
                res = res.expand(dim, size)

        return res

    def union(self, other: NamedShape) -> NamedShape:
        """Computes the union of two shapes.

        In the returned result, the dimensions of {self} come before \
            dimensions that are only in {other}. For shared dimensions, \
            the size is the maximum of their sizes in {self} and {other}.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedShape: Union of {self} and {other}.
        """
        return (self
                .unsqueeze_as(other)
                .expand_as(
                    other
                    .unsqueeze_as(self)))

    def intersection(self, other: NamedShape) -> NamedShape:
        """Computes the intersection of two shapes.

        The returned result contains only the shared dimensions of {self} and \
            {other}, in the order that they appear in {self}. The size for \
            each dimension is the maximum of its size in {self} and {other}.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedShape: Intersection of {self} and {other}.
        """
        union = self.union(other)
        self_only = union.reduce(other.dims)
        other_only = union.reduce(self.dims)
        return union.reduce(self_only.dims).reduce(other_only.dims)

    def transpose(self, dim0: str, dim1: str) -> NamedShape:
        """Swap the places of dim0 and dim1.

        Args:
            dim0 (str): Target dimension.
            dim1 (str): Target dimension.

        Returns:
            NamedShape: Copy of {self} with dim0 and dim1 swapped.
        """
        res = self.clone()
        idx0, idx1 = res.idx(dim0), res.idx(dim1)
        res.dims[idx0] = dim1
        res.sizes[idx0] = self.size(dim1)
        res.pseudos[idx0] = self.pseudo(dim1)
        res.dims[idx1] = dim0
        res.sizes[idx1] = self.size(dim0)
        res.pseudos[idx1] = self.pseudo(dim0)
        return res

    def permute(self, dims: List[str]) -> NamedShape:
        """Permutes the ordering of dims to the target ordering.

        Args:
            dims (List[str]): Target dimension ordering.

        Raises:
            ValueError: If {self.dims} and dims in target ordering are not \
                the same.

        Returns:
            NamedShape: Copy of {self} with dimensions in target ordering.
        """
        if set(dims) != set(self.dims):
            raise ValueError("dims not 1:1 with self.dims")
        idxs = [self.idx(dim) for dim in dims]
        res_dims = [self.dims[idx] for idx in idxs]
        res_sizes = [self.sizes[idx] for idx in idxs]
        res_pseudos = [self.pseudos[idx] for idx in idxs]
        return NamedShape(res_dims, res_sizes, res_pseudos)

    def permute_as(self, other: NamedShape) -> NamedShape:
        """Permute the ordering of dims to match the ordering in {other}.

        Args:
            other (NamedShape): Target shape.

        Returns:
            NamedShape: Copy of {self} with dimensions reordered \
                to match {other}.
        """
        return self.permute(other.dims)

    def with_idx(self, dim: str, idx: int) -> NamedShape:
        """Move dimension to target index.

        Args:
            dim (str): Target dimension.
            idx (int): Target index.

        Returns:
            NamedShape: Copy of {self} with dimension moved to target index.
        """
        res_dims = list(filter(lambda d: d != dim), self.dims)
        if idx < 0:
            idx += (self.ndim + 1)
        res_dims.insert(idx, dim)
        return self.permute(res_dims)

    def with_first(self, dim: str) -> NamedShape:
        """Move dimension to front.

        Args:
            dim (str): Target dimension.

        Returns:
            NamedShape: Copy of {self} with target dimension moved to front.
        """
        return self.with_idx(dim, 0)

    def with_last(self, dim: str) -> NamedShape:
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
    ) -> NamedShape:
        """Join target dimensions into a new dimension.

        Args:
            dims (List[str]): Dimensions to join.
            new_dim (str): Name of new dimension.
            idx (Optional[int]): Index of new dimension. \
                Defaults to None, which places it at the index of \
                {dims[0]}.

        Raises:
            ValueError: If trying to join a combination of true and pseudo-\
                dimensions.

        Returns:
            NamedShape: Copy of {self} with {dims} joined into {new_dim}.
        """
        if idx is None:
            idx = self.idx(dims[0])

        pseudos = [self.pseudo(dim) for dim in dims]
        if not all([p == pseudos[0] for p in pseudos[1:]]):
            raise ValueError("cannot join combination of true and pseudo "
                             "dims - shape is ambiguous. expand the pseudo "
                             "dims to their true size first.")

        # first permute dims into correct ordering
        tmp_dims = list(filter(lambda d: d not in dims, self.dims))
        new_size = 1
        dim_idx = idx
        for dim in dims:
            tmp_dims.insert(dim_idx, dim)
            dim_idx += 1
            new_size *= self.size(dim)
        res = self.permute(tmp_dims)

        res = res.reduce(dims)
        res.dims.insert(idx, new_dim)
        res.sizes.insert(idx, new_size)
        res.pseudos.insert(idx, pseudos[0])
        return res

    def split_dim(
        self,
        dim: str,
        new_dims: List[str],
        new_sizes: Optional[List[int]] = None
    ) -> NamedShape:
        """Split target dimension into new dims.

        Args:
            dim (str): Dimension to split.
            new_dims (List[str]): Names of new dimensions.
            new_sizes (Optional[List[int]]): Sizes of new dimensions.\
                Defaults to None, which can only be used if the dimension being\
                split is pseudo-.

        Raises:
            ValueError: If {new_sizes} is None but {dim} is not pseudo-.
            ValueError: If the product of the {new_sizes} is not equal to the \
                size of {dim}.

        Returns:
            NamedShape: Copy of {self} with {dim} split into {new_dims}.
        """
        if new_sizes is None:
            if not self.pseudo(dim):
                raise ValueError(f"must provide new sizes to split "
                                 f"real dim {dim}")
            else:
                new_sizes = [1] * len(new_dims)

        if not self.pseudo(dim) and int(np.prod(new_sizes)) != self.size(dim):
            raise ValueError(f"product of new_sizes does not match size "
                             f"{self.size(dim)} of real dim {dim}")

        res = self.clone()
        pseudo = res.pseudo(dim)
        idx = res.idx(dim)
        res = res.reduce(dim)
        for (dim, size) in zip(new_dims, new_sizes):
            res.dims.insert(idx, dim)
            res.sizes.insert(idx, size)
            res.pseudos.insert(idx, pseudo)
            idx += 1
        return res

    def __repr__(self):
        dim_str = [f"{dim:<12}" for dim in self.dims]
        size_str = [f"{size:<12}" for size in self.sizes]
        pseudos_str = ["True        " if pseudos else "False       "
                       for pseudos in self.pseudos]
        return (f"NamedShape: {self.name}\n"
                f"Dims      : {dim_str}\n"
                f"Sizes     : {size_str}\n"
                f"Unsqueezed: {pseudos_str}"
                .replace("'", ""))

    def __eq__(self, other: NamedShape):
        return (
            (self.dims == other.dims)
            and (self.sizes == other.sizes)
            and (self.pseudos == other.pseudos))
