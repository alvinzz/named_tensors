import torch
import torch.nn.functional as F

from named_tensor import NamedTensor


def test_einsum():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 10
    X = 3
    R = 256
    C = 384

    X_IN = X
    X_OUT = 64
    KR = 3
    KC = 3

    im = NamedTensor(
        torch.rand(B, X, R, C).to(device),
        dims=['B', 'X_IN', 'R', 'C'])
    kernel = NamedTensor(
        torch.rand(X_OUT, X_IN, KR, KC).to(device),
        dims=['X_OUT', 'X_IN', 'KR', 'KC'])

    conv = F.conv2d(im.data, kernel.data, padding=(KR//2, KC//2))

    unfold = NamedTensor(
        torch.nn.functional.unfold(im.data, (KR, KC), padding=(KR//2, KC//2))
        .view(B, X_IN, KR, KC, R, C),
        dims=['B', 'X_IN', 'KR', 'KC', 'R', 'C'])
    unfold = unfold.permute(['KR', 'C', 'KC', 'B', 'R', 'X_IN'])
    assert unfold.shape.sizes == [KR, C, KC, B, R, X_IN]

    einsum = unfold.einsum(kernel, ['X_IN', 'KR', 'KC'])
    assert einsum.shape.sizes == [C, B, R, X_OUT]

    einsum = einsum.permute(['B', 'X_OUT', 'R', 'C'])
    assert torch.allclose(einsum.data, conv)


def test_reduce():
    test = NamedTensor(
        torch.arange(24).reshape(2, 3, 4),
        dims=['A', 'B', 'C'])

    assert torch.allclose(
        test.reduce(lambda t, idxs: t.sum(idxs), ['A']).tensor,
        test.tensor.sum(0))

    assert torch.allclose(
        test.reduce(lambda t, idxs: t.sum(idxs), ['B']).tensor,
        test.tensor.sum(1))

    assert torch.allclose(
        test.reduce(lambda t, idxs: t.sum(idxs), ['A', 'C']).tensor,
        test.tensor.sum((0, 2)))


def test_broadcasting():
    a = NamedTensor(
        torch.arange(2*3*4*5).reshape(1, 1, 1, 1, 2, 3, 4, 5),
        dims=['0', '1', '2', '3', 'A', 'D', 'C', 'B'],
        pseudos=[True, True, False, True, False, False, False, False])
    b = NamedTensor(
        torch.arange(2*3*4*5).reshape(4, 2, 3, 1, 5, 1, 1, 1),
        dims=['E', 'F', 'D', 'B', '0', '1', '4', '5'],
        pseudos=[False, False, False, True, False, True, False, True])

    res = (a > b)
    assert res.dims == \
        ['0', '1', '2', '3', 'A', 'D', 'C', 'B', 'E', 'F', '4', '5']
    assert res.sizes == \
        [5, 1, 1, 1, 2, 3, 4, 5, 4, 2, 1, 1]
    assert res.pseudos == \
        [False, True, False, True,
         False, False, False, False,
         False, False, False, True]

    assert torch.allclose(
        (a * b).tensor,
        (a.tensor.view(1, 1, 1, 1, 2, 3, 4, 5, 1, 1, 1, 1)
         * b.tensor
            .permute(4, 5, 2, 3, 0, 1, 6, 7)
            .view(5, 1, 1, 1, 1, 3, 1, 1, 4, 2, 1, 1)))


def test_multi():
    a = NamedTensor(torch.rand(3, 4, 5), dims=['A', 'B', 'C'])
    b = NamedTensor(torch.rand(3, 4, 5), dims=['D', 'B', 'C'])
    c = NamedTensor(torch.rand(3, 4, 5), dims=['A', 'E', 'C'])

    def mean(tensors):
        res = torch.zeros_like(tensors[0])
        for tensor in tensors:
            res = res + tensor
        return res / len(tensors)

    assert torch.allclose(
        NamedTensor.apply_multi(mean, [a, b, c]).tensor,
        (a.tensor.view(3, 4, 5, 1, 1)
         + b.tensor.permute(1, 2, 0).view(1, 4, 5, 3, 1)
         + c.tensor.permute(0, 2, 1).view(3, 1, 5, 1, 4)
         ) / 3.,
        atol=1e-3
    )
