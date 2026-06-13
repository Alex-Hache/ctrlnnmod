import pytest

from ctrlnmod.diagram.graph import WiringGraph, AlgebraicLoopError, WiringError


def _ports(n_in, n_out, width=1):
    return ({f"i{k}": width for k in range(n_in)}, {f"o{k}": width for k in range(n_out)})


def test_topological_order_simple_chain():
    # a (ft) -> b (ft) -> c (ft); a seeded
    block_ports = {
        "a": _ports(1, 1), "b": _ports(1, 1), "c": _ports(1, 1),
    }
    g = WiringGraph(
        block_ports=block_ports,
        block_feedthrough={"a": True, "b": True, "c": True},
        connections=[(("a", "o0"), ("b", "i0")), (("b", "o0"), ("c", "i0"))],
        seeded_inputs=[("a", "i0")],
    )
    assert g.order.index("a") < g.order.index("b") < g.order.index("c")


def test_strictly_proper_source_breaks_loop():
    # dynamic strictly-proper plant p (not feedthrough) feeds controller k (ft),
    # k feeds back into p. No algebraic loop because p is a source.
    block_ports = {"p": _ports(1, 1), "k": _ports(1, 1)}
    g = WiringGraph(
        block_ports=block_ports,
        block_feedthrough={"p": False, "k": True},
        connections=[(("p", "o0"), ("k", "i0")), (("k", "o0"), ("p", "i0"))],
        seeded_inputs=[],
    )
    # only feedthrough block k appears in order
    assert g.order == ["k"]


def test_algebraic_loop_raises():
    block_ports = {"a": _ports(1, 1), "b": _ports(1, 1)}
    with pytest.raises(AlgebraicLoopError):
        WiringGraph(
            block_ports=block_ports,
            block_feedthrough={"a": True, "b": True},
            connections=[(("a", "o0"), ("b", "i0")), (("b", "o0"), ("a", "i0"))],
            seeded_inputs=[],
        )


def test_width_mismatch_raises():
    block_ports = {"a": _ports(1, 1, width=1), "b": _ports(1, 1, width=2)}
    with pytest.raises(WiringError):
        WiringGraph(
            block_ports=block_ports,
            block_feedthrough={"a": True, "b": True},
            connections=[(("a", "o0"), ("b", "i0"))],
            seeded_inputs=[("a", "i0")],
        )


def test_unconnected_input_raises():
    block_ports = {"a": _ports(1, 1)}
    with pytest.raises(WiringError):
        WiringGraph(
            block_ports=block_ports,
            block_feedthrough={"a": True},
            connections=[],
            seeded_inputs=[],  # a.i0 left unconnected
        )


def test_double_driven_input_raises():
    block_ports = {"a": _ports(0, 1), "b": _ports(0, 1), "c": _ports(1, 1)}
    with pytest.raises(WiringError):
        WiringGraph(
            block_ports=block_ports,
            block_feedthrough={"a": False, "b": False, "c": True},
            connections=[(("a", "o0"), ("c", "i0")), (("b", "o0"), ("c", "i0"))],
            seeded_inputs=[],
        )
