def test_math_smoke():
    # Check numpy is present and usable
    import numpy as np

    a = np.array([1.0, 2.0, 3.0])
    assert a.mean() == 2.0
