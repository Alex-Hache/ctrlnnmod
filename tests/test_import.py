def test_import_ctrlnmod():
    import sys
    print("PYTHONPATH:", sys.path)
    import ctrlnmod
    assert ctrlnmod is not None
