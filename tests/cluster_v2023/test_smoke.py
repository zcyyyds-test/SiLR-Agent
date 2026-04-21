def test_package_importable():
    import domains.cluster_v2023
    assert hasattr(domains.cluster_v2023, "__version__")
