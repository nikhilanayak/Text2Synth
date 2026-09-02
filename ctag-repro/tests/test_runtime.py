from ctag_repro.runtime import COLAB_RUNTIME_SPECS, version_issues


def test_current_colab_matrix_is_accepted_with_cuda_local_versions():
    versions = {
        package: requirement.removeprefix("==").removesuffix(".*")
        for package, requirement in COLAB_RUNTIME_SPECS.items()
    }
    versions["python"] = "3.13.15"
    versions["torch"] = "2.11.0+cu128"
    versions["torchvision"] = "0.26.0+cu128"

    assert version_issues(versions) == []


def test_incompatible_accelerator_version_is_reported():
    versions = {
        package: requirement.removeprefix("==").removesuffix(".*")
        for package, requirement in COLAB_RUNTIME_SPECS.items()
    }
    versions["python"] = "3.13.15"
    versions["jax"] = "0.7.2"

    assert version_issues(versions) == [
        "jax 0.7.2 does not satisfy ==0.11.1"
    ]


def test_missing_package_is_reported():
    versions = {
        package: requirement.removeprefix("==").removesuffix(".*")
        for package, requirement in COLAB_RUNTIME_SPECS.items()
        if package != "synthax"
    }
    versions["python"] = "3.13.15"

    assert version_issues(versions) == [
        "synthax is missing (expected ==0.2.2)"
    ]
