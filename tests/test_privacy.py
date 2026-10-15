import pytest

from autoforge.policy.privacy import Access, PrivacyDenied, data_access, require_access
from autoforge.schemas.core import BrainKind, Sensitivity


@pytest.mark.parametrize(
    "brain,sens,expected",
    [
        (BrainKind.local, Sensitivity.sensitive, Access.full),
        (BrainKind.api, Sensitivity.public, Access.full),
        (BrainKind.api, Sensitivity.internal, Access.summary),
        (BrainKind.subscription, Sensitivity.internal, Access.summary),
        (BrainKind.api, Sensitivity.sensitive, Access.denied),
        (BrainKind.subscription, Sensitivity.sensitive, Access.denied),
    ],
)
def test_access_matrix(brain, sens, expected):
    assert data_access(brain, sens) is expected


def test_require_access_raises_for_sensitive_cloud():
    with pytest.raises(PrivacyDenied) as e:
        require_access(BrainKind.api, Sensitivity.sensitive)
    assert e.value.code == "PRIVACY_DENIED"
