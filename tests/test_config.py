from opentak.config import Config


def test_config():
    # Given
    Config.init(lang="ENG")

    # Then
    assert Config.lang == "ENG"
    assert Config.round_digits == 2

    # Check that we can update values
    Config.init(lang="ENG", round_digits=4)
    assert Config.round_digits == 4
