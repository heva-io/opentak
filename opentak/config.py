from typing import Literal

class Config:
    """Static class containing configuration attributes that can be used accross entire projects.

    Initialize config class by running `Config.init(...)` method.
    Access attributes by calling them directly: `Config.my_attr`
    """

    lang: str
    round_digits: int

    @staticmethod
    def init(lang: Literal["ENG", "FRA"], round_digits: int = 2):
        """Initialize Config static class.

        :param lang: Language used for reports.
        :param round_digits: Number of digits used by default when rounding numbers.
        """
        Config.lang = lang
        Config.round_digits = round_digits
