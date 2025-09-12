import pytest

from opentak.tak_theme._utils import interpolate_colors, lighten_color, set_style


def test_color_multiple_forms_ok():
    # Given
    red_as_str = "red"
    red_as_tuple = (255, 0, 0)
    red_as_hex = "#FF0000"
    opacity = 0.8
    expected_color = "rgba(255, 0, 0, 0.8)"
    # When
    lightened_str = lighten_color(red_as_str, opacity)
    lightened_tuple = lighten_color(red_as_tuple, opacity)
    lightened_hex = lighten_color(red_as_hex, opacity)
    # Then
    assert lightened_str == expected_color
    assert lightened_tuple == expected_color
    assert lightened_hex == expected_color


@pytest.mark.parametrize(
    "colors, nb_colors",
    [
        (["#ff0000", "#0000ff"], 5),
        (["red", "blue"], 3),
    ],
)
def test_interpolate_colors_basic(colors, nb_colors):
    # Given / When
    result = interpolate_colors(colors, nb_colors)
    # Then
    assert len(result) == nb_colors
    assert all(isinstance(c, str) and c.startswith("#") for c in result)


def test_interpolate_colors_return_steps():
    # Given / When
    interpolated, steps = interpolate_colors(
        ["#00ff00", "#0000ff"], 4, return_steps=True
    )
    # Then
    assert len(interpolated) == 4
    assert len(steps) == 4
    assert steps[0] == 0
    assert steps[-1] == 1


@pytest.mark.parametrize(
    "colors, nb_colors, match",
    [
        (["#ff0000"], 3, "color_list should contain at least two colors"),
        (["#ff0000", "#00ff00"], 1, "nb_colors should be higher than color_list"),
    ],
)
def test_interpolate_colors_value_errors(colors, nb_colors, match):
    with pytest.raises(ValueError, match=match):
        interpolate_colors(colors, nb_colors)


def test_set_style():
    # Check that the function runs without error
    set_style()
