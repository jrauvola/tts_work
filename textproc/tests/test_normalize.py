from textproc.app import normalize


def test_abbreviations():
    assert normalize("Dr. Smith lives on St. Paul Ave.").startswith("Doctor Smith lives on Street Paul Avenue")


def test_numbers_to_words():
    assert "one hundred twenty three" in normalize("There are 123 apples.")


def test_math_integral():
    s = normalize(r"\\int_0^1 x^2 dx")
    assert "the integral from 0 to 1 of x to the power of 2 dx" in s


def test_control_chars_removed():
    assert normalize("hello\x00world") == "hello world"


