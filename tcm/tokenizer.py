# Copyright (C) 2023-2024 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of tcm.
#
# tcm is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tcm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tcm. If not, see <http://www.gnu.org/licenses/>
"""Topic Context Model (TCM) tokenizer module."""

import re

from typing import List


def default_tokenizer(text: str) -> List[str]:
    """Tokenize words using a simple regex matching word boundaries.

    Args:
     * text: text to tokenize

    Returns:
     * text as a list of words
    """
    words = []
    text = re.sub(r"https?://[^\s]+", "URL", text)
    for m in re.finditer(r"\b(\w+(-\w+)+|\w+&\w+|\w+)\b", text):
        words.append(
            "NUM"
            if re.fullmatch(r"\d+", m.group())
            else (m.group() if m.group() == "URL" else m.group().lower())
        )
    words = [
        word
        for i, word in enumerate(words)
        if i == 0 or (word == "NUM" and words[i - 1] != word) or word != "NUM"
    ]
    return words
