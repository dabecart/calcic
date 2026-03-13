"""
debug_info.py

Information shared through all stages of the compiler to generate debugging information.

calcic. Written by @dabecart, 2026.
"""

from dataclasses import dataclass

# Locations in file.
@dataclass
class DebugLocator:
    file: str       = "?"
    lineStart: int  = -1
    lineEnd: int    = -1
    colStart: int   = -1
    colEnd: int     = -1

    astName: str = ""

    def __str__(self) -> str:
        return f'{self.astName} {self.file}:{self.lineStart}:{self.colStart}'