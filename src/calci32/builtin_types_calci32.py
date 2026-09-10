"""
builtin_types_calci.py

Special types which cannot be declared on code files and are built-in the compiler. These types are
specially designed for the CALCI architecture.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from src.calcic_types import *
from src.parser import Context
from src.global_context import globalContext

class BuiltInTypes_calci32:
    def __init__(self, ctx: Context) -> None:
        self.context = ctx

        # Initialize the context built-in types.
        # TODO
        