"""
builtin_types.py

Special types which cannot be declared on code files and are built in the compiler.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.calcic_types import *

@dataclass
class BuiltInTypes:
    va_list: DeclaratorType =  field(default_factory=lambda: TypeSpecifier.VOID.toBaseType())