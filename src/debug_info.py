"""
debug_info.py

Information shared through all stages of the compiler to generate debugging information.

calcic. Written by @dabecart, 2026.
"""

from dataclasses import dataclass, field

from src.calcic_types import *

# Locations in file. Where breakpoints can be positioned by the debugger.
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

@dataclass
class VariableDebugInformation:
    name: str
    file: str
    declLine: int
    declCol: int
    idType: DeclaratorType

    mangledIdentifier: str
    # Offset from the frame base register (fbreg).
    memoryLocation: int

@dataclass
class SubprocessDebugInfo:
    name: str
    file: str
    declLine: int
    declCol: int
    returnType: DeclaratorType

    innerVariables: list[VariableDebugInformation] = field(default_factory=list)

@dataclass
class ProgramDebugInfo:
    file: str                   = ""
    compilationDirectory: str   = "" # pwd
    
    subprocesses: list[SubprocessDebugInfo] = field(default_factory=list)
