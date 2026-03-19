"""
debug_info.py

Information shared through all stages of the compiler to generate debugging information.

calcic. Written by @dabecart, 2026.
"""

from dataclasses import dataclass, field
from typing import Any

from src.global_context import *
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
class SubprogramDebugInfo:
    name: str
    file: str
    declLine: int
    declCol: int
    returnType: DeclaratorType
    isGlobal: bool

    innerVariables: list[VariableDebugInformation] = field(default_factory=list)

@dataclass
class ProgramDebugInfo:
    file: str                   = ""
    compilationDirectory: str   = "" # pwd
    
    subprograms: list[SubprogramDebugInfo] = field(default_factory=list)

class DebugValueTypes(enum.Enum):
    BYTE = enum.auto()
    SHORT = enum.auto()
    LONG = enum.auto()
    QUAD = enum.auto()
    FLOAT = enum.auto()
    DOUBLE = enum.auto()
    STRING = enum.auto()
    ULEB128 = enum.auto()
    SLEB128 = enum.auto()
    # Assembler label.
    LABEL = enum.auto()

@dataclass
class DebugValue:
    type: DebugValueTypes
    value: Any
    
    @property
    def bytesize(self) -> int:
        match self.type:
            case DebugValueTypes.BYTE:
                return 1
            case DebugValueTypes.SHORT:
                return 2
            case DebugValueTypes.LONG:
                return 4
            case DebugValueTypes.QUAD:
                return 8
            case DebugValueTypes.FLOAT:
                return 4
            case DebugValueTypes.DOUBLE:
                return 8
            case DebugValueTypes.STRING:
                return len(self.value) + 1
            case DebugValueTypes.ULEB128:
                n = int(self.value)
                if n < 0:
                    raise ValueError("ULEB128 is only for non-negative integers.")
                if n == 0:
                    return 1
                # n.bit_length() gives the minimum bits needed. 
                # We add 6 before dividing by 7 to perform a ceiling division.
                return (n.bit_length() + 6) // 7            
            
            case DebugValueTypes.SLEB128:
                n = int(self.value)
                size = 0
                while True:
                    size += 1
                    byte = n & 0x7f  # Extract low 7 bits
                    n >>= 7          # Arithmetic shift right
                    
                    # Termination conditions:
                    # 1. Positive: Remaining is 0 AND the high bit of the current 7-bit byte is 0.
                    # 2. Negative: Remaining is -1 AND the high bit of the current 7-bit byte is 1.
                    if (n == 0 and (byte & 0x40) == 0) or (n == -1 and (byte & 0x40) != 0):
                        break
                return size
            
            case DebugValueTypes.LABEL:
                return globalContext.ADDRS_SIZE
            
            case _:
                raise ValueError()

@dataclass
class DebugSections:
    # debug_info
    # debug_abbrev
    # debug_line
    debug_str: dict[str, str]           = field(default_factory=dict)
    debug_line_str: dict[str, str]      = field(default_factory=dict)

# Global variables.
debugSections = DebugSections()
debugInfo = ProgramDebugInfo()