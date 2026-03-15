"""
DWARF_V.py

DWARF Version V is a standard used to add debug information to a program. The standard can be 
downloaded from http://www.dwarfstd.org/.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from src.debug_info import *
from src.debug.DWARF_V_definitions import *

# Debugging Information Entry
@dataclass
class DIE:
    tag: DW_TAG
    hasChildren: bool
    attributes: list[DWAttribute]   = field(default_factory=list)
    children: list[DIE]             = field(default_factory=list)

class DWARF_DebugInfo:
    def __init__(self, debugInfo: ProgramDebugInfo) -> None:
        # First DIE entry is the DW_TAG_compile_unit.
        compileUnit = DIE(
            tag=DW_TAG.DW_TAG_compile_unit,
            hasChildren=True,
            attributes=[
                DWAttribute(DW_AT.DW_AT_producer, DW_FORM.DW_FORM_strp, "calcic"),
                # From Table 7.17: DW_LANG_C99 = 0x0c
                DWAttribute(DW_AT.DW_AT_language, DW_FORM.DW_FORM_data1, 0x0c), 
                DWAttribute(DW_AT.DW_AT_name, DW_FORM.DW_FORM_line_strp, debugInfo.file), 
                DWAttribute(DW_AT.DW_AT_comp_dir, DW_FORM.DW_FORM_line_strp, debugInfo.compilationDirectory),
                DWAttribute(DW_AT.DW_AT_low_pc, DW_FORM.DW_FORM_addr, ".Ltext"),
                # TODO: For 64 byte systems, FORM_data8.
                DWAttribute(DW_AT.DW_AT_high_pc, DW_FORM.DW_FORM_data8, ".eLtext-.Ltext"),
                DWAttribute(DW_AT.DW_AT_stmt_list, DW_FORM.DW_FORM_sec_offset, ".Ldebug_line"),
            ]
        )