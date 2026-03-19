"""
DWARF_V.py

DWARF Version V is a standard used to add debug information to a program. The standard can be 
downloaded from http://www.dwarfstd.org/.

calcic. Written by @dabecart, 2026.
"""

from src.debug_info import *
from src.debug.DWARF_V_definitions import *

class DWARF_DebugInfo:
    def __init__(self, debugInfo: ProgramDebugInfo) -> None:
        # First DIE entry is the DW_TAG_compile_unit.
        self.compileUnit = DIE(
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
                DWAttribute(DW_AT.DW_AT_high_pc, DW_FORM.DW_FORM_data8, ".Lendtext-.Ltext"),
                DWAttribute(DW_AT.DW_AT_stmt_list, DW_FORM.DW_FORM_sec_offset, ".Ldebug_line"),
            ]
        )

        # For all subprograms (functions in C), create a DW_TAG_subprogram.
        for subp in debugInfo.subprograms:
            self.compileUnit.children.append(self.createSubprogramDIE(subp))

        # Once the subprogram is parsed, add a DW_AT_sibling to each children for the debugger to 
        # easily jump between DIEs.
        # DWAttribute(DW_AT.DW_AT_sibling, DW_FORM.DW_FORM_ref8, ...),

    def createSubprogramDIE(self, info: SubprogramDebugInfo) -> DIE:
        # Create the DIE for the return type.
        typeDIE = self.createTypeDIE(info.returnType)

        # Create the DIE for the function.
        subprogram = DIE(
            tag=DW_TAG.DW_TAG_subprogram,
            hasChildren=True,
            attributes=[
                DWAttribute(DW_AT.DW_AT_name, DW_FORM.DW_FORM_strp, info.name), 
                # There will be only one file per compilation unit, and this variable belongs to it.
                DWAttribute(DW_AT.DW_AT_decl_file, DW_FORM.DW_FORM_implicit_const, 1), 
                DWAttribute(DW_AT.DW_AT_decl_line, DW_FORM.DW_FORM_data8, info.declLine),
                DWAttribute(DW_AT.DW_AT_decl_column, DW_FORM.DW_FORM_data8, info.declCol),
                DWAttribute(DW_AT.DW_AT_type, DW_FORM.DW_FORM_ref8, typeDIE),
                DWAttribute(DW_AT.DW_AT_low_pc, DW_FORM.DW_FORM_addr, f".L{info.name}"),
                DWAttribute(DW_AT.DW_AT_high_pc, DW_FORM.DW_FORM_data8, f".Lend{info.name}-.L{info.name}"),
                # cfa = Canonical Frame Address. When unwind tables are implemented we can use this.
                # DWAttribute(DW_AT.DW_AT_frame_base, DW_FORM.DW_FORM_exprloc, DW_OP.DW_OP_call_frame_cfa),
                # In the meantime, use the RBP register (register 6, according to the AMD64 ABI).
                DWAttribute(DW_AT.DW_AT_frame_base, DW_FORM.DW_FORM_exprloc, DW_OP.DW_OP_reg6),
                DWAttribute(DW_AT.DW_AT_call_all_calls, DW_FORM.DW_FORM_flag_present),
            ]
        )

        if info.isGlobal:
            subprogram.attributes.append(
                DWAttribute(DW_AT.DW_AT_external, DW_FORM.DW_FORM_flag_present)
            )

        # Create the variables' DIEs inside the subprogram.
        for var in info.innerVariables:
            subprogram.children.append(self.createVariableDIE(var))

        return subprogram
    
    def createVariableDIE(self, info: VariableDebugInformation) -> DIE:
        # Parse the DIE of the variable type.
        typeDIE = self.createTypeDIE(info.idType)

        return DIE(
            tag=DW_TAG.DW_TAG_subprogram,
            hasChildren=True,
            attributes=[
                DWAttribute(DW_AT.DW_AT_name, DW_FORM.DW_FORM_string, info.name), 
                # There will be only one file per compilation unit, and this variable belongs to it.
                DWAttribute(DW_AT.DW_AT_decl_file, DW_FORM.DW_FORM_implicit_const, 1), 
                DWAttribute(DW_AT.DW_AT_decl_line, DW_FORM.DW_FORM_data8, info.declLine),
                DWAttribute(DW_AT.DW_AT_decl_column, DW_FORM.DW_FORM_data8, info.declCol),
                DWAttribute(DW_AT.DW_AT_type, DW_FORM.DW_FORM_ref8, typeDIE),
                # The variable is stored in the stack, it is represented by an offset from the frame
                # base register RBP.
                # When unwind tables are added:
                # DWAttribute(DW_AT.DW_AT_location, DW_FORM.DW_FORM_exprloc, DW_OP.DW_OP_fbreg, info.memoryLocation),
                DWAttribute(DW_AT.DW_AT_location, DW_FORM.DW_FORM_exprloc, DW_OP.DW_OP_breg6, info.memoryLocation),
            ]
        )
    
    def createTypeDIE(self, t: DeclaratorType) -> DIE:
        # Remove the qualifiers, they don't matter to the debugger. 
        t = t.unqualified()

        if isinstance(t, BaseDeclaratorType):
            if t.baseType.isDecimal():
                ate = DW_ATE.DW_ATE_float
            elif t.baseType == TypeSpecifier.CHAR:
                ate = DW_ATE.DW_ATE_signed_char
            elif t.baseType.isSignedInt():
                ate = DW_ATE.DW_ATE_signed
            else:
                ate = DW_ATE.DW_ATE_unsigned

            ret = DIE(
                tag=DW_TAG.DW_TAG_base_type,
                hasChildren=False,
                attributes=[
                    DWAttribute(DW_AT.DW_AT_byte_size, DW_FORM.DW_FORM_data1, t.getByteSize()),
                    DWAttribute(DW_AT.DW_AT_encoding, DW_FORM.DW_FORM_data1, ate.value),
                    DWAttribute(DW_AT.DW_AT_name, DW_FORM.DW_FORM_string, str(t)),
                ]
            )
        else:
            raise ValueError("Not implemented")
        
        return ret