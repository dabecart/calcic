"""
builtin_types_x64.py

Special types which cannot be declared on code files and are built-in the compiler. These types are
specially designed for the x64 architecture.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from src.calcic_types import *
from src.parser import Context
from src.global_context import globalContext

class BuiltInTypes_x64:
    def __init__(self, ctx: Context) -> None:
        self.context = ctx

        # Initialize the context built-in types.
        globalContext.builtInTypes.va_list = self.builtIn_va_list()

    def builtIn_va_list(self) -> DeclaratorType:
        """
        According to Section 3.5.7 of the System V ABI:
        typedef struct {
            unsigned int gp_offset;
            unsigned int fp_offset;
            void *overflow_arg_area;
            void *reg_save_area;
        } va_list[1];
        
        This struct is defined internally as __builtin_va_list.
        """

        # Anonymous struct inside the typedef.
        originalIdentifier = "__builtin_va_list.anon"
        mangledIdentifier = self.context.mangleIdentifier(originalIdentifier)
        byteSize = 24
        alignment = 8
        members = [
            ParameterInformation(TypeSpecifier.UINT.toBaseType(), "gp_offset", 0),
            ParameterInformation(TypeSpecifier.UINT.toBaseType(), "fp_offset", 4),
            ParameterInformation(PointerDeclaratorType(TypeSpecifier.VOID.toBaseType()), 'overflow_arg_area', 8),
            ParameterInformation(PointerDeclaratorType(TypeSpecifier.VOID.toBaseType()), 'reg_save_area', 16)
        ]
        t = TypeSpecifier.STRUCT(originalIdentifier, mangledIdentifier, byteSize, alignment, members)
        self.context.addStruct(originalIdentifier, mangledIdentifier, t)
        self.context.completeStruct(originalIdentifier, byteSize, alignment, members)

        # Array type.
        arrT = ArrayDeclaratorType(t.toBaseType(), 1)

        # Setup the typedef.
        arrIdentifier = "__builtin_va_list"
        arrMangledIdentifier = self.context.mangleIdentifier(arrIdentifier)
        arrT.setAlias(arrIdentifier)
        self.context.addTypedefIdentifier(arrIdentifier, arrMangledIdentifier, arrT)

        return arrT

