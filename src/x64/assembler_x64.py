"""
assembler_x64.py

Converts the TAC into x86-64 assembly language, using AT&T syntax. Handles all the nuances of the 
x64 language and the System V ABI.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from ..TAC import *
from ..types import *
from typing import Type, TypeVar
import math
import enum
from ..global_context import globalContext

AssemblyT = TypeVar("AssemblyT", bound="AssemblyAST")

class AssemblyBaseType(enum.Enum):
    DOUBLE   = "sd"
    FLOAT    = "ss"
    QUADWORD = "q"
    LONGWORD = "l"
    WORD     = "w"
    BYTE     = "b"

    # Not a real type, but used to represent an unmapped region of memory.
    BYTEARRAY = "?"

class AssemblyClassType(enum.Enum):
    # Written in order of decreasing priority.
    MEMORY   = enum.auto()
    INTEGER  = enum.auto()
    SSE      = enum.auto()   # Float or double.
    NO_CLASS = enum.auto()   # Unclassified/padding.

# Used to classify eightbytes. Bytesize might be irregular.
class AssemblyClass:
    def __init__(self, classType: AssemblyClassType, byteSize: int) -> None:
        self.classType = classType
        self.byteSize = byteSize

    # To enable sorting in the enum classifying algorithm. 
    def __lt__(self, other):
        if not isinstance(other, AssemblyClass):
            raise ValueError()
        return self.classType.value < other.classType.value

    def toAssemblyType(self) -> AssemblyType:
        if self.classType == AssemblyClassType.SSE:
            match self.byteSize:
                case 8: asmbType = AssemblyType.DOUBLE
                case 4: asmbType = AssemblyType.FLOAT
                case _: raise ValueError()

        elif self.classType == AssemblyClassType.INTEGER:
            match self.byteSize:
                case 8: asmbType = AssemblyType.QUADWORD
                case 4: asmbType = AssemblyType.LONGWORD
                case 2: asmbType = AssemblyType.WORD
                case 1: asmbType = AssemblyType.BYTE
                # For non standard types (3, 5, 6 and 7 bytes)...
                case _: asmbType = AssemblyType(AssemblyBaseType.BYTEARRAY, self.byteSize, 0)
        else:
            asmbType = AssemblyType(AssemblyBaseType.BYTEARRAY, self.byteSize, 0)

        return asmbType

class AssemblyType:
    DOUBLE: AssemblyType
    FLOAT: AssemblyType
    QUADWORD: AssemblyType
    LONGWORD: AssemblyType
    WORD: AssemblyType
    BYTE: AssemblyType

    def __init__(self, baseType: AssemblyBaseType, size: int, alignment: int, sectionName: str = "",
                 members: list[AssemblyClass] = []) -> None:
        self.baseType = baseType
        self.size = size
        self.alignment = alignment
        self.sectionName = sectionName
        self.members = members

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AssemblyType):
            return False
        return (self.baseType == value.baseType) and (self.size == value.size) and (self.alignment == value.alignment)

    def isQuad(self) -> bool:
        return self.baseType in (AssemblyBaseType.DOUBLE, AssemblyBaseType.QUADWORD)
    
    def isDecimal(self) -> bool:
        return self.baseType in (AssemblyBaseType.DOUBLE, AssemblyBaseType.FLOAT)
    
    def isScalar(self) -> bool:
        return self.baseType in (AssemblyBaseType.QUADWORD, AssemblyBaseType.LONGWORD, AssemblyBaseType.WORD, AssemblyBaseType.BYTE)

    def getDataSectionName(self) -> str:
        if self.baseType == AssemblyBaseType.BYTEARRAY:
            raise ValueError()
        return self.sectionName
    
    def getTypeName(self) -> str:
        return self.baseType.value

    @staticmethod
    def _classifyArray(arrayDeclarator: ArrayDeclaratorType) -> list[AssemblyClass]:
        # TODO: Think if alignment is to be considered here.
        arrayBaseType = AssemblyType.fromTAC(arrayDeclarator.getArrayBaseType())
        return arrayBaseType.members * arrayDeclarator.size

    @staticmethod
    def _classifyStruct(structDeclarator: BaseDeclaratorType) -> list[AssemblyClass]:
        # Classify the structure.
        memberClass: list[AssemblyClass] = []
        structSize = structDeclarator.getByteSize()
        if structSize > 16:
            fullEightBytes = structSize // 8
            memberClass.extend([AssemblyClass(AssemblyClassType.MEMORY, 8)] * fullEightBytes)
            finalEightByte = structSize % 8
            if finalEightByte != 0:
                memberClass.append(AssemblyClass(AssemblyClassType.MEMORY, finalEightByte))
        else:
            # Convert the struct into a flat array of fields.
            flattenedMembers = structDeclarator.baseType.flattenStruct()
            # Convert these fields into assembly classes.
            asmbClasses: list[AssemblyClass] = []
            for mem in flattenedMembers:
                if mem == TypeSpecifier.VOID.toBaseType():
                    # This is padding.
                    asmbClasses.append(AssemblyClass(AssemblyClassType.NO_CLASS, 1))
                else:
                    asmbMembers = AssemblyType.fromTAC(mem).members
                    # To make the next step easier, split all the members into single byte members.
                    for subMember in asmbMembers:
                        asmbClasses.extend([AssemblyClass(subMember.classType, 1)] * subMember.byteSize)
            # Now group them into eightbytes.
            byteCount: int = 0
            currentClass: AssemblyClassType = AssemblyClassType.NO_CLASS
            for cl in asmbClasses:
                byteCount += 1

                # Stick to the most strict class type.
                if cl.classType.value < currentClass.value:
                    currentClass = cl.classType

                if byteCount == 8:
                    # New eightbyte.
                    memberClass.append(AssemblyClass(currentClass, 8))
                    byteCount = 0
                    currentClass = AssemblyClassType.NO_CLASS

            # There may be an incomplete eightbyte at the end.
            if byteCount != 0:
                memberClass.append(AssemblyClass(currentClass, byteCount))
        
        return memberClass
    
    @staticmethod
    def _classifyUnion(unionDeclarator: BaseDeclaratorType) -> list[AssemblyClass]:
        memberClass: list[AssemblyClass] = []

        unionSize = unionDeclarator.getByteSize()
        if unionSize > 16:
            fullEightBytes = unionSize // 8
            memberClass.extend([AssemblyClass(AssemblyClassType.MEMORY, 8)] * fullEightBytes)
            finalEightByte = unionSize % 8
            if finalEightByte != 0:
                memberClass.append(AssemblyClass(AssemblyClassType.MEMORY, finalEightByte))
        else:
            # Get the classes of each member but split them into 1 byte classes. This will make the 
            # matching algorithm easier to follow.
            classifiedMembers: list[list[AssemblyClass]] = []
            greatestMemberListCount: int = 0
            for member in unionDeclarator.baseType.getMembers():
                # Get the assembly type of the union member, this will calculate its members.
                asmbType = AssemblyType.fromTAC(member.type)
                # Now, split them into 1 byte classes.
                singleByteMemberList: list[AssemblyClass] = []
                for innerMember in asmbType.members:
                    singleByteMemberList.extend([AssemblyClass(innerMember.classType, 1)] * innerMember.byteSize)
                # Add to the outer loop list.
                classifiedMembers.append(singleByteMemberList)
                # Monitor which list of assembly members is bigger for the next for loop.
                if len(singleByteMemberList) > greatestMemberListCount:
                    greatestMemberListCount = len(singleByteMemberList)
            
            # Compare each byte of the classified members and get the most restrictive.
            byteClasses: list[AssemblyClass] = []
            for byteIndex in range(greatestMemberListCount):
                # Join all the classes at this byteIndex in an array and sort it.
                currentByteClasses: list[AssemblyClass] = []
                for memberIndex in range(len(classifiedMembers)):
                    # Check limits.
                    if byteIndex >= len(classifiedMembers[memberIndex]):
                        continue
                    
                    currentByteClasses.append(classifiedMembers[memberIndex][byteIndex])
                
                currentByteClasses.sort()
                # The first is the highest priority class of this eightbyte.
                byteClasses.append(AssemblyClass(currentByteClasses[0].classType, 1))

            # Finally, group the bytes into eightbytes.
            byteCount: int = 0
            currentClass: AssemblyClassType = AssemblyClassType.NO_CLASS
            for cl in byteClasses:
                byteCount += 1

                # Stick to the most strict class type.
                if cl.classType.value < currentClass.value:
                    currentClass = cl.classType

                if byteCount == 8:
                    # New eightbyte.
                    memberClass.append(AssemblyClass(currentClass, 8))
                    byteCount = 0
                    currentClass = AssemblyClassType.NO_CLASS

            # There may be an incomplete eightbyte at the end.
            if byteCount != 0:
                memberClass.append(AssemblyClass(currentClass, byteCount))
        
        return memberClass

    @staticmethod
    def fromTAC(tac: DeclaratorType) -> AssemblyType:
        if isinstance(tac, BaseDeclaratorType):
            match tac.baseType.name:
                case "CHAR" | "SIGNED_CHAR" | "UCHAR":   
                    return AssemblyType.BYTE
                case "SHORT" | "USHORT":  
                    return AssemblyType.WORD
                case "INT" | "UINT":    
                    return AssemblyType.LONGWORD
                case "LONG" | "ULONG":   
                    return AssemblyType.QUADWORD
                case "DOUBLE": 
                    return AssemblyType.DOUBLE
                case "FLOAT": 
                    return AssemblyType.FLOAT
                case "STRUCT":
                    return AssemblyType(
                            AssemblyBaseType.BYTEARRAY, 
                            tac.getByteSize(), tac.getAlignment(),
                            members=AssemblyType._classifyStruct(tac))
                case "UNION":
                    return AssemblyType(
                            AssemblyBaseType.BYTEARRAY, 
                            tac.getByteSize(), tac.getAlignment(),
                            members=AssemblyType._classifyUnion(tac))
                case "ENUM":
                    return AssemblyType.LONGWORD
                case _:
                    raise ValueError(f"Cannot convert {tac} to AssemblyType")
        elif isinstance(tac, ArrayDeclaratorType):
            byteLen = tac.getByteSize()
            # As the System V x64 ABI says...
            if byteLen > 16:
                # Alignment is always 16 when the size of the array is greater than 16.
                alignment = 16
            else:
                # Alignment is the same as the array base type.
                alignment = tac.getArrayBaseType().getByteSize()

            return AssemblyType(
                AssemblyBaseType.BYTEARRAY, 
                byteLen, alignment, 
                members=AssemblyType._classifyArray(tac))
        else:
            # Both pointers and functions are quad values in 64 bit systems.
            return AssemblyType.QUADWORD

# Common assembly types.
AssemblyType.DOUBLE      = AssemblyType(AssemblyBaseType.DOUBLE,    8, 8, "double", [AssemblyClass(AssemblyClassType.SSE,     8)])
AssemblyType.FLOAT       = AssemblyType(AssemblyBaseType.FLOAT,     4, 4, "float",  [AssemblyClass(AssemblyClassType.SSE,     4)])
AssemblyType.QUADWORD    = AssemblyType(AssemblyBaseType.QUADWORD,  8, 8, "quad",   [AssemblyClass(AssemblyClassType.INTEGER, 8)])
AssemblyType.LONGWORD    = AssemblyType(AssemblyBaseType.LONGWORD,  4, 4, "long",   [AssemblyClass(AssemblyClassType.INTEGER, 4)])
AssemblyType.WORD        = AssemblyType(AssemblyBaseType.WORD,      2, 2, "word",   [AssemblyClass(AssemblyClassType.INTEGER, 2)])
AssemblyType.BYTE        = AssemblyType(AssemblyBaseType.BYTE,      1, 1, "byte",   [AssemblyClass(AssemblyClassType.INTEGER, 1)])

class AssemblyAST(ABC):
    def __init__(self, parentAST: AssemblyAST|None = None) -> None:
        super().__init__()
        # Assembly AST containing this Assembly AST node.
        self.parent = parentAST

        self.firstPass()

    # First assembler pass.
    # Traverses the TAC AST and converts it into Assembly AST.
    @abstractmethod
    def firstPass(self):
        pass

    # Second assembler pass.
    # Used to replace the Pseudo registers for a position in the stack.
    @abstractmethod
    def secondPass(self):
        pass

    # Third assembler pass.
    # Fixes invalid instructions (instructions that use two memory locations).
    # Adds allocation calls for the stack.
    @abstractmethod
    def thirdPass(self):
        pass

    # Emits assembly.
    @abstractmethod
    def emitCode(self) -> str:
        pass

    # Used to debug.
    @abstractmethod
    def print(self) -> str:
        pass

    def __str__(self) -> str:
        return self.print()
    
    def createChild(self, assemblerType: Type[AssemblyT], *args) -> AssemblyT:
        ret = assemblerType(*args, parentAST=self)
        # Set the parent of all the args to be this new instruction.
        for arg in args:
            if isinstance(arg, AssemblyAST):  arg.parent = ret
        return ret

    def fromTACValue(self, tacValue: TACValue, offset: int = 0) -> AssemblerOperand:
        if tacValue.isConstant:
            # First, check if the constant is defined inside the AssemblerStaticConstant.CONSTANTS 
            # array.
            found: AssemblerStaticConstant|None = None
            for cnst in AssemblerStaticConstant.CONSTANTS:
                if cnst.identifier == tacValue.constantValue:
                    found = cnst
                    break

            if found is not None:
                return Data(AssemblyType.fromTAC(tacValue.valueType), found.identifier, 0, self)

            # In x64, decimal numbers are stored into memory and cannot be used as immediate.
            if tacValue.valueType.isDecimal():
                doubleConstant = AssemblerStaticConstant.newSimpleConstant(
                    tacValue.valueType, tacValue.print())
                return Data(AssemblyType.fromTAC(tacValue.valueType), doubleConstant.identifier, 0, self)
            
            return Immediate(tacValue, self)
        else:
            if isinstance(tacValue.valueType, ArrayDeclaratorType) or \
               (isinstance(tacValue.valueType, BaseDeclaratorType) and tacValue.valueType.baseType.name in ("STRUCT", "UNION")) or \
               offset != 0:
                return PseudoMemory(AssemblyType.fromTAC(tacValue.valueType), tacValue.print(), offset, self)
            
            return Pseudo(tacValue, self)
        
    def copyBytes(self, src: AssemblerOperand, dst: AssemblerOperand, asmbType: AssemblyType) -> list[AssemblerInstruction]:
        # Basic MOV instruction.
        if not isinstance(src, (Memory, PseudoMemory)) or not isinstance(dst, (Memory, PseudoMemory)):
            return [MOV(asmbType, src, dst)]

        # Multiple MOV instructions.
        ret: list[AssemblerInstruction] = []

        def applyOffset(origin: Memory|PseudoMemory, offset: int):
            copyObject = origin.createCopy()

            if isinstance(copyObject, Memory):
                copyObject.offset += offset
            elif isinstance(copyObject, PseudoMemory):
                copyObject.offset += offset
            else:
                raise ValueError(f"{origin} is not a Memory or PseudoMemory")

            return copyObject

        # Move it in chunks of 8, 4, 2 or 1.
        byteCount: int = asmbType.size
        offset: int = 0
        while offset < byteCount:
            pendingBytes = byteCount - offset
            if pendingBytes >= 8:
                subMOVAsmbType = AssemblyType.QUADWORD
            elif pendingBytes >= 4:
                subMOVAsmbType = AssemblyType.LONGWORD
            elif pendingBytes >= 2:
                subMOVAsmbType = AssemblyType.WORD
            else:
                subMOVAsmbType = AssemblyType.BYTE

            offsetedSrc = applyOffset(src, offset)
            offsetedDst = applyOffset(dst, offset)

            ret.append(MOV(subMOVAsmbType, offsetedSrc, offsetedDst))

            offset += subMOVAsmbType.size

        return ret
    
    """
    Used to transfer bytes to a register when the number of bytes is not standard.
    """
    def copyBytesToRegister(self, src: AssemblerOperand, reg: REG, byteCount: int) -> list[AssemblerInstruction]:
        if not isinstance(src, PseudoMemory):
            raise ValueError("Expected a PseudoMemory in copyBytesToRegister")
        
        instList: list[AssemblerInstruction] = []
        offset: int = byteCount - 1

        while offset >= 0:
            srcOp = src.createCopy()
            srcOp.offset = src.offset + offset

            # Move to the LSB of the register.
            instList.append(MOV(AssemblyType.BYTE, srcOp, Register(AssemblyType.BYTE, reg)))
            # Shift it to the left (except for the last byte of src).
            if offset > 0:
                instList.append(
                    BINARY(AssemblyType.QUADWORD, 
                           BinaryOperator.LOGIC_LEFT_SHIFT,
                           self.fromTACValue(TACValue(True, TypeSpecifier.UCHAR.toBaseType(), "8")),
                           Register(AssemblyType.QUADWORD, reg)))
            offset -= 1

        return instList

    """
    Used to transfer bytes from a register when the number of bytes is not standard.
    """
    def copyBytesFromRegister(self, reg: REG, dst: AssemblerOperand, byteCount: int) -> list[AssemblerInstruction]:
        if not isinstance(dst, PseudoMemory):
            raise ValueError("Expected a PseudoMemory in copyBytesToRegister")
        
        instList: list[AssemblerInstruction] = []
        offset: int = 0

        while offset < byteCount:
            dstCopy = dst.createCopy()
            dstCopy.offset = dst.offset + offset

            # Move to the LSB of the register.
            instList.append(MOV(AssemblyType.BYTE, Register(AssemblyType.BYTE, reg), dstCopy))
            # Shift it to the left (except for the last byte of src).
            if offset < byteCount - 1:
                instList.append(
                    BINARY(AssemblyType.QUADWORD, 
                           BinaryOperator.LOGIC_RIGHT_SHIFT, 
                           self.fromTACValue(TACValue(True, TypeSpecifier.UCHAR.toBaseType(), "8")),
                           Register(AssemblyType.QUADWORD, reg)))
            offset += 1

        return instList

class AssemblerProgram(AssemblyAST):
    def __init__(self, program: TACProgram, parentAST: AssemblyAST | None = None) -> None:
        self.program = program
        super().__init__(parentAST)
        self.secondPass()
        self.thirdPass()

    def firstPass(self):
        self.programDefs: list[AssemblyAST] = []

        for topLevel in self.program.topLevel:
            match topLevel:
                case TACConstantVariable():
                    AssemblerStaticConstant.newComplexConstant(
                        topLevel.valueType, topLevel.identifier, topLevel.initialization)
                
                case TACStaticVariable():
                    topLevelAssembly = self.createChild(
                        AssemblerStaticVariable, 
                        AssemblyType.fromTAC(topLevel.valueType),
                        topLevel.isGlobal,
                        topLevel.identifier,
                        topLevel.initialization
                    )
                    self.programDefs.append(topLevelAssembly)

                case TACFunction():
                    topLevelAssembly = self.createChild(AssemblerFunction, topLevel)
                    self.programDefs.append(topLevelAssembly)

                case _:
                    raise ValueError(f"Invalid type {topLevel} in top level instructions")

    def secondPass(self):
        pass

    def thirdPass(self):
        pass

    def emitCode(self) -> str:
        ret = ""
        for func in self.programDefs:
            ret += func.emitCode() + "\n"

        # Emit the constant section.
        ret += "\t.section\t.rodata\n"
        for constant in AssemblerStaticConstant.CONSTANTS:
            ret += constant.emitCode() + "\n"
        ret += '\t.section .note.GNU-stack,"",@progbits\n'
        return ret

    def print(self) -> str:
        ret = ""
        for func in self.programDefs:
            ret += func.print() + "\n"
        return ret

class AssemblerStaticVariable(AssemblyAST):
    def __init__(self, assemblyType: AssemblyType, isGlobal: bool, 
                 identifier: str, initialization: list[Constant],
                 parentAST: AssemblyAST | None = None) -> None:
        self.assemblyType = assemblyType
        self.isGlobal = isGlobal
        self.identifier = identifier
        self.initialization = initialization
        super().__init__(parentAST)

    def firstPass(self):
        pass

    def secondPass(self):
        pass

    def thirdPass(self):
        pass

    def emitCode(self) -> str:
        ret = ""
        if self.isGlobal:
            ret =  f"\t.globl {self.identifier}\n"

        if len(self.initialization) == 0:
            ret +=  "\t.bss\n"
            ret += f"\t.align {self.assemblyType.alignment}\n"
            ret += f"{self.identifier}:\n"
            ret += f"\t.zero {self.assemblyType.size}\n"
        else:
            ret +=  "\t.data\n"
            ret += f"\t.align {self.assemblyType.alignment}\n"
            ret += f"{self.identifier}:\n"
            
            for const in self.initialization:
                if isinstance(const, ZeroPaddingInitializer):
                    ret += f"\t.zero {const.byteCount}\n"
                elif isinstance(const, PointerInitializer):
                    asmbType = AssemblyType.fromTAC(const.typeId)
                    if const.offset > 0:
                        ret += f"\t.{asmbType.getDataSectionName()} {const.constValue}+{const.offset}\n"
                    elif const.offset == 0:
                        ret += f"\t.{asmbType.getDataSectionName()} {const.constValue}\n"
                    else:
                        ret += f"\t.{asmbType.getDataSectionName()} {const.constValue}{const.offset}\n"
                else:
                    asmbType = AssemblyType.fromTAC(const.typeId)
                    ret += f"\t.{asmbType.getDataSectionName()} {const.constValue}\n"
        return ret
    
    def print(self) -> str:
        ret = f"--- {self.identifier} ---\n"
        for init in self.initialization:
            ret += init.print(0)
        return ret    
    
class AssemblerStaticConstant(AssemblyAST):
    CONSTANTS: list[AssemblerStaticConstant] = []

    # Utility to generate constants during the assembly stage.
    @staticmethod
    def newSimpleConstant(valueType: DeclaratorType, initialization: str, alignment: int|None = None) -> AssemblerStaticConstant:
        for prevConst in AssemblerStaticConstant.CONSTANTS:
            if prevConst.valueType == valueType and prevConst.initialization == initialization:
                return prevConst
        
        # Start with .L as it is hidden.
        identifier: str = f".Lconst{len(AssemblerStaticConstant.CONSTANTS)}"
        asmbType = AssemblyType.fromTAC(valueType)
        # By default the alignment is calculated from the type.
        if alignment is None:
            alignment = asmbType.alignment

        return AssemblerStaticConstant(valueType, identifier, [(asmbType.getDataSectionName(), initialization)], alignment)

    # Utility to generate assembly code for C constants.
    @staticmethod
    def newComplexConstant(valueType: DeclaratorType, identifier: str, 
                           initialization: list[Constant], alignment: int|None = None) -> AssemblerStaticConstant:
        for prevConst in AssemblerStaticConstant.CONSTANTS:
            # TODO: Fix the prevConst.initialization == initialization.
            if prevConst.valueType == valueType and prevConst.initialization == initialization:
                return prevConst
        
        # By default the alignment is calculated from the type.
        if alignment is None:
            alignment = AssemblyType.fromTAC(valueType).alignment

        resultInitializationList: list[tuple[str, str]] = []
        for const in initialization:
            resultInitializationList.append(
                (AssemblyType.fromTAC(const.typeId).getDataSectionName(), const.constValue)
            )

        return AssemblerStaticConstant(valueType, identifier, resultInitializationList, alignment)

    # "initialization" is a list of (data section name, value)
    def __init__(self, valueType: DeclaratorType, identifier: str, initialization: list[tuple[str, str]], 
                 alignment: int, parentAST: AssemblyAST | None = None) -> None:
        self.valueType = valueType
        self.identifier = identifier
        self.initialization = initialization
        self.alignment = alignment
    
        AssemblerStaticConstant.CONSTANTS.append(self)
        super().__init__(parentAST)

    def firstPass(self):
        pass

    def secondPass(self):
        pass

    def thirdPass(self):
        pass

    # Converts byte objects grouped together into ascii or asciz  
    def preprocessInitializations(self):
        newInit: list[tuple[str, str]] = []
        byteGrouping: str = ""

        def unescapeString(string: str) -> str:
            unesc = repr(string)
            unquoted = unesc[1:-1]
            if unesc[0] == '"':
                # Surrounded with "".
                unquoted = unquoted.replace("'", "\\'")
            else:
                # Surrounded with ''.
                unquoted = unquoted.replace('"', '\\"')
            return f'"{unquoted}"'

        def addAsciiInitializer():
            nonlocal byteGrouping
            if byteGrouping == "":
                return

            if byteGrouping[-1] == "\0":
                # Add an asciz instruction removing the null character.
                newInit.append(("asciz", unescapeString(byteGrouping[:-1])))
            else:
                # Add an ascii instruction.
                newInit.append(("ascii", unescapeString(byteGrouping)))
            # Restart the variable.
            byteGrouping = ""

        for dataSection, value in self.initialization:
            if dataSection == "byte":
                intVal = int(value)
                if 0 <= intVal < 128:
                    if byteGrouping != "" and byteGrouping[-1] == '\0':
                        # The previous string is null terminated.
                        addAsciiInitializer()
                    byteGrouping += chr(intVal)
                else:
                    addAsciiInitializer()
                    # Add it directly as a "byte".
                    newInit.append((dataSection, value))
            else:
                addAsciiInitializer()
                # Add the same initializer to the list.
                newInit.append((dataSection, value))

        addAsciiInitializer()
        return newInit

    def emitCode(self) -> str:
        ret  = f"\t.align {self.alignment}\n"
        ret += f"{self.identifier}:\n"
        for dataSection, value in self.preprocessInitializations():
            ret += f"\t.{dataSection} {value}\n"
        return ret

    def print(self) -> str:
        return f"StaticConstant({self.initialization})\n"

class AssemblerFunction(AssemblyAST):
    def __init__(self, function: TACFunction, parentAST: AssemblyAST | None = None) -> None:
        self.function = function
        self.instructions: list[AssemblerInstruction] = []
        super().__init__(parentAST)
        self.secondPass()
        self.thirdPass()

    def createInst(self, assemblerType: Type[AssemblerInstruction], *args) -> AssemblerInstruction:
        ret = assemblerType(*args, parentAST=self)
        self.instructions.append(ret)

        # Set the parent of all the args to be this new instruction.
        for arg in args:
            if isinstance(arg, AssemblyAST):  arg.parent = ret

        return ret

    """
    Returns:
    - List of arguments stored in the general registers.
    - List of arguments stored in the SSE registers.
    - List of arguments stored in the stack.

    Each argument consists of its origical value (AssemblerOperand) and the type to use to transfer 
    data (AssemblyType). This is needed for structs.
    """
    def classifyArguments(self, arguments: list[TACValue], returnStoredInStack: bool) -> \
        tuple[list[tuple[AssemblerOperand, AssemblyType]], list[tuple[AssemblerOperand, AssemblyType]], list[tuple[AssemblerOperand, AssemblyType]]]:
        
        intRegisterArgs: list[tuple[AssemblerOperand, AssemblyType]] = []
        doubleRegisterArgs: list[tuple[AssemblerOperand, AssemblyType]] = []
        stackArgs: list[tuple[AssemblerOperand, AssemblyType]] = []

        intRegsAvailable: int = 5 if returnStoredInStack else 6
        doubleRegsAvailable: int = 8

        for arg in arguments:
            destinationValue = self.fromTACValue(arg)
            argType = destinationValue.assemblyType

            if argType.isDecimal():
                if doubleRegsAvailable > 0:
                    doubleRegisterArgs.append((destinationValue, argType))
                    doubleRegsAvailable -= 1
                else:
                    stackArgs.append((destinationValue, argType))
            elif argType.isScalar():
                if intRegsAvailable > 0:
                    intRegisterArgs.append((destinationValue, argType))
                    intRegsAvailable -= 1
                else:
                    stackArgs.append((destinationValue, argType))
            else:
                # This is a structure argument.
                useStack: bool = True

                # If the first member is not memory, then the structure will use a mix of registers.
                # If it is, all following members will also be memory, and so the struct will go 
                # into the stack.
                if argType.members[0].classType != AssemblyClassType.MEMORY:
                    tentativeInts: list[tuple[AssemblerOperand, AssemblyType]] = []
                    tentativeDoubles: list[tuple[AssemblerOperand, AssemblyType]] = []
                    offset = 0
                    for member in argType.members:
                        # Create an offsetted operand with the original type of the member.
                        asmbMember = PseudoMemory(argType, arg.vbeName, offset, self)
                        offset += member.byteSize

                        # Now, this operand needs to be "moved" using the following type.
                        memberAsmbType = member.toAssemblyType()
                        if member.classType == AssemblyClassType.INTEGER:
                            tentativeInts.append((asmbMember, memberAsmbType))
                        elif member.classType == AssemblyClassType.SSE:
                            tentativeDoubles.append((asmbMember, memberAsmbType))
                        else:
                            raise ValueError()

                    # If there are enough free registers, add them to the register arrays.
                    if len(tentativeInts) <= intRegsAvailable and len(tentativeDoubles) <= doubleRegsAvailable:
                        intRegisterArgs.extend(tentativeInts)
                        doubleRegisterArgs.extend(tentativeDoubles)
                        intRegsAvailable -= len(tentativeInts)
                        doubleRegsAvailable -= len(tentativeDoubles)
                        # Don't use the stack.
                        useStack = False
                
                if useStack:
                    offset = 0
                    for member in argType.members:
                        asmbMember = PseudoMemory(argType, arg.vbeName, offset, self)
                        offset += member.byteSize
                        stackArgs.append((asmbMember, member.toAssemblyType()))

        return (intRegisterArgs, doubleRegisterArgs, stackArgs)
        
    """
    Returns:
    - List of arguments stored in the general registers.
    - List of arguments stored in the SSE registers.
    - True if the return value is returned as a variable in the stack.
    """
    def classifyReturnValue(self, retValue: TACValue) -> \
        tuple[list[tuple[AssemblerOperand, AssemblyType]], list[tuple[AssemblerOperand, AssemblyType]], bool]:
        asmbType = AssemblyType.fromTAC(retValue.valueType)

        if asmbType.isDecimal():
            retAsmbVal = self.fromTACValue(retValue)
            return ([], [(retAsmbVal, retAsmbVal.assemblyType)], False)
        
        if asmbType.isScalar():
            retAsmbVal = self.fromTACValue(retValue)
            return ([(retAsmbVal, retAsmbVal.assemblyType)], [], False)

        if asmbType.members[0].classType == AssemblyClassType.MEMORY:
            # The return is stored in the stack.
            return ([], [], True)
        
        # The return is stored in registers.
        intRegisterArgs: list[tuple[AssemblerOperand, AssemblyType]] = []
        doubleRegisterArgs: list[tuple[AssemblerOperand, AssemblyType]] = []
        offset: int = 0
        for member in asmbType.members:
            asmbMember = PseudoMemory(asmbType, retValue.vbeName, offset, self)
            offset += member.byteSize

            if member.classType == AssemblyClassType.SSE:
                doubleRegisterArgs.append((asmbMember, member.toAssemblyType()))
            elif member.classType == AssemblyClassType.INTEGER:
                intRegisterArgs.append((asmbMember, member.toAssemblyType()))
            else:
                raise ValueError()

        return (intRegisterArgs, doubleRegisterArgs, False)

    def firstPass(self):
        self.identifier: str = self.function.identifier

        # Is the return value passed from the stack?
        returnInStack = self.function.funDecl.typeId.returnDeclarator != TypeSpecifier.VOID.toBaseType() and \
                        AssemblyType.fromTAC(self.function.funDecl.typeId.returnDeclarator). \
                        members[0].classType == AssemblyClassType.MEMORY

        # Restart the stack to calculate the right offset.
        # If the return value is stored in the stack, its address will be the first stack variable.
        # TODO: For 64 bit systems, this is an 8 byte variable.
        Memory.restartStackVariables(-8 if returnInStack else 0)

        # Create the variables used as arguments. These are stored in registers or in the stack.
        tacArgs = [TACValue(False, arg.type, arg.name) for arg in self.function.arguments]
        intRegArgs, doubleRegArgs, stackInputArgs = self.classifyArguments(tacArgs, returnInStack)

        if returnInStack:
            # Store the address of the return value, which is stored in DI to the stack.
            self.createInst(MOV, 
                            AssemblyType.QUADWORD,
                            Register(AssemblyType.QUADWORD, REG.DI), 
                            Memory(AssemblyType.QUADWORD, REG.BP, -8))

        # The order of arguments is: DI, SI, DX, CX, R8, R9 and then stack (pushed in reversed order).
        # In parallel, double arguments must be pushed to XMM0 to XMM7 and then to the stack.
        # Do not use DI if the return value is stored in the stack.
        INT_REG_ORDER = [REG.DI, REG.SI, REG.DX, REG.CX, REG.R8, REG.R9]
        for (value, movAsmbType), reg in zip(intRegArgs, INT_REG_ORDER[(1 if returnInStack else 0):]):
            if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                self.instructions.extend(self.copyBytesFromRegister(reg, value, movAsmbType.size))
            else:
                self.createInst(MOV, 
                                movAsmbType, 
                                Register(value.assemblyType, reg), 
                                value)

        DOUBLE_REG_ORDER = [REG.XMM0, REG.XMM1, REG.XMM2, REG.XMM3, REG.XMM4, REG.XMM5, REG.XMM6, REG.XMM7]
        for (value, movAsmbType), reg in zip(doubleRegArgs, DOUBLE_REG_ORDER):
            self.createInst(MOV, 
                            movAsmbType, 
                            Register(value.assemblyType, reg), 
                            value)

        # The arguments in the stack start at Stack(16). From then on, add in groups of eight.
        stackOffset = 16
        for (value, movAsmbType) in stackInputArgs:
            if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                self.instructions.extend(
                    self.copyBytes(Memory(AssemblyType.QUADWORD, REG.BP, stackOffset), value, movAsmbType)
                )
            else:
                # Move to the stack.
                self.createInst(MOV, 
                                movAsmbType, 
                                Memory(value.assemblyType, REG.BP, stackOffset), 
                                value)
            # Increment the stack offset.
            stackOffset += 8

        # Convert the function's TAC instructions into assembler instructions.
        for inst in self.function.instructions:
            # Add a comment between instructions to know what each block of assembler instructions 
            # is doing. Skip labels.
            if not isinstance(inst, TACLabel):
                self.createInst(COMMENT, inst.print())

            # TODO: Temporary
            if globalContext.isBuiltInTACFunction(inst):
                continue

            match inst:
                case TACReturn():
                    if inst.result.valueType != TypeSpecifier.VOID.toBaseType():
                        intRegArgs, doubleRegArgs, returnInStack = self.classifyReturnValue(inst.result)
                        if returnInStack:
                            # Get the address of where the return value should be stored.
                            self.createInst(MOV, 
                                            AssemblyType.QUADWORD,
                                            Memory(AssemblyType.QUADWORD, REG.BP, -8), 
                                            Register(AssemblyType.QUADWORD, REG.AX))
                            # Transfer the return value to this address.
                            transferInsts = self.copyBytes(self.fromTACValue(inst.result), 
                                                          Memory(AssemblyType.QUADWORD, REG.AX, 0), 
                                                          AssemblyType.fromTAC(inst.result.valueType))
                            self.instructions.extend(transferInsts)
                        else:
                            INT_RETURN_REGS = [REG.AX, REG.DX]
                            DOUBLE_RETURN_REGS = [REG.XMM0, REG.XMM1]

                            for (value, movAsmbType), reg in zip(intRegArgs, INT_RETURN_REGS):
                                if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                                    self.instructions.extend(self.copyBytesToRegister(value, reg, movAsmbType.size))
                                else:
                                    self.createInst(MOV, 
                                                    movAsmbType, 
                                                    value, 
                                                    Register(value.assemblyType, reg))

                            for (value, movAsmbType), reg in zip(doubleRegArgs, DOUBLE_RETURN_REGS):
                                self.createInst(MOV, 
                                                movAsmbType, 
                                                value, 
                                                Register(value.assemblyType, reg))

                    self.createInst(RET)

                case TACUnary():
                    exp: AssemblerOperand = self.fromTACValue(inst.exp)
                    dest: AssemblerOperand = self.fromTACValue(inst.result)
                    match inst.operator:
                        case UnaryOperator.NOT:
                            # !(x) is the same as x == 0.
                            self.createInst(CMP, 
                                exp.assemblyType,
                                self.fromTACValue(TACValue(True, inst.exp.valueType, "0")), 
                                exp)
                            # Fill the destination with a zero. This does not affect the flags.
                            self.createInst(MOV,
                                            dest.assemblyType, 
                                            self.fromTACValue(TACValue(True, inst.result.valueType, "0")), 
                                            dest)

                            if inst.exp.valueType.isDecimal():
                                # A NaN != 0, so !x when x is Nan should return 0.
                                jumpOverSet = LABEL.generateLabel("nanDetected")
                                self.createInst(JMPIF, ConditionCode.EVEN_PARITY, jumpOverSet)
                                self.createInst(SETIF, ConditionCode.EQUAL, dest)
                                self.createInst(LABEL, jumpOverSet)
                            else:
                                self.createInst(SETIF, ConditionCode.EQUAL, dest)

                        case UnaryOperator.NEGATION:
                            if inst.result.valueType.isDecimal():
                                # To negate a double number, XOR with -0.0. XOR is a vector operation
                                # when working with double numbers.
                                if inst.result.valueType == TypeSpecifier.DOUBLE.toBaseType():
                                    negZero = AssemblerStaticConstant.newSimpleConstant(
                                        TypeSpecifier.DOUBLE.toBaseType(), "-0.0", 16)
                                    negZeroData = Data(AssemblyType.DOUBLE, negZero.identifier, 0, self)
                                else:
                                    negZero = AssemblerStaticConstant.newSimpleConstant(
                                        TypeSpecifier.FLOAT.toBaseType(), "-0.0", 16)
                                    negZeroData = Data(AssemblyType.FLOAT, negZero.identifier, 0, self)

                                self.createInst(MOV, exp.assemblyType, exp, dest)
                                self.createInst(BINARY, 
                                                dest.assemblyType, BinaryOperator.BITWISE_XOR, 
                                                negZeroData, 
                                                dest)
                            else:
                                self.createInst(MOV, exp.assemblyType, exp, dest)
                                self.createInst(UNARY, exp.assemblyType, inst.operator, dest)

                        case UnaryOperator.INCREMENT:
                            if inst.result.valueType.isDecimal():
                                self.createInst(MOV, exp.assemblyType, exp, dest)
                                self.createInst(BINARY, 
                                                dest.assemblyType, BinaryOperator.SUM,
                                                self.fromTACValue(TACValue(True, inst.exp.valueType, "1")), 
                                                dest)
                            else:
                                self.createInst(MOV, exp.assemblyType, exp, dest)
                                self.createInst(UNARY, exp.assemblyType, inst.operator, dest)

                        case UnaryOperator.DECREMENT:
                            if inst.result.valueType.isDecimal():
                                self.createInst(MOV, exp.assemblyType, exp, dest)
                                self.createInst(BINARY, 
                                                exp.assemblyType, BinaryOperator.SUBTRACT,
                                                self.fromTACValue(TACValue(True, inst.exp.valueType, "1")), 
                                                dest)
                            else:
                                self.createInst(MOV, exp.assemblyType, exp, dest)
                                self.createInst(UNARY, exp.assemblyType, inst.operator, dest)

                        case _:
                            self.createInst(MOV, exp.assemblyType, exp, dest)
                            self.createInst(UNARY, exp.assemblyType, inst.operator, dest)

                case TACBinary():
                    exp1: AssemblerOperand = self.fromTACValue(inst.exp1)
                    exp2: AssemblerOperand = self.fromTACValue(inst.exp2)
                    dest: AssemblerOperand = self.fromTACValue(inst.result)
                    match inst.operator:
                        case BinaryOperator.DIVISION | BinaryOperator.MODULUS:
                            if inst.result.valueType.isDecimal():
                                # Double division works as with the rest of binary operations.
                                self.createInst(MOV, exp1.assemblyType, exp1, dest)
                                self.createInst(BINARY, exp1.assemblyType, inst.operator, exp2, dest)
                            else:
                                # Division in x64 is different from the other arithmetical operations.
                                self.createInst(MOV, exp1.assemblyType, exp1, Register(exp1.assemblyType, REG.AX))
                                if inst.result.valueType.isSigned():
                                    # Sign extend AX into DX.
                                    self.createInst(CDQ, exp1.assemblyType)
                                    self.createInst(IDIV, exp1.assemblyType, inst.operator, exp2)
                                else:
                                    # Set DX to zero.
                                    self.createInst(MOV, 
                                        exp1.assemblyType,
                                        self.fromTACValue(TACValue(True, inst.result.valueType, "0")), 
                                        Register(exp1.assemblyType, REG.DX))
                                    self.createInst(DIV, exp1.assemblyType, inst.operator, exp2)

                                if inst.operator == BinaryOperator.DIVISION:
                                    self.createInst(MOV, 
                                                    exp1.assemblyType, 
                                                    Register(exp1.assemblyType, REG.AX), 
                                                    dest)
                                else:
                                    self.createInst(MOV, 
                                                    exp1.assemblyType, 
                                                    Register(exp1.assemblyType, REG.DX), 
                                                    dest)
                                
                        case BinaryOperator.GREATER_THAN | BinaryOperator.GREATER_OR_EQUAL | \
                             BinaryOperator.LESS_THAN    | BinaryOperator.LESS_OR_EQUAL    | \
                             BinaryOperator.EQUAL        | BinaryOperator.NOT_EQUAL:
                            self.createInst(CMP, exp1.assemblyType, exp2, exp1)
                            # Fill the destination with a 32-bit zero. This does not affect the flags.
                            self.createInst(MOV, 
                                            dest.assemblyType, 
                                            self.fromTACValue(TACValue(True, inst.result.valueType, "0")), 
                                            dest)
                            if inst.exp2.valueType.isDecimal() and \
                               inst.operator in (BinaryOperator.EQUAL, BinaryOperator.NOT_EQUAL, BinaryOperator.LESS_THAN, BinaryOperator.LESS_OR_EQUAL):
                                # To account for NaN, as they are unordered, they should always return "0"
                                # when comparing them. 
                                # When there's a NaN, the ZF = CF = PF = 1 by the CMP instruction. 

                                if inst.operator == BinaryOperator.NOT_EQUAL:
                                    # For != if a NaN is detected, it should return one.
                                    jumpOverSet = LABEL.generateLabel("nanDetected")
                                    nanDetectedExit = LABEL.generateLabel("nanDetectedExit")
                                    self.createInst(JMPIF, ConditionCode.EVEN_PARITY, jumpOverSet)
                                    self.createInst(SETIF, 
                                        ConditionCode.fromBinaryOperator(inst.operator, inst.exp1.valueType), 
                                        dest)
                                    self.createInst(JMP, nanDetectedExit)
                                    self.createInst(LABEL, jumpOverSet)
                                    # As ZF = CF = 1, this is the same as below or equal.
                                    self.createInst(SETIF, 
                                        ConditionCode.BELOW_EQUAL, 
                                        dest)
                                    self.createInst(LABEL, nanDetectedExit)
                                else:
                                    # For ==, < and <= if a NaN is detected, it should return zero.
                                    jumpOverSet = LABEL.generateLabel("nanDetected")
                                    self.createInst(JMPIF, ConditionCode.EVEN_PARITY, jumpOverSet)
                                    self.createInst(SETIF, 
                                        ConditionCode.fromBinaryOperator(inst.operator, inst.exp1.valueType), 
                                        dest)
                                    self.createInst(LABEL, jumpOverSet)
                            else:
                                self.createInst(SETIF, 
                                    ConditionCode.fromBinaryOperator(inst.operator, inst.exp1.valueType), 
                                    dest)

                        case _:
                            self.createInst(MOV, exp1.assemblyType, exp1, dest)
                            self.createInst(BINARY, exp1.assemblyType, inst.operator, exp2, dest)
                
                case TACJump():
                    self.createInst(JMP, inst.target)

                case TACJumpIfValue():
                    self.createInst(CMP, 
                                    AssemblyType.fromTAC(inst.condition.valueType),
                                    self.fromTACValue(inst.value), 
                                    self.fromTACValue(inst.condition))
                    if inst.condition.valueType.isDecimal():
                        # To account for NaN, as they are unordered, they should always return "0"
                        # when comparing them. NaN are detected with the Parity Flag and the CMP 
                        # instruction.
                        jumpOverJump = LABEL.generateLabel("nanDetected")
                        self.createInst(JMPIF, ConditionCode.EVEN_PARITY, jumpOverJump)
                        self.createInst(JMPIF, ConditionCode.EQUAL, inst.target)
                        self.createInst(LABEL, jumpOverJump)
                    else:
                        self.createInst(JMPIF, ConditionCode.EQUAL, inst.target)

                case TACJumpIfZero():
                    self.createInst(CMP, 
                                    AssemblyType.fromTAC(inst.condition.valueType),
                                    self.fromTACValue(TACValue(True, inst.condition.valueType, "0")), 
                                    self.fromTACValue(inst.condition))

                    if inst.condition.valueType.isDecimal():
                        # To account for NaN, as they are unordered, they should always return "0"
                        # when comparing them. NaN are detected with the Parity Flag and the CMP 
                        # instruction.
                        jumpOverJump = LABEL.generateLabel("nanDetected")
                        self.createInst(JMPIF, ConditionCode.EVEN_PARITY, jumpOverJump)
                        self.createInst(JMPIF, ConditionCode.EQUAL, inst.target)
                        self.createInst(LABEL, jumpOverJump)
                    else:
                        self.createInst(JMPIF, ConditionCode.EQUAL, inst.target)

                case TACJumpIfNotZero():
                    self.createInst(CMP, 
                                    AssemblyType.fromTAC(inst.condition.valueType),
                                    self.fromTACValue(TACValue(True, inst.condition.valueType, "0")), 
                                    self.fromTACValue(inst.condition))
                    
                    if inst.condition.valueType.isDecimal():
                        # If the number is Nan that's definitely not zero. Take the jump.
                        self.createInst(JMPIF, ConditionCode.EVEN_PARITY, inst.target)

                    self.createInst(JMPIF, ConditionCode.NOT_EQUAL, inst.target)

                case TACCopy():
                    src = self.fromTACValue(inst.src)
                    dst = self.fromTACValue(inst.dst)
                    movInstructions = self.copyBytes(src, dst, AssemblyType.fromTAC(inst.src.valueType))
                    self.instructions.extend(movInstructions)

                case TACLoad():
                    self.createInst(MOV, 
                                    AssemblyType.QUADWORD, 
                                    self.fromTACValue(inst.src),
                                    Register(AssemblyType.QUADWORD, REG.AX))

                    dst = self.fromTACValue(inst.dst)
                    src = Memory(dst.assemblyType, REG.AX, 0)
                    movInstructions = self.copyBytes(src, dst, dst.assemblyType)
                    self.instructions.extend(movInstructions)

                case TACStore():
                    self.createInst(MOV, 
                                    AssemblyType.QUADWORD,
                                    self.fromTACValue(inst.dst), 
                                    Register(AssemblyType.QUADWORD, REG.AX))
                    
                    src = self.fromTACValue(inst.src)
                    dst = Memory(src.assemblyType, REG.AX, 0)
                    movInstructions = self.copyBytes(src, dst, src.assemblyType)
                    self.instructions.extend(movInstructions)

                case TACGetAddress():
                    self.createInst(LEA, self.fromTACValue(inst.src), self.fromTACValue(inst.dst))

                case TACCopyToOffset():
                    src = self.fromTACValue(inst.src)
                    dst = self.fromTACValue(inst.dst, inst.byteOffset)
                    movInstructions = self.copyBytes(src, dst, src.assemblyType)
                    self.instructions.extend(movInstructions)

                case TACCopyFromOffset():
                    src = self.fromTACValue(inst.src, inst.byteOffset)
                    dst = self.fromTACValue(inst.dst)
                    movInstructions = self.copyBytes(src, dst, dst.assemblyType)
                    self.instructions.extend(movInstructions)

                case TACAddToPointer():
                    # Both pointer and index should be qwords if running for a 64 bit system.
                    pointer = self.fromTACValue(inst.pointer)
                    index = self.fromTACValue(inst.index)
                    
                    self.createInst(MOV, 
                                    AssemblyType.QUADWORD, 
                                    pointer, 
                                    Register(AssemblyType.QUADWORD, REG.AX))

                    if isinstance(index, Immediate):
                        # The index is a constant, and so is the scale. Calculate it during compilation.
                        byteOffset = int(index.value.constantValue) * inst.scale
                        self.createInst(LEA, 
                                        Memory(AssemblyType.QUADWORD, REG.AX, byteOffset), 
                                        self.fromTACValue(inst.dst))
                    else:
                        self.createInst(MOV, 
                                        index.assemblyType, 
                                        index, 
                                        Register(index.assemblyType, REG.DX))

                        if inst.scale in (1, 2, 4, 8):
                            self.createInst(LEA, 
                                Indexed(
                                    pointer.assemblyType, 
                                    Register(AssemblyType.QUADWORD, REG.AX), 
                                    Register(index.assemblyType, REG.DX), 
                                    inst.scale
                                ), self.fromTACValue(inst.dst))
                        else:
                            # TODO: for 64 bit systems, TypeSpecifier.LONG.toBaseType().
                            self.createInst(BINARY, 
                                index.assemblyType, BinaryOperator.MULTIPLICATION, 
                                self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), str(inst.scale))),
                                Register(index.assemblyType, REG.DX))
                            self.createInst(LEA, 
                                            Indexed(
                                                pointer.assemblyType, 
                                                Register(AssemblyType.QUADWORD, REG.AX), 
                                                Register(index.assemblyType, REG.DX), 
                                                1
                                            ), self.fromTACValue(inst.dst))
                    
                case TACLabel():
                    self.createInst(LABEL, inst.identifier)

                case TACFunctionCall():
                    returnIntRegs: list[tuple[AssemblerOperand, AssemblyType]] = []
                    returnDoubleRegs: list[tuple[AssemblerOperand, AssemblyType]] = []
                    returnInStack: bool = False

                    # Classify the return value.
                    if inst.returnType != TypeSpecifier.VOID.toBaseType():
                        returnIntRegs, returnDoubleRegs, returnInStack = self.classifyReturnValue(inst.result)

                    if returnInStack:
                        # When the value is returned in the stack, the return value's space is 
                        # reserved on the caller. It's address is stored in DI.
                        retAsmbVal = self.fromTACValue(inst.result)
                        # TODO: This is only for 64-bit...
                        self.createInst(LEA, retAsmbVal, Register(AssemblyType.QUADWORD, REG.DI))

                    # Split between arguments stored in registers and arguments stored in the stack.
                    intRegisterArgs, doubleRegisterArgs, stackArgs = self.classifyArguments(inst.arguments, returnInStack)

                    # The stack needs to be padded so that the function arguments start from a 
                    # multiple of 16. Each cell in the stack is 8 bytes long.
                    if len(stackArgs) % 2 == 0:
                        stackPadding = 0
                    else:
                        # Pad so that it starts from a multiple of 16.
                        stackPadding = 8

                    if stackPadding != 0:
                        # ALLOCATE_STACK: stackPadding
                        self.createInst(BINARY, 
                            AssemblyType.QUADWORD, BinaryOperator.SUBTRACT, 
                            self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), str(stackPadding))),
                            Register(AssemblyType.QUADWORD, REG.SP))

                    # Pass the function's arguments to registers and then the stack.
                    # The order of arguments is: DI, SI, DX, CX, R8, R9 and then stack (pushed in 
                    # reversed order).
                    # Skip DI if the return value is saved in the stack, it contains the address of 
                    # the return value.
                    for (value, movAsmbType), reg in zip(intRegisterArgs, INT_REG_ORDER[(1 if returnInStack else 0):]):
                        if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                            # There may be part of a struct/union returned in a register whose byte 
                            # size is not standard, i.e. 3, 5, 6 or 7 bytes.
                            self.instructions.extend(self.copyBytesToRegister(value, reg, movAsmbType.size))
                        else:
                            self.instructions.extend(
                                self.copyBytes(value, Register(value.assemblyType, reg), movAsmbType)
                            )

                    # Now with the double registers.
                    for (value, movAsmbType), reg in zip(doubleRegisterArgs, DOUBLE_REG_ORDER):
                        # Move the variable to the register in order.
                        self.createInst(MOV, movAsmbType, value, Register(value.assemblyType, reg))

                    # Push the values in reversed order.
                    for (value, movAsmbType) in reversed(stackArgs):
                        if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                            # There may be part of a struct/union returned in a register whose byte 
                            # size is not standard. For this case, allocate as 8 bytes in the stack 
                            # (ABI tells us to always reserve 8 bytes for an eightbyte, even if it's 
                            # incomplete).
                            self.createInst(BINARY, 
                                AssemblyType.QUADWORD, BinaryOperator.SUBTRACT, 
                                self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), "8")),
                                Register(AssemblyType.QUADWORD, REG.SP))
                            self.instructions.extend(self.copyBytes(value, Memory(value.assemblyType, REG.SP, 0), movAsmbType))
                        elif isinstance(value, (Register, Immediate)) or value.assemblyType in (AssemblyType.QUADWORD, AssemblyType.DOUBLE):
                            # This value can be directly pushed as it is 8 bytes.
                            self.createInst(PUSH, value)
                        else:
                            # This value is under 8 bytes so it must be transferred to a register and
                            # then pushed. As I will be using AX (XMM registers cannot be pushed), decimal
                            # values must be converted to their integer counterparts.
                            if movAsmbType == AssemblyType.DOUBLE:
                                movAsmbType = AssemblyType.QUADWORD
                            elif movAsmbType == AssemblyType.FLOAT:
                                movAsmbType = AssemblyType.LONGWORD

                            self.createInst(MOV, movAsmbType, value, Register(movAsmbType, REG.AX))
                            self.createInst(PUSH, Register(AssemblyType.QUADWORD, REG.AX))
                    
                    # Emit the call instruction.
                    self.createInst(CALL, inst.identifier)

                    # Readjust the stack pointer.
                    deallocBytes = 8 * len(stackArgs) + stackPadding
                    if deallocBytes != 0:
                        # DEALLOCATE_STACK: deallocBytes
                        self.createInst(BINARY, 
                                        AssemblyType.QUADWORD, BinaryOperator.SUM, 
                                        self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), str(deallocBytes))),
                                        Register(AssemblyType.QUADWORD, REG.SP))
                    
                    # Retrieve the return value if function is not void and the return value is not 
                    # stored in the stack.
                    if inst.returnType != TypeSpecifier.VOID.toBaseType() and not returnInStack:
                        INT_RETURN_REGS = [REG.AX, REG.DX]
                        DOUBLE_RETURN_REGS = [REG.XMM0, REG.XMM1]

                        for (value, movAsmbType), reg in zip(returnIntRegs, INT_RETURN_REGS):
                            if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                                self.instructions.extend(self.copyBytesFromRegister(reg, value, movAsmbType.size))
                            else:
                                self.createInst(MOV, movAsmbType, Register(value.assemblyType, reg), value)

                        for (value, movAsmbType), reg in zip(returnDoubleRegs, DOUBLE_RETURN_REGS):
                            self.createInst(MOV, movAsmbType, Register(value.assemblyType, reg), value)

                case TACSignExtend():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)
                    self.createInst(MOVS, exp.assemblyType, result.assemblyType, exp, result)
                
                case TACTruncate():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)
                    self.createInst(MOV, result.assemblyType, exp, result)

                case TACZeroExtend():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)
                    self.createInst(MOVZ, exp.assemblyType, result.assemblyType, exp, result)

                case TACIntToDecimal():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)
                    
                    # INT2DEC only supports long or quadwords as inputs.
                    if exp.assemblyType.size < AssemblyType.LONGWORD.size:
                        if inst.exp.valueType.isSigned():
                            # Sign extend to a longword.
                            self.createInst(MOVS,
                                            exp.assemblyType, AssemblyType.LONGWORD,  
                                            exp, 
                                            Register(AssemblyType.LONGWORD, REG.AX))
                        else:
                            # Zero extend to a longword.
                            self.createInst(MOVZ, 
                                            exp.assemblyType, AssemblyType.LONGWORD,  
                                            exp, 
                                            Register(AssemblyType.LONGWORD, REG.AX))
                        self.createInst(INT2DEC,
                                        exp.assemblyType, result.assemblyType, 
                                        Register(AssemblyType.LONGWORD, REG.AX), 
                                        result)
                    else:
                        self.createInst(INT2DEC, 
                                        exp.assemblyType, result.assemblyType, 
                                        exp, 
                                        result)

                case TACDecimalToInt():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)

                    # DEC2INT only supports long or quadwords as outputs.
                    if result.assemblyType.size < AssemblyType.LONGWORD.size:
                        self.createInst(DEC2INT, 
                                        exp.assemblyType, AssemblyType.LONGWORD, 
                                        exp, 
                                        Register(AssemblyType.LONGWORD, REG.AX))
                        # Transfer the result using the result type.
                        self.createInst(MOV, 
                                        result.assemblyType,
                                        Register(AssemblyType.LONGWORD, REG.AX), 
                                        result)
                    else:
                        self.createInst(DEC2INT, 
                                        exp.assemblyType, result.assemblyType,
                                        exp, 
                                        result)

                case TACDecimalToDecimal():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)

                    self.createInst(DEC2DEC, exp.assemblyType, result.assemblyType, exp, result)

                case TACUIntToDecimal():
                    if not isinstance(inst.exp.valueType, BaseDeclaratorType):
                        raise ValueError()
                    
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)
                    
                    if inst.exp.valueType.baseType in (TypeSpecifier.UINT, TypeSpecifier.USHORT, TypeSpecifier.UCHAR):
                        # Zero extend from x-bit to 64-bit.
                        self.createInst(MOVZ, 
                                        exp.assemblyType, AssemblyType.QUADWORD, 
                                        exp, 
                                        Register(AssemblyType.QUADWORD, REG.AX))
                        # Convert to decimal.
                        self.createInst(INT2DEC, 
                                        AssemblyType.QUADWORD, result.assemblyType,
                                        Register(AssemblyType.QUADWORD, REG.AX), 
                                        result)
                    else: # ULONG
                        outOfRangeLabel = LABEL.generateLabel("ui2d.outOfRange")
                        endConversionLabel = LABEL.generateLabel("ui2d.endConversion")
                        self.createInst(CMP,
                                        exp.assemblyType,
                                        self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), "0")), 
                                        exp)
                        self.createInst(JMPIF,
                                        ConditionCode.ABOVE_EQUAL,
                                        outOfRangeLabel)
                        self.createInst(INT2DEC, 
                                        exp.assemblyType, result.assemblyType,
                                        exp, 
                                        result)
                        self.createInst(JMP, endConversionLabel)
                        self.createInst(LABEL, outOfRangeLabel)
                        self.createInst(MOV,
                                        exp.assemblyType,
                                        exp,
                                        Register(exp.assemblyType, REG.AX))
                        self.createInst(MOV,
                                        exp.assemblyType,
                                        Register(exp.assemblyType, REG.AX),
                                        Register(exp.assemblyType, REG.DX))
                        self.createInst(BINARY,
                                        exp.assemblyType, BinaryOperator.LOGIC_RIGHT_SHIFT,
                                        self.fromTACValue(TACValue(True, inst.exp.valueType, "1")),
                                        Register(exp.assemblyType, REG.DX))
                        self.createInst(BINARY,
                                        exp.assemblyType, BinaryOperator.BITWISE_AND,
                                        self.fromTACValue(TACValue(True, inst.exp.valueType, "1")),
                                        Register(exp.assemblyType, REG.AX))
                        self.createInst(BINARY,
                                        exp.assemblyType, BinaryOperator.BITWISE_OR,
                                        Register(exp.assemblyType, REG.AX),
                                        Register(exp.assemblyType, REG.DX))
                        self.createInst(INT2DEC, 
                                        exp.assemblyType, result.assemblyType,
                                        Register(exp.assemblyType, REG.DX), 
                                        result)
                        self.createInst(BINARY,
                                        result.assemblyType, BinaryOperator.SUM,
                                        result, 
                                        result)
                        self.createInst(LABEL, endConversionLabel)
                    
                case TACDecimalToUInt():
                    if not isinstance(inst.result.valueType, BaseDeclaratorType):
                        raise ValueError()

                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)

                    if inst.result.valueType.baseType in (TypeSpecifier.UINT, TypeSpecifier.USHORT, TypeSpecifier.UCHAR):
                        self.createInst(DEC2INT,
                                        exp.assemblyType, AssemblyType.QUADWORD, 
                                        exp, 
                                        Register(AssemblyType.QUADWORD, REG.AX))
                        self.createInst(MOV, 
                                        result.assemblyType,
                                        Register(result.assemblyType, REG.AX),
                                        result)
                    else: # ULONG
                        outOfRangeLabel = LABEL.generateLabel("ui2d.outOfRange")
                        endConversionLabel = LABEL.generateLabel("ui2d.endConversion")
                        self.createInst(CMP,
                                        exp.assemblyType,
                                        self.fromTACValue(TACValue(True, inst.exp.valueType, "9223372036854775808.0")), 
                                        exp)
                        self.createInst(JMPIF,
                                        ConditionCode.ABOVE_EQUAL,
                                        outOfRangeLabel)
                        self.createInst(DEC2INT, 
                                        exp.assemblyType, result.assemblyType,
                                        exp, 
                                        result)
                        self.createInst(JMP, endConversionLabel)
                        self.createInst(LABEL, outOfRangeLabel)
                        self.createInst(MOV,
                                        exp.assemblyType,
                                        exp,
                                        Register(exp.assemblyType, REG.XMM0))
                        self.createInst(BINARY,
                                        exp.assemblyType, BinaryOperator.SUBTRACT,
                                        self.fromTACValue(TACValue(True, inst.exp.valueType, "9223372036854775808.0")),
                                        Register(exp.assemblyType, REG.XMM0))
                        self.createInst(DEC2INT,
                                        exp.assemblyType, result.assemblyType,
                                        Register(exp.assemblyType, REG.XMM0), 
                                        result)
                        self.createInst(MOV,
                                        result.assemblyType,
                                        self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), "9223372036854775808")), 
                                        Register(result.assemblyType, REG.AX))
                        self.createInst(BINARY,
                                        result.assemblyType, BinaryOperator.SUM,
                                        Register(result.assemblyType, REG.AX),
                                        result)
                        self.createInst(LABEL, endConversionLabel)

                case _:
                    raise ValueError(f"Unexpected statement {inst} when parsing an AssemblerFunction")

    def secondPass(self):
        for inst in self.instructions:
            inst.secondPass()

    def thirdPass(self):
        newInstructions: list[AssemblerInstruction] = []

        # Allocate the stack.
        functionStackAlloc = -Memory.STACK_OFFSET
        # Round to the next multiple of 16. Makes it easier to align function calls.
        functionStackAlloc = 16 * math.ceil(functionStackAlloc / 16)

        if functionStackAlloc > 0:
            # ALLOCATE_STACK: functionStackAlloc
            stackAllocInstruction = BINARY( 
                AssemblyType.QUADWORD, BinaryOperator.SUBTRACT, 
                self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), str(functionStackAlloc))),
                Register(AssemblyType.QUADWORD, REG.SP))
            newInstructions.append(stackAllocInstruction)

        # Fix the instructions.
        for inst in self.instructions:
            newInstructions.extend(inst.thirdPass())

        self.instructions = newInstructions

    def emitCode(self) -> str:
        ret = ""
        if self.function.isGlobal:
            ret =  f"\t.globl {self.identifier}\n"
        
        ret += "\t.text\n"
        ret += f"{self.identifier}:\n"
        ret += f"\tpushq\t%rbp\n"
        ret += f"\tmovq\t%rsp, %rbp\n"

        for inst in self.instructions:
            ret += inst.emitCode()

        return ret

    def print(self) -> str:
        ret = f"--- {self.identifier} ---\n"
        for inst in self.instructions:
            ret += inst.print()
        return ret
"""
INSTRUCTIONS
"""
class AssemblerInstruction(AssemblyAST):
    def firstPass(self):
        pass

    # Can be overriden.
    def secondPass(self):
        pass

    # Returns the fixed instruction, which may include multiple instructions.
    # Can be overriden.
    def thirdPass(self) -> list[AssemblerInstruction]:
        return [self]

    @abstractmethod
    def emitCode(self) -> str:
        pass

    @abstractmethod
    def print(self) -> str:
        pass

    def convertFromPseudo(self, var): # var is an AssemblerOperand
        if isinstance(var, Pseudo):
            if var.name in TACStaticVariable.staticVariables:
                return Data(var.assemblyType, var.name, 0, var.parent)
            else:
                return Memory.convertToStackVariable(var)
        elif isinstance(var, PseudoMemory):
            if var.name in TACStaticVariable.staticVariables:
                return Data(var.assemblyType, var.name, var.offset, var.parent)
            else:
                return Memory.convertToStackMemory(var)

        return var

class MOV(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, src: AssemblerOperand, dst: AssemblerOperand,
                 parentAST: AssemblyAST | None = None) -> None:
        if not isinstance(src, AssemblerOperand):
            raise ValueError(f"Invalid argument {src}: MOV only receives AssemblerOperands")
        if not isinstance(dst, AssemblerOperand):
            raise ValueError(f"Invalid argument {dst}: MOV only receives AssemblerOperands")
        
        self.asmbType = asmbType
        self.src = src
        self.dst = dst

        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if (isinstance(self.src, (Memory, Data)) and isinstance(self.dst, (Memory, Data))) \
            or \
           (isinstance(self.src, Immediate) and self.src.isQuadImmediate() and isinstance(self.dst, (Memory, Data))):
            # - MOV cannot have two memory addresses. Save the src into a temporary register and 
            # then pass it to the dst.
            # - Cannot move a 64-bit immediate to memory, instead move it to a register and then to 
            # memory.
            if self.asmbType.isDecimal():
                tempReg = REG.XMM14
            else:
                tempReg = REG.R10

            movToReg = self.createChild(MOV, self.asmbType, 
                                        self.src, 
                                        Register(self.asmbType, tempReg))
            movFromReg = self.createChild(MOV, self.asmbType,
                                        Register(self.asmbType, tempReg), 
                                        self.dst)
            return [movToReg, movFromReg]

        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.src.assemblyType = self.asmbType
        self.dst.assemblyType = self.asmbType

        # If the source is a common register, movsd should be a movq instruction.
        if isinstance(self.src, Register) and self.dst.assemblyType == AssemblyType.DOUBLE and \
           not self.src.reg.isFloatRegister():
            return f"\tmovq\t{self.src.emitCode()}, {self.dst.emitCode()}\n"
        else:
            return f"\tmov{self.asmbType.getTypeName()}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"Mov({self.src}, {self.dst})\n"

# Move with Sign Extension
class MOVS(AssemblerInstruction):
    def __init__(self, srcAsmbType: AssemblyType, dstAsmbType: AssemblyType, 
                 src: AssemblerOperand, dst: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        
        self.srcAsmbType = srcAsmbType
        self.dstAsmbType = dstAsmbType
        self.src = src
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.dst, (Memory, Data)) \
            or \
           (isinstance(self.src, Immediate) and self.src.isQuadImmediate() and isinstance(self.dst, (Memory, Data))):
            # - MOVS cannot have a memory address as destination. 
            # - Cannot move a 64-bit immediate to memory, instead move it to a register and then to 
            # memory.
            movToReg = self.createChild(MOV, 
                                        self.src.assemblyType, 
                                        self.src, 
                                        Register(self.src.assemblyType, REG.R10))
            extendSign = self.createChild(MOVS, 
                                          self.src.assemblyType, self.dst.assemblyType,
                                          Register(self.src.assemblyType, REG.R10), 
                                          Register(self.dst.assemblyType, REG.R11))
            movToDst = self.createChild(MOV, 
                                        self.dst.assemblyType,
                                        Register(self.dst.assemblyType, REG.R11), 
                                        self.dst)
            return [movToReg, extendSign, movToDst]
        
        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.src.assemblyType = self.srcAsmbType
        self.dst.assemblyType = self.dstAsmbType

        return f"\tmovs{self.srcAsmbType.getTypeName()}{self.dst.assemblyType.getTypeName()}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"Movs({self.src}, {self.dst})\n"

# Move with Zero Extension
class MOVZ(AssemblerInstruction):
    def __init__(self, srcAsmbType: AssemblyType, dstAsmbType: AssemblyType,
                 src: AssemblerOperand, dst: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        self.srcAsmbType = srcAsmbType
        self.dstAsmbType = dstAsmbType
        self.src = src
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.dst, (Memory, Data)) \
            or \
           (isinstance(self.src, Immediate) and self.src.isQuadImmediate() and isinstance(self.dst, (Memory, Data))):
            # - MOVZ cannot have a memory address as destination. 
            # - Cannot move a 64-bit immediate to memory, instead move it to a register and then to 
            # memory.
            if self.src.assemblyType.isDecimal():
                tempReg1 = REG.XMM14
            else:
                tempReg1 = REG.R10

            if self.dst.assemblyType.isDecimal():
                tempReg2 = REG.XMM15
            else:
                tempReg2 = REG.R11

            movToReg = self.createChild(MOV, 
                                        self.src.assemblyType, 
                                        self.src, 
                                        Register(self.src.assemblyType, tempReg1))

            if self.src.assemblyType == AssemblyType.LONGWORD and self.dst.assemblyType.isQuad():
                # Longword to quadword is automatically zero extended with a MOV instruction.
                # The idea is to move to the second register with the "size" of the first.
                extendSign = self.createChild(MOV, 
                                              self.src.assemblyType,
                                              Register(self.src.assemblyType, tempReg1), 
                                              Register(self.src.assemblyType, tempReg2))
            else:
                extendSign = self.createChild(MOVZ, 
                                              self.src.assemblyType, self.dst.assemblyType,
                                              Register(self.src.assemblyType, tempReg1), 
                                              Register(self.dst.assemblyType, tempReg2))
            
            movToDst = self.createChild(MOV, 
                                        self.dst.assemblyType, 
                                        Register(self.dst.assemblyType, tempReg2), 
                                        self.dst)
            return [movToReg, extendSign, movToDst]
        
        if self.src.assemblyType == AssemblyType.LONGWORD and self.dst.assemblyType.isQuad():
            # Longword to quadword is automatically zero extended with a MOV instruction. 
            # Simply move it to the destination with the "size" of the source.
            extendSign = self.createChild(MOV, self.src.assemblyType, self.src, self.dst)
            return [extendSign]

        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.src.assemblyType = self.srcAsmbType
        self.dst.assemblyType = self.dstAsmbType

        return f"\tmovz{self.src.assemblyType.getTypeName()}{self.dst.assemblyType.getTypeName()}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"Movz({self.src}, {self.dst})\n"

class LEA(AssemblerInstruction):
    def __init__(self, src: AssemblerOperand, dst: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        
        if dst.assemblyType != AssemblyType.QUADWORD:
            raise ValueError("LEA expects a QUADWORD as destination")

        self.src = src
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.dst, (Memory, Data)):
            # - LEA cannot have a memory address as destination. 
            leaToReg = self.createChild(LEA, 
                                        self.src, 
                                        Register(AssemblyType.QUADWORD, REG.R11))
            movToDst = self.createChild(MOV, 
                                        AssemblyType.QUADWORD, 
                                        Register(AssemblyType.QUADWORD, REG.R11), 
                                        self.dst)
            return [leaToReg, movToDst]
        
        return [self]

    def emitCode(self) -> str:
        return f"\tleaq\t{self.src.emitCode()}, {self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"LoadAddress({self.src}, {self.dst})\n"

# Decimal to signed int.
class DEC2INT(AssemblerInstruction):
    def __init__(self, srcAsmbType: AssemblyType, dstAsmbType: AssemblyType,
                 src: AssemblerOperand, dst: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        
        self.srcAsmbType = srcAsmbType
        self.dstAsmbType = dstAsmbType
        self.src = src
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        # The destination must be a register.
        if not isinstance(self.dst, Register):
            resultInReg = self.createChild(DEC2INT, 
                                           self.src.assemblyType, self.dst.assemblyType, 
                                           self.src, 
                                           Register(self.dst.assemblyType, REG.R11))
            copyToDst = self.createChild(MOV, 
                                         self.dst.assemblyType, 
                                         Register(self.dst.assemblyType, REG.R11), 
                                         self.dst)
            return [resultInReg, copyToDst]

        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.src.assemblyType = self.srcAsmbType
        self.dst.assemblyType = self.dstAsmbType

        if self.srcAsmbType == AssemblyType.DOUBLE:
            return f"\tcvttsd2si{self.dst.assemblyType.getTypeName()}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"
        elif self.srcAsmbType == AssemblyType.FLOAT:
            return f"\tcvttss2si{self.dst.assemblyType.getTypeName()}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"
        else:
            raise ValueError(f"Invalid type {self.dst.assemblyType} in DEC2INT")

    def print(self) -> str:
        return f"DEC2INT({self.src}, {self.dst})\n"

# Signed int to decimal.
class INT2DEC(AssemblerInstruction):
    def __init__(self, srcAsmbType: AssemblyType, dstAsmbType: AssemblyType,
                 src: AssemblerOperand, dst: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        
        self.srcAsmbType = srcAsmbType
        self.dstAsmbType = dstAsmbType
        self.src = src
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        # The source cannot be a constant.
        srcIsConstant = isinstance(self.src, Immediate)
        # The destination must be a register.
        dstIsRegister = isinstance(self.dst, Register)

        if srcIsConstant and dstIsRegister:
            moveSrc = self.createChild(MOV, 
                                       self.src.assemblyType, 
                                       self.src, 
                                       Register(self.src.assemblyType, REG.R10))
            resultInReg = self.createChild(INT2DEC, 
                                           self.src.assemblyType, self.dst.assemblyType,
                                           Register(self.src.assemblyType, REG.R10), 
                                           self.dst)
            return [moveSrc, resultInReg]

        if not srcIsConstant and not dstIsRegister:
            resultInReg = self.createChild(INT2DEC, 
                                           self.src.assemblyType, self.dst.assemblyType,
                                           self.src, 
                                           Register(self.dst.assemblyType, REG.XMM15))
            copyToDst = self.createChild(MOV, 
                                         self.dst.assemblyType, 
                                         Register(self.dst.assemblyType, REG.XMM15), 
                                         self.dst)
            return [resultInReg, copyToDst]
        
        if srcIsConstant and not dstIsRegister:
            moveSrc = self.createChild(MOV, 
                                       self.src.assemblyType, 
                                       self.src, 
                                       Register(self.src.assemblyType, REG.R10))
            resultInReg = self.createChild(INT2DEC, 
                                           self.src.assemblyType, self.dst.assemblyType,
                                           Register(self.src.assemblyType, REG.R10), 
                                           Register(self.dst.assemblyType, REG.XMM15))
            copyToDst = self.createChild(MOV, 
                                         self.dst.assemblyType,
                                         Register(self.dst.assemblyType, REG.XMM15), 
                                         self.dst)
            return [moveSrc, resultInReg, copyToDst]

        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.src.assemblyType = self.srcAsmbType
        self.dst.assemblyType = self.dstAsmbType

        if self.dstAsmbType == AssemblyType.DOUBLE:
            return f"\tcvtsi2sd{self.src.assemblyType.getTypeName()}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"
        elif self.dstAsmbType == AssemblyType.FLOAT:
            return f"\tcvtsi2ss{self.src.assemblyType.getTypeName()}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"
        else:
            raise ValueError(f"Invalid type {self.dst.assemblyType} in INT2DEC")

    def print(self) -> str:
        return f"INT2DEC({self.src}, {self.dst})\n"

# Decimal to decimal.
class DEC2DEC(AssemblerInstruction):
    def __init__(self, srcAsmbType: AssemblyType, dstAsmbType: AssemblyType,
                 src: AssemblerOperand, dst: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        
        self.srcAsmbType = srcAsmbType
        self.dstAsmbType = dstAsmbType
        self.src = src
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        # The source and destination must be a register.

        if not isinstance(self.src, Register) and not isinstance(self.dst, Register):
            srcToReg = self.createChild(MOV, 
                                        self.src.assemblyType, 
                                        self.src, 
                                        Register(self.src.assemblyType, REG.XMM14))
            resultInReg = self.createChild(DEC2DEC,
                                           self.src.assemblyType, self.dst.assemblyType, 
                                           Register(self.src.assemblyType, REG.XMM14), 
                                           Register(self.dst.assemblyType, REG.XMM15))
            copyToDst = self.createChild(MOV, 
                                         self.dst.assemblyType,
                                         Register(self.dst.assemblyType, REG.XMM15), 
                                         self.dst)
            return [srcToReg, resultInReg, copyToDst]

        if not isinstance(self.dst, Register):
            resultInReg = self.createChild(DEC2DEC, 
                                           self.src.assemblyType, self.dst.assemblyType,
                                           self.src, 
                                           Register(self.dst.assemblyType, REG.XMM15))
            copyToDst = self.createChild(MOV, 
                                         self.dst.assemblyType,
                                         Register(self.dst.assemblyType, REG.XMM15), 
                                         self.dst)
            return [resultInReg, copyToDst]


        if not isinstance(self.src, Register):
            srcToReg = self.createChild(MOV, 
                                        self.src.assemblyType,
                                        self.src, 
                                        Register(self.src.assemblyType, REG.XMM14))
            resultInReg = self.createChild(DEC2DEC, 
                                           self.src.assemblyType, self.dst.assemblyType,
                                           Register(self.src.assemblyType, REG.XMM14), 
                                           self.dst)
            return [srcToReg, resultInReg]

        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.src.assemblyType = self.srcAsmbType
        self.dst.assemblyType = self.dstAsmbType

        if self.dstAsmbType == AssemblyType.DOUBLE:
            return f"\tcvtss2sd\t{self.src.emitCode()}, {self.dst.emitCode()}\n"
        elif self.dstAsmbType == AssemblyType.FLOAT:
            return f"\tcvtsd2ss\t{self.src.emitCode()}, {self.dst.emitCode()}\n"
        else:
            raise ValueError(f"Invalid type {self.dst.assemblyType} in DEC2DEC")

    def print(self) -> str:
        return f"DEC2DEC({self.src}, {self.dst})\n"

class UNARY(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, operator: UnaryOperator, operand: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        self.asmbType = asmbType
        self.operator = operator
        self.operand = operand
        super().__init__(parentAST)

    def secondPass(self):
        self.operand = self.convertFromPseudo(self.operand)

    def emitCode(self) -> str:
        self.operand.assemblyType = self.asmbType
        match self.operator:
            case UnaryOperator.BITWISE_COMPLEMENT:
                return f"\tnot{self.asmbType.getTypeName()}\t{self.operand.emitCode()}\n"
            case UnaryOperator.NEGATION:
                return f"\tneg{self.asmbType.getTypeName()}\t{self.operand.emitCode()}\n"
            case UnaryOperator.INCREMENT:
                return f"\tinc{self.asmbType.getTypeName()}\t{self.operand.emitCode()}\n"
            case UnaryOperator.DECREMENT:
                return f"\tdec{self.asmbType.getTypeName()}\t{self.operand.emitCode()}\n"
            case _:
                raise ValueError(f"There is no code emission for UNARY {self.operator}")

    def print(self) -> str:
        match self.operator:
            case UnaryOperator.BITWISE_COMPLEMENT:
                return f"Not({self.operand})\n"
            case UnaryOperator.NEGATION:
                return f"Neg({self.operand})\n"
            case UnaryOperator.INCREMENT:
                return f"Inc({self.operand})\n"
            case UnaryOperator.DECREMENT:
                return f"Dec({self.operand})\n"
            case _:
                raise ValueError(f"Invalid operator {self.operator} in UNARY")

class PUSH(AssemblerInstruction):
    def __init__(self, operand: AssemblerOperand, parentAST: AssemblyAST | None = None) -> None:
        self.operand = operand
        super().__init__(parentAST)

    def secondPass(self):
        self.operand = self.convertFromPseudo(self.operand)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.operand, (Memory, Data)) \
            or \
           (isinstance(self.operand, Immediate) and self.operand.isQuadImmediate()):
            # - Stack cannot have a memory address. Save the src into temporary register AX and then
            # push the 64 bit AX.
            # - Cannot push a 64-bit immediate, instead move it to a register and then push it.
            
            # A movsd (move double) operation cannot be done to REG.AX, but XMMx registers cannot be
            # pushed to the stack.
            if self.operand.assemblyType.isDecimal():
                asmbType = AssemblyType.QUADWORD
            else:
                asmbType = self.operand.assemblyType

            movToReg = self.createChild(MOV, asmbType, self.operand, Register(asmbType, REG.AX))
            pushFromReg = self.createChild(PUSH, Register(AssemblyType.QUADWORD, REG.AX))
            return [movToReg, pushFromReg]
        
        return [self]

    def emitCode(self) -> str:
        return f"\tpushq\t{self.operand.emitCode()}\n"
    
    def print(self) -> str:
        return f"Push({self.operand})\n"
    
class CALL(AssemblerInstruction):
    def __init__(self, funcIdentifier: str, parentAST: AssemblyAST | None = None) -> None:
        self.funcIdentifier = funcIdentifier
        super().__init__(parentAST)

    def emitCode(self) -> str:
        if self.funcIdentifier in TACFunction.functions:
            return f"\tcall\t{self.funcIdentifier}\n"
        else:
            # If the function is not defined in the code, maybe it's located somewhere else.
            # Add @PLT to link it externally. 
            return f"\tcall\t{self.funcIdentifier}@PLT\n"
    
    def print(self) -> str:
        return f'Call({self.funcIdentifier})\n'

class RET(AssemblerInstruction):
    def __init__(self, parentAST: AssemblyAST | None = None) -> None:
        super().__init__(parentAST)

    def emitCode(self) -> str:
        ret  = f"\tmovq\t%rbp, %rsp\n"
        ret += f"\tpopq\t%rbp\n"
        ret += f"\tret\n"
        return ret

    def print(self) -> str:
        return "Ret\n"

class BINARY(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, operator: BinaryOperator, 
                 op1: AssemblerOperand, op2: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        self.asmbType = asmbType
        self.operator = operator
        # op2 = op2 operator op1
        self.op1 = op1
        self.op2 = op2

        super().__init__(parentAST)

    def secondPass(self):
        self.op1 = self.convertFromPseudo(self.op1)
        self.op2 = self.convertFromPseudo(self.op2)

    def thirdPass(self) -> list[AssemblerInstruction]:
        match self.operator:
            case BinaryOperator.SUM | BinaryOperator.SUBTRACT | BinaryOperator.MULTIPLICATION | \
                 BinaryOperator.DIVISION | BinaryOperator.BITWISE_AND | \
                 BinaryOperator.BITWISE_OR | BinaryOperator.BITWISE_XOR:
                
                # Can't have immediate longs.
                op1IsLong = isinstance(self.op1, Immediate) and self.op1.isQuadImmediate()
                # IMUL instruction can't use a memory address as destination.
                op2IsMemory = isinstance(self.op2, (Memory, Data))

                if self.op1.assemblyType.isDecimal():
                    tempRegOP1 = REG.XMM14
                else:
                    tempRegOP1 = REG.R10
                
                if self.op2.assemblyType.isDecimal():
                    tempRegOP2 = REG.XMM15
                else:
                    tempRegOP2 = REG.R11

                # Integer operations (except multiplication) can have the second argument in memory.
                if op1IsLong and \
                   (not op2IsMemory or \
                        (op2IsMemory and self.operator != BinaryOperator.MULTIPLICATION and \
                        not self.op2.assemblyType.isDecimal())
                   ):
                    movOp1ToReg = self.createChild(MOV, 
                                                   self.op1.assemblyType, 
                                                   self.op1, 
                                                   Register(self.op1.assemblyType, tempRegOP1))
                    operateOnReg = self.createChild(BINARY, 
                                                    self.asmbType, self.operator, 
                                                    Register(self.op1.assemblyType, tempRegOP1), 
                                                    self.op2)
                    return [movOp1ToReg, operateOnReg]
                
                if not op1IsLong and op2IsMemory:
                    movOp2ToReg = self.createChild(MOV, 
                                                   self.op2.assemblyType,
                                                   self.op2, 
                                                   Register(self.op2.assemblyType, tempRegOP2))
                    operateOnReg = self.createChild(BINARY, 
                                                    self.asmbType, self.operator, 
                                                    self.op1, 
                                                    Register(self.op2.assemblyType, tempRegOP2))
                    movFromReg = self.createChild(MOV, 
                                                  self.asmbType,
                                                  Register(self.op2.assemblyType, tempRegOP2), 
                                                  self.op2)
                    return [movOp2ToReg, operateOnReg, movFromReg]
                
                if op1IsLong and op2IsMemory:
                    movOp1ToReg = self.createChild(MOV, 
                                                   self.op1.assemblyType,
                                                   self.op1, 
                                                   Register(self.op1.assemblyType, tempRegOP1))
                    movOp2ToReg = self.createChild(MOV, 
                                                   self.op2.assemblyType,
                                                   self.op2, 
                                                   Register(self.op2.assemblyType, tempRegOP2))
                    operateOnReg = self.createChild(BINARY, 
                                                    self.asmbType, self.operator, 
                                                    Register(self.op1.assemblyType, tempRegOP1), 
                                                    Register(self.op2.assemblyType, tempRegOP2))
                    movFromReg = self.createChild(MOV, 
                                                  self.asmbType,
                                                  Register(self.op2.assemblyType, tempRegOP2), 
                                                  self.op2)
                    return [movOp1ToReg, movOp2ToReg, operateOnReg, movFromReg]

            case BinaryOperator.LOGIC_LEFT_SHIFT | BinaryOperator.LOGIC_RIGHT_SHIFT | \
                 BinaryOperator.ARITHMETIC_LEFT_SHIFT | BinaryOperator.ARITHMETIC_RIGHT_SHIFT:
                if isinstance(self.op1, (Memory, Data)):
                    # In x86, you cannot shift by an amount stored in memory, it must be stored in
                    # register CX (8 bits).
                    movToReg = self.createChild(MOV, 
                                                AssemblyType.BYTE, 
                                                self.op1, Register(AssemblyType.BYTE, REG.CX))
                    operateOnReg = self.createChild(BINARY, 
                                                    self.asmbType, self.operator, 
                                                    Register(AssemblyType.BYTE, REG.CX), 
                                                    self.op2)
                    return [movToReg, operateOnReg]

            case _:
                raise ValueError(f"Invalid operator in BinaryOperator: {self.operator}")
    
        return [self]
    
    def emitCode(self) -> str:
        self.op1.assemblyType = self.asmbType
        self.op2.assemblyType = self.asmbType

        match self.operator:
            case BinaryOperator.SUM:
                return f"\tadd{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.SUBTRACT:
                return f"\tsub{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.MULTIPLICATION:
                if self.asmbType.isDecimal():
                    return f"\tmul{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
                else:
                    return f"\timul{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.DIVISION:
                return f"\tdiv{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.LOGIC_LEFT_SHIFT:
                return f"\tshl{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.ARITHMETIC_LEFT_SHIFT:
                return f"\tsal{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.LOGIC_RIGHT_SHIFT:
                return f"\tshr{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.ARITHMETIC_RIGHT_SHIFT:
                return f"\tsar{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.BITWISE_AND:
                return f"\tand{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.BITWISE_XOR:
                if self.asmbType.isDecimal():
                    # The XOR for decimal types uses a vector instuction.
                    return f"\txorpd\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
                else:
                    return f"\txor{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case BinaryOperator.BITWISE_OR:
                return f"\tor{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
            case _:
                raise ValueError(f"Invalid operator in BinaryOperator: {self.operator}")
    
    def print(self) -> str:
        return f"Binary({self.operator.name}, {self.op1}, {self.op2})\n"

# Integer division.
class IDIV(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, operator: BinaryOperator, op: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        self.asmbType = asmbType
        self.operator = operator
        self.op = op
        super().__init__(parentAST)

    def secondPass(self):
        self.op = self.convertFromPseudo(self.op)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.op, Immediate):
            # IDIV cannot operate on immediate values.
            movToReg = self.createChild(MOV, 
                                        self.op.assemblyType, 
                                        self.op, 
                                        Register(self.op.assemblyType, REG.R10))
            operateOnReg = self.createChild(IDIV, 
                                            self.asmbType, self.operator, 
                                            Register(self.op.assemblyType, REG.R10))
            return [movToReg, operateOnReg]
        
        return [self]

    def emitCode(self) -> str:
        self.op.assemblyType = self.asmbType
        return f"\tidiv{self.op.assemblyType.getTypeName()}\t{self.op.emitCode()}\n"

    def print(self) -> str:
        return f"Binary({self.operator.name}, {self.op}, AX)\n"

# Float division.
class DIV(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, operator: BinaryOperator, op: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        self.asmbType = asmbType
        self.operator = operator
        self.op = op
        super().__init__(parentAST)

    def secondPass(self):
        self.op = self.convertFromPseudo(self.op)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.op, Immediate):
            # DIV cannot operate on immediate values.
            movToReg = self.createChild(MOV, 
                                        self.op.assemblyType,
                                        self.op, 
                                        Register(self.op.assemblyType, REG.R10))
            operateOnReg = self.createChild(DIV, 
                                            self.asmbType, self.operator, 
                                            Register(self.op.assemblyType, REG.R10))
            return [movToReg, operateOnReg]
        
        return [self]

    def emitCode(self) -> str:
        return f"\tdiv{self.asmbType.getTypeName()}\t{self.op.emitCode()}\n"

    def print(self) -> str:
        return f"Binary({self.operator.name}, {self.op}, AX)\n"

# Sign extend AX register.
class CDQ(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, parentAST: AssemblyAST | None = None) -> None:
        self.asmbType = asmbType
        super().__init__(parentAST)

    def emitCode(self) -> str:
        match self.asmbType:
            case AssemblyType.LONGWORD:
                return "\tcdq\n"
            case AssemblyType.QUADWORD:
                return "\tcqo\n"
            case _:
                raise ValueError(f"Cannot generate code for CDQ of {self.asmbType}")

    def print(self) -> str:
        return f"SignExtend AX\n"

class CMP(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, op1: AssemblerOperand, op2: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        self.asmbType = asmbType
        self.op1 = op1
        self.op2 = op2

        super().__init__(parentAST)

    def secondPass(self):
        self.op1 = self.convertFromPseudo(self.op1)
        self.op2 = self.convertFromPseudo(self.op2)

    def thirdPass(self) -> list[AssemblerInstruction]:
        # - These instructions cannot have two memory addresses. Save the src into a 
        # temporary register and then pass it to the dst.
        # - Cannot move a 64-bit immediate, instead move it to a register and then 
        # operate.
        moveOp1ToReg =  (isinstance(self.op1, (Memory, Data)) and isinstance(self.op2, (Memory, Data))) \
                        or \
                        (isinstance(self.op1, Immediate) and self.op1.isQuadImmediate())
        # CMP cannot have a constant as its second argument.
        # For double operations, the second argument must always be a register.
        moveOp2ToReg = isinstance(self.op2, Immediate) or \
                       (self.op2.assemblyType.isDecimal() and \
                       not isinstance(self.op2, Register))

        if self.op1.assemblyType.isDecimal():
            tempRegOP1 = REG.XMM14
        else:
            tempRegOP1 = REG.R10
        
        if self.op2.assemblyType.isDecimal():
            tempRegOP2 = REG.XMM15
        else:
            tempRegOP2 = REG.R11

        if moveOp1ToReg and not moveOp2ToReg:
            movOp1ToReg = self.createChild(MOV, 
                                           self.op1.assemblyType,
                                           self.op1, 
                                           Register(self.op1.assemblyType, tempRegOP1))
            cmpInst = self.createChild(CMP,
                                       self.asmbType, 
                                       Register(self.op1.assemblyType, tempRegOP1), 
                                       self.op2)
            return [movOp1ToReg, cmpInst]

        if not moveOp1ToReg and moveOp2ToReg:
            movOp1ToReg = self.createChild(MOV, 
                                           self.op2.assemblyType,
                                           self.op2, 
                                           Register(self.op2.assemblyType, tempRegOP2))
            compareOnReg = self.createChild(CMP,
                                            self.asmbType, 
                                            self.op1, 
                                            Register(self.op2.assemblyType, tempRegOP2))
            return [movOp1ToReg, compareOnReg]

        if moveOp1ToReg and moveOp2ToReg:
            movOp1ToReg = self.createChild(MOV, 
                                           self.op1.assemblyType,
                                           self.op1, 
                                           Register(self.op1.assemblyType, tempRegOP1))
            movOp2ToReg = self.createChild(MOV,
                                           self.op2.assemblyType, 
                                           self.op2, 
                                           Register(self.op2.assemblyType, tempRegOP2))
            cmpInst = self.createChild(CMP, 
                                       self.asmbType,
                                       Register(self.op1.assemblyType, tempRegOP1), 
                                       Register(self.op2.assemblyType, tempRegOP2))
            return [movOp1ToReg, movOp2ToReg, cmpInst] 

        return [self]

    def emitCode(self) -> str:
        self.op1.assemblyType = self.asmbType
        self.op2.assemblyType = self.asmbType

        if self.asmbType.isDecimal():
            return f"\tcomi{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"
        else:
            return f"\tcmp{self.asmbType.getTypeName()}\t{self.op1.emitCode()}, {self.op2.emitCode()}\n"

    def print(self) -> str:
        return f"Compare({self.op1}, {self.op2})\n"

class JMP(AssemblerInstruction):
    def __init__(self, identifier: str, parentAST: AssemblyAST | None = None) -> None:
        self.identifier = identifier
        super().__init__(parentAST)

    def emitCode(self) -> str:
        return f"\tjmp \t.L{self.identifier}\n"

    def print(self) -> str:
        return f"Jump({self.identifier})\n"

class ConditionCode(enum.Enum):
    GREATER         = "g  "     # ZF = 0, SF = OF
    GREATER_EQUAL   = "ge "     # SF = 0
    LESS            = "l  "     # SF != 0
    LESS_EQUAL      = "le "     # ZF = 1, SF != OF
    EQUAL           = "e  "     # ZF = 1
    NOT_EQUAL       = "ne "     # ZF = 0
    ABOVE           = "a  "     # ZF = CF = 0
    ABOVE_EQUAL     = "ae "     # CF = 0
    BELOW           = "b  "     # CF = 1
    BELOW_EQUAL     = "be "     # ZF = CF = 1
    EVEN_PARITY     = "p  "     # PF = 1
    ODD_PARITY      = "p  "     # PF = 0

    @staticmethod
    def fromBinaryOperator(op: BinaryOperator, valueType: DeclaratorType) -> ConditionCode:
        def matchSignedOperations(op: BinaryOperator) -> ConditionCode:
            match op:
                case BinaryOperator.GREATER_THAN:       return ConditionCode.GREATER
                case BinaryOperator.GREATER_OR_EQUAL:   return ConditionCode.GREATER_EQUAL
                case BinaryOperator.LESS_THAN:          return ConditionCode.LESS
                case BinaryOperator.LESS_OR_EQUAL:      return ConditionCode.LESS_EQUAL
                case BinaryOperator.EQUAL:              return ConditionCode.EQUAL
                case BinaryOperator.NOT_EQUAL:          return ConditionCode.NOT_EQUAL
                case _:
                    raise ValueError(f"Invalid conversion from BinaryOperator {op} to ConditionCode for type {valueType}")
        
        def matchUnsignedOperations(op: BinaryOperator) -> ConditionCode:
            match op:
                case BinaryOperator.GREATER_THAN:       return ConditionCode.ABOVE
                case BinaryOperator.GREATER_OR_EQUAL:   return ConditionCode.ABOVE_EQUAL
                case BinaryOperator.LESS_THAN:          return ConditionCode.BELOW
                case BinaryOperator.LESS_OR_EQUAL:      return ConditionCode.BELOW_EQUAL
                case BinaryOperator.EQUAL:              return ConditionCode.EQUAL
                case BinaryOperator.NOT_EQUAL:          return ConditionCode.NOT_EQUAL
                case _:
                    raise ValueError(f"Invalid conversion from BinaryOperator {op} to ConditionCode for type {valueType}")

        # Signed operations apply only to signed types.
        if isinstance(valueType, BaseDeclaratorType) and valueType.baseType.isSignedInt():
            return matchSignedOperations(op)

        # The rest, unsigned, decimals, pointers, use unsigned.
        return matchUnsignedOperations(op)
        
class JMPIF(AssemblerInstruction):
    def __init__(self, condition: ConditionCode, identifier: str, parentAST: AssemblyAST | None = None) -> None:
        self.condition = condition
        self.identifier = identifier
        super().__init__(parentAST)

    def emitCode(self) -> str:
        return f"\tj{self.condition.value}\t.L{self.identifier}\n"

    def print(self) -> str:
        return f"JumpIf({self.condition.name}, {self.identifier})\n"

class SETIF(AssemblerInstruction):
    def __init__(self, condition: ConditionCode, op: AssemblerOperand, parentAST: AssemblyAST | None = None) -> None:
        self.condition = condition
        self.op = op
        super().__init__(parentAST)

    def secondPass(self):
        self.op = self.convertFromPseudo(self.op)

    def emitCode(self) -> str:
        if isinstance(self.op, Register):
            # Set instruction use BYTE registers as argument.
            self.op.assemblyType = AssemblyType.BYTE

        return f"\tset{self.condition.value}\t{self.op.emitCode()}\n"

    def print(self) -> str:
        return f"SetIf({self.condition.name}, {self.op})\n"

class LABEL(AssemblerInstruction):
    GENERAL_LABEL_COUNT: int = 0
    @staticmethod
    def generateLabel(posfix: str) -> str:
        label = f".{LABEL.GENERAL_LABEL_COUNT}.{posfix}"
        LABEL.GENERAL_LABEL_COUNT += 1
        return label

    def __init__(self, identifier: str, parentAST: AssemblyAST | None = None) -> None:
        self.identifier = identifier
        super().__init__(parentAST)

    def emitCode(self) -> str:
        return f"\n.L{self.identifier}:\n"

    def print(self) -> str:
        return f"Label({self.identifier})\n"

class COMMENT(AssemblerInstruction):
    def __init__(self, comment: str, parentAST: AssemblyAST | None = None) -> None:
        self.comment = comment
        super().__init__(parentAST)

    def emitCode(self) -> str:
        return f"\t# {self.comment}"

    def print(self) -> str:
        return f"\n# {self.comment}"

"""
OPERANDS
"""
class AssemblerOperand(AssemblyAST):
    def __init__(self, assemblyType: AssemblyType, parentAST: AssemblyAST | None = None) -> None:
        self.assemblyType = assemblyType
        super().__init__(parentAST)

    def firstPass(self):
        # Nothing to do when parsing.
        pass

    def secondPass(self):
        # Nothing to do in this case.
        pass

    def thirdPass(self):
        pass

    @abstractmethod
    def createCopy(self) -> AssemblerOperand:
        pass

    @abstractmethod
    def emitCode(self) -> str:
        pass

    @abstractmethod
    def print(self) -> str:
        pass

class REG(enum.Enum):
    AX = "rax"
    CX = "rcx"
    DX = "rdx"
    BX = "rbx"
    SI = "rsi"
    DI = "rdi"
    SP = "rsp"
    BP = "rbp"
    R8 = "r8"
    R9 = "r9"
    R10 = "r10"
    R11 = "r11"
    R12 = "r12"
    R13 = "r13"
    R14 = "r14"
    R15 = "r15"

    XMM0  = "xmm0"
    XMM1  = "xmm1"
    XMM2  = "xmm2"
    XMM3  = "xmm3"
    XMM4  = "xmm4"
    XMM5  = "xmm5"
    XMM6  = "xmm6"
    XMM7  = "xmm7"
    XMM8  = "xmm8"
    XMM9  = "xmm9"
    XMM10 = "xmm10"
    XMM11 = "xmm11"
    XMM12 = "xmm12"
    XMM13 = "xmm13"
    XMM14 = "xmm14"
    XMM15 = "xmm15"

    @staticmethod
    def getRegisterDict() -> dict[str, tuple]:
        return {
            "rax"   : ("eax",   "ax",   "al"),
            "rcx"   : ("ecx",   "cx",   "cl"),
            "rdx"   : ("edx",   "dx",   "dl"),
            "rbx"   : ("ebx",   "bx",   "bl"),
            "rsi"   : ("esi",   "si",   "sil"),
            "rdi"   : ("edi",   "di",   "dil"),
            "rsp"   : ("esp",   "sp",   "spl"),
            "rbp"   : ("ebp",   "bp",   "bpl"),
            "r8"    : ("r8d",   "r8w",  "r8b"),
            "r9"    : ("r9d",   "r9w",  "r9b"),
            "r10"   : ("r10d",  "r10w", "r10b"),
            "r11"   : ("r11d",  "r11w", "r11b"),
            "r12"   : ("r12d",  "r12w", "r12b"),
            "r13"   : ("r13d",  "r13w", "r13b"),
            "r14"   : ("r14d",  "r14w", "r14b"),
            "r15"   : ("r15d",  "r15w", "r15b"),
        }

    def get64Name(self) -> str:
        return self.value
    def get32Name(self) -> str:
        d = REG.getRegisterDict()
        return d[self.value][0] if self.value in d else self.value
    def get16Name(self) -> str:
        d = REG.getRegisterDict()
        return d[self.value][1] if self.value in d else self.value
    def get8Name(self) -> str:
        d = REG.getRegisterDict()
        return d[self.value][2] if self.value in d else self.value
    def isFloatRegister(self) -> bool:
        return self.name.startswith("XMM")

class Register(AssemblerOperand):
    def __init__(self, assemblyType: AssemblyType, reg: REG, parentAST: AssemblyAST | None = None) -> None:
        self.reg = reg
        super().__init__(assemblyType, parentAST)

    def createCopy(self) -> Register:
        return Register(self.assemblyType, self.reg, self.parent)

    def emitCode(self) -> str:
        regId = ""

        match self.assemblyType.baseType:
            case AssemblyBaseType.BYTE:         regId = self.reg.get8Name()
            case AssemblyBaseType.WORD:         regId = self.reg.get16Name()
            case AssemblyBaseType.LONGWORD:     regId = self.reg.get32Name()
            case AssemblyBaseType.QUADWORD:     regId = self.reg.get64Name()
            case AssemblyBaseType.DOUBLE:       regId = self.reg.get64Name()
            case AssemblyBaseType.FLOAT:        regId = self.reg.get32Name()
            case _:
                raise ValueError()
                
        return f"%{regId}"

    def print(self) -> str:
        regId = ""

        match self.assemblyType.baseType:
            case AssemblyBaseType.BYTE:         regId = self.reg.get8Name()
            case AssemblyBaseType.WORD:         regId = self.reg.get16Name()
            case AssemblyBaseType.LONGWORD:     regId = self.reg.get32Name()
            case AssemblyBaseType.QUADWORD:     regId = self.reg.get64Name()
            case AssemblyBaseType.DOUBLE:       regId = self.reg.get64Name()
            case AssemblyBaseType.FLOAT:        regId = self.reg.get32Name()
            case _:                             regId = "R??"
                
        return f"{regId.upper()}"

class Immediate(AssemblerOperand):
    def __init__(self, value: TACValue, parentAST: AssemblyAST | None = None) -> None:
        self.value = value
        self.valueStr = self.value.print()

        if isinstance(self.value.valueType, BaseDeclaratorType):
            baseType = self.value.valueType.baseType
        else:
            baseType = TypeSpecifier.ULONG

        # Immediate values are always taken as signed, so the unsigned values must be converted
        # to the 2's complement equivalent.
        self.intVal = int(self.valueStr)
        if baseType == TypeSpecifier.UINT:
            if self.intVal >= 0x80000000:
                self.intVal -= 0x100000000
                self.valueStr = str(self.intVal)
        elif baseType == TypeSpecifier.ULONG:
            if self.intVal >= 0x8000000000000000:
                self.intVal -= 0x10000000000000000
                self.valueStr = str(self.intVal)

        if not self.value.isConstant:
            raise ValueError("Cannot create an Immediate operand from a not constant value")
        super().__init__(AssemblyType.fromTAC(self.value.valueType), parentAST)

    def createCopy(self) -> Immediate:
        return Immediate(self.value, self.parent)

    def emitCode(self) -> str:
        return f"${self.valueStr}"

    def print(self) -> str:
        return f"Imm({self.value})"
    
    def isQuadImmediate(self) -> bool:
        return not(-0x80000000 <= self.intVal < 0x80000000)

# Stores a temporary variable from TAC into an imaginary register. Used for single variables.
class Pseudo(AssemblerOperand):
    def __init__(self, value: TACValue, parentAST: AssemblyAST | None = None) -> None:
        self.value = value
        self.name = value.print()
        if self.value.isConstant:
            raise ValueError("Cannot create a Pseudo operand from a constant value")
        super().__init__(AssemblyType.fromTAC(self.value.valueType), parentAST)

    def createCopy(self) -> AssemblerOperand:
        return Pseudo(self.value, self.parent)

    def emitCode(self) -> str:
        raise ValueError("Should not emit code for a Pseudo")

    def print(self) -> str:
        return f"Pseudo({self.value})"

# Stores a temporary variable from TAC into memory. Used for arrays.
class PseudoMemory(AssemblerOperand):
    def __init__(self, asmbType: AssemblyType, name: str, offset: int, parentAST: AssemblyAST | None = None) -> None:
        self.name = name
        self.offset = offset
        
        super().__init__(asmbType, parentAST)

    def createCopy(self) -> PseudoMemory:
        return PseudoMemory(self.assemblyType, self.name, self.offset, self.parent)

    def emitCode(self) -> str:
        raise ValueError("Should not emit code for a PseudoMemory")

    def print(self) -> str:
        return f"PseudoMemory({self.name})"

# An operand stored in memory. 
# Can be used to load variables from the stack. In this case, REG = BP, which points to the end of 
# the reserved stack and some integer offset. Is also used to load pointers.
class Memory(AssemblerOperand):
    # This value is always negative!
    STACK_OFFSET: int = 0
    stackVariables: dict[str, Memory] = {}

    def __init__(self, assemblyType: AssemblyType, register: REG, offset: int, 
                 parentAST: AssemblyAST | None = None) -> None:
        self.register = Register(AssemblyType.QUADWORD, register)
        self.offset = offset
        super().__init__(assemblyType, parentAST)

    def createCopy(self) -> Memory:
        return Memory(self.assemblyType, self.register.reg, self.offset, self.parent)

    def emitCode(self) -> str:
        if self.offset == 0:
            return f"({self.register.emitCode()})"
        else:
            return f"{self.offset}({self.register.emitCode()})"

    def print(self) -> str:
        return f"Memory({self.register}, {self.offset})"
    
    @staticmethod
    def restartStackVariables(startValue: int = 0):
        Memory.stackVariables.clear()
        Memory.STACK_OFFSET = startValue

    @staticmethod
    def convertToStackVariable(pseudo: Pseudo) -> Memory:
        if pseudo.name in Memory.stackVariables:
            return Memory.stackVariables[pseudo.name].createCopy()
        else:
            pseudoByteLen = pseudo.assemblyType.size
            alignment = pseudo.assemblyType.alignment
            Memory.STACK_OFFSET -= pseudoByteLen

            # ABI tells us to align 8 byte values to the next multiple of 8.
            # The same goes for 4 byte values, to next multiple of 4...
            # As an example, if I push an int and then a long:
            # - The int would be located at -4(%rsp), which is aligned (4 % 4 = 0).
            # - If the long is put at -12(%rsp) I would be failing the ABI (12 % 8 != 0). 
            #   It should go at -16(%rsp) and leave the bytes from -5 to -8 as padding.
            Memory.STACK_OFFSET = -(alignment * math.ceil((-Memory.STACK_OFFSET) / alignment))

            ret = Memory(pseudo.assemblyType, REG.BP, Memory.STACK_OFFSET, pseudo.parent)
            Memory.stackVariables[pseudo.name] = ret
            return ret

    @staticmethod
    def convertToStackMemory(pseudo: PseudoMemory) -> Memory:
        # An array loaded in stack has its base address (position 0). Parting from this zero position
        # you calculate the variable value by adding the "byte offset" to its "base address". 
        if pseudo.name not in Memory.stackVariables:
            # Calculate the base address.
            pseudoByteLen = pseudo.assemblyType.size
            alignment = pseudo.assemblyType.alignment
            Memory.STACK_OFFSET -= pseudoByteLen

            Memory.STACK_OFFSET = -(alignment * math.ceil((-Memory.STACK_OFFSET) / alignment))

            base = Memory(pseudo.assemblyType, REG.BP, Memory.STACK_OFFSET, pseudo.parent)
            Memory.stackVariables[pseudo.name] = base

        base = Memory.stackVariables[pseudo.name].createCopy()
        return Memory(pseudo.assemblyType, REG.BP, base.offset + pseudo.offset, pseudo.parent)

# Static or extern variables.
class Data(AssemblerOperand):
    def __init__(self, assemblyType: AssemblyType, identifier: str, offset: int,
                 parentAST: AssemblyAST | None = None) -> None:
        self.identifier = identifier
        self.offset = offset
        super().__init__(assemblyType, parentAST)

    def createCopy(self) -> Data:
        return Data(self.assemblyType, self.identifier, self.offset, self.parent)

    def emitCode(self) -> str:
        if self.offset == 0:
            return f"{self.identifier}(%rip)"
        elif self.offset > 0:
            return f"{self.identifier}+{self.offset}(%rip)"
        else:
            return f"{self.identifier}{self.offset}(%rip)"

    def print(self) -> str:
        return f"Data({self.identifier})"

# (regA, regB, offset: 1,2,4,8) -> regA + regB * offset
class Indexed(AssemblerOperand):
    def __init__(self, assemblyType: AssemblyType, base: Register, index: Register, offset: int,
                 parentAST: AssemblyAST | None = None) -> None:
        self.base = base
        self.index = index
        self.offset = offset
        super().__init__(assemblyType, parentAST)

    def createCopy(self) -> Indexed:
        return Indexed(self.assemblyType, self.base, self.index, self.offset, self.parent)

    def emitCode(self) -> str:
        if self.offset not in (1, 2, 4, 8):
            raise ValueError(f"Invalid offset in Indexed: {self.offset}")
        
        return f"({self.base.emitCode()}, {self.index.emitCode()}, {self.offset})"

    def print(self) -> str:
        return f"Indexed({self.base}, {self.index}, {self.offset})"
