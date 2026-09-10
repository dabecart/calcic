"""
assembler_calci.py

Converts the TAC into CALCI assembly language. 

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import enum
from typing import Type, TypeVar, Final

from src.TAC import *
from src.debug_info import *
from src.calcic_types import *
from src.builtin.builtin_functions_TAC import *
from src.calci32.types_calci32 import *

AssemblyT = TypeVar("AssemblyT", bound="AssemblyAST")

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
            # Check if the constant is defined inside the AssemblerStaticConstant.CONSTANTS array.
            found: AssemblerStaticConstant|None = AssemblerStaticConstant.CONSTANTS_MAP.get(tacValue.constantValue)
            if found is not None:
                return Data(AssemblyType.fromTAC(tacValue.valueType), found.identifier, 0, self)

            # In x64, decimal numbers are stored into memory and cannot be used as immediate.
            if tacValue.valueType.isDecimal():
                doubleConstant = AssemblerStaticConstant.newSimpleConstant(
                    tacValue.valueType, tacValue.print(), isGlobal=False)
                return Data(AssemblyType.fromTAC(tacValue.valueType), doubleConstant.identifier, 0, self)
            
            # Functions behave like Data constants.
            if isinstance(tacValue.valueType, FunctionDeclaratorType):
                return Data(AssemblyType.LONGWORD, tacValue.constantValue, 0, self)

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
            return [MOVE(asmbType, src, dst)]

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

            ret.append(MOVE(subMOVAsmbType, offsetedSrc, offsetedDst))

            offset += subMOVAsmbType.size

        return ret

    """
    Used to transfer bytes from a register when the number of bytes is not standard.
    """
    def copyBytesFromRegister(self, reg: REG, dst: AssemblerOperand, byteCount: int) -> list[AssemblerInstruction]:
        if not isinstance(dst, PseudoMemory):
            raise ValueError("Expected a PseudoMemory in copyBytesToRegister")

        instList: list[AssemblerInstruction] = []
        offset: int = 0

        # We will be shifting right by 8 bits. Load 8 into OP2.
        instList.append(
            MOVE(AssemblyType.BYTE, 
                self.fromTACValue(TACValue(True, TypeSpecifier.UCHAR.toBaseType(), "8")), 
                Register(AssemblyType.BYTE, REG.OP2)),
        )

        while offset < byteCount:
            dstCopy = dst.createCopy()
            dstCopy.offset = dst.offset + offset

            # Move the LSB to the destination.
            instList.append(MOVE(AssemblyType.BYTE, Register(AssemblyType.BYTE, reg), dstCopy))
            if offset < byteCount - 1:
                # Move reg into OP1 to shift it.
                instList.append(
                    MOVE(AssemblyType.WORD, Register(AssemblyType.WORD, reg), Register(AssemblyType.WORD, REG.OP1))
                )
                # Shift right and store in reg.
                instList.append(
                    ALU(ALUOP.SHR, AssemblyType.WORD, Register(AssemblyType.WORD, reg))
                )
            offset += 1

        return instList

    """
    Used to transfer bytes to a register when the number of bytes is not standard.
    """
    def copyBytesToRegister(self, src: AssemblerOperand, reg: REG, byteCount: int) -> list[AssemblerInstruction]:
        if not isinstance(src, PseudoMemory):
            raise ValueError("Expected a PseudoMemory in copyBytesToRegister")
        
        instList: list[AssemblerInstruction] = []
        offset: int = byteCount - 1

        # We will be shifting left by 8 bits. Load 8 into OP2.
        instList.append(
            MOVE(AssemblyType.BYTE, 
                self.fromTACValue(TACValue(True, TypeSpecifier.UCHAR.toBaseType(), "8")), 
                Register(AssemblyType.BYTE, REG.OP2)),
        )

        while offset >= 0:
            srcOp = src.createCopy()
            srcOp.offset = src.offset + offset

            # Move to the LSB of the register.
            instList.append(MOVE(AssemblyType.BYTE, srcOp, Register(AssemblyType.BYTE, reg)))
            if offset > 0:
                # Move reg into OP1 to shift it.
                instList.append(
                    MOVE(AssemblyType.WORD, Register(AssemblyType.WORD, reg), Register(AssemblyType.WORD, REG.OP1))
                )
                # Shift left and store in reg.
                instList.append(
                    ALU(ALUOP.SHL, AssemblyType.WORD, Register(AssemblyType.WORD, reg))
                )
            offset -= 1

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
                case TACStaticVariable():
                    if topLevel.isReadOnly:
                        # Create a 'section .rodata' constant.
                        AssemblerStaticConstant.newComplexConstant(
                            topLevel.valueType,
                            topLevel.identifier,
                            topLevel.isGlobal,
                            topLevel.initialization
                        )
                    else:
                        # Create a 'section .data' constant.
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
        ret = """
    # Set the reset vector.
    .section reset_v
    .long _start

    .section text
    .globl	_start
_start:
    # Initialize the stack.
    mov     $0x1000, %rsp
    mov     %rsp, %rsb

    # Call the main function.
    fun     main

    # Infinite loop.
_end:
    jmp     _end
    
"""

        for func in self.programDefs:
            ret += func.emitCode() + "\n"

        # Emit the constant section.
        ret += "\t.section\trodata\n"
        for constant in AssemblerStaticConstant.CONSTANTS:
            if constant.isGlobal:
                ret += f"\t.globl {constant.identifier}\n"
            ret += constant.emitCode() + "\n"
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
            ret +=  "\t.section bss\n"
            ret += f"\t.align {self.assemblyType.alignment}\n"
            ret += f"{self.identifier}:\n"
            ret += f"\t.zero {self.assemblyType.size}\n"
        else:
            ret +=  "\t.section data\n"
            ret += f"\t.align {self.assemblyType.alignment}\n"
            ret += f"{self.identifier}:\n"
            
            for const in self.initialization:
                if isinstance(const, ZeroPaddingInitializer):
                    ret += f"\t.zero {const.byteCount}\n"
                elif isinstance(const, PointerInitializer):
                    asmbType = AssemblyType.fromTAC(const.typeId)
                    
                    # Constant may be aliased with another name to avoid duplicated constants.
                    if const.constValue in AssemblerStaticConstant.CONSTANTS_MAP:
                        identifier = AssemblerStaticConstant.CONSTANTS_MAP[const.constValue].identifier
                    else:
                        identifier = const.constValue

                    if const.offset > 0:
                        ret += f"\t.{asmbType.getDataSectionName()} {identifier}+{const.offset}\n"
                    elif const.offset == 0:
                        ret += f"\t.{asmbType.getDataSectionName()} {identifier}\n"
                    else:
                        ret += f"\t.{asmbType.getDataSectionName()} {identifier}{const.offset}\n"
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
    # Stores the static constants of the program, without duplicates.
    CONSTANTS: list[AssemblerStaticConstant] = []
    # Key: identifier of the TACValue. Value: corresponding constant in the assembler.
    # Example: you may have str.0 = str.1 = "hi!". To save up space, both str.0 and str.1 will point
    # to the same constant.
    CONSTANTS_MAP: dict[str, AssemblerStaticConstant] = {}

    # Utility to generate constants during the assembly stage.
    @staticmethod
    def newSimpleConstant(valueType: DeclaratorType, initialization: str, isGlobal: bool,
                          alignment: int|None = None) -> AssemblerStaticConstant:
        asmbType = AssemblyType.fromTAC(valueType)
        initializationList: list[tuple[str, str]] = [(asmbType.getDataSectionName(), initialization)]

        for prevConst in AssemblerStaticConstant.CONSTANTS_MAP.values():
            if prevConst.valueType == valueType and \
               len(initializationList) == len(prevConst.initialization) and \
               all(a == b for a,b in zip(initializationList, prevConst.initialization)):
                return prevConst.copy()
        
        # Start with .L as it is hidden.
        identifier: str = f".Lconst{len(AssemblerStaticConstant.CONSTANTS_MAP)}"
        # By default the alignment is calculated from the type.
        if alignment is None:
            alignment = asmbType.alignment

        ret = AssemblerStaticConstant(valueType, identifier, isGlobal, initializationList, alignment)
        AssemblerStaticConstant.CONSTANTS_MAP[identifier] = ret
        AssemblerStaticConstant.CONSTANTS.append(ret)
        return ret

    # Utility to generate assembly code for C constants.
    @staticmethod
    def newComplexConstant(valueType: DeclaratorType, identifier: str, isGlobal: bool,
                           initialization: list[Constant], alignment: int|None = None) -> AssemblerStaticConstant:
        initializationList: list[tuple[str, str]] = []
        for const in initialization:
            if isinstance(const, ZeroPaddingInitializer):
                sectionName = 'zero'
                constValue = str(const.byteCount)
            else:
                sectionName = AssemblyType.fromTAC(const.typeId).getDataSectionName()
                constValue = const.constValue

            initializationList.append((sectionName, constValue))

        for prevConst in AssemblerStaticConstant.CONSTANTS_MAP.values():
            if prevConst.valueType == valueType and \
               len(initializationList) == len(prevConst.initialization) and \
               all(a == b for a,b in zip(initializationList, prevConst.initialization)):
                # Add a new key to the constants dictionary.
                prev = prevConst.copy()
                AssemblerStaticConstant.CONSTANTS_MAP[identifier] = prev
                return prev
        
        # By default the alignment is calculated from the type.
        if alignment is None:
            alignment = AssemblyType.fromTAC(valueType).alignment

        ret = AssemblerStaticConstant(valueType, identifier, isGlobal, initializationList, alignment)
        AssemblerStaticConstant.CONSTANTS_MAP[identifier] = ret
        AssemblerStaticConstant.CONSTANTS.append(ret)
        return ret

    # "initialization" is a list of (data section name, value)
    def __init__(self, valueType: DeclaratorType, identifier: str, isGlobal: bool, 
                 initialization: list[tuple[str, str]], 
                 alignment: int, parentAST: AssemblyAST | None = None) -> None:
        self.valueType = valueType
        self.identifier = identifier
        self.isGlobal = isGlobal
        self.initialization = initialization
        self.alignment = alignment
    
        super().__init__(parentAST)

    def copy(self) -> AssemblerStaticConstant:
        return AssemblerStaticConstant(
            self.valueType, self.identifier, self.isGlobal, self.initialization, self.alignment, self.parent)

    def firstPass(self):
        pass

    def secondPass(self):
        pass

    def thirdPass(self):
        pass

    # Converts byte objects grouped together into ascii or asciz.
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

        self.identifier: str = self.function.identifier
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
    - List of arguments stored in the stack.

    Each argument consists of its origical value (AssemblerOperand) and the type to use to transfer 
    data (AssemblyType). This is needed for structs.
    """
    def classifyArguments(self, arguments: list[TACValue], returnStoredInStack: bool) -> \
        tuple[
            list[tuple[AssemblerOperand, AssemblyType]], 
            list[tuple[AssemblerOperand, AssemblyType]]
        ]:
        
        regArgs: list[tuple[AssemblerOperand, AssemblyType]] = []
        stackArgs: list[tuple[AssemblerOperand, AssemblyType]] = []

        # Saved registers are R2 to R7, except when the return is stored in the stack. In that case, is from R3 to R7.
        regsAvailable: int = 5 if returnStoredInStack else 6

        for arg in arguments:
            destinationValue = self.fromTACValue(arg)
            argType = destinationValue.assemblyType

            if argType.isScalar():
                if regsAvailable > 0:
                    regArgs.append((destinationValue, argType))
                    regsAvailable -= 1
                else:
                    stackArgs.append((destinationValue, argType))
            else:
                # This is a structure argument.
                useStack: bool = True

                # If the first member is not memory, then the structure will use a mix of registers.
                # If it is, all following members will also be memory, and so the struct will go 
                # into the stack.
                if argType.members[0].classType != AssemblyClassType.MEMORY:
                    tentativeRegs: list[tuple[AssemblerOperand, AssemblyType]] = []
                    offset = 0
                    for member in argType.members:
                        # Create an offsetted operand with the original type of the member.
                        asmbMember = PseudoMemory(argType, arg.vbeName, offset, self)
                        offset += member.byteSize

                        # Now, this operand needs to be "moved" using the following type.
                        memberAsmbType = member.toAssemblyType()
                        if member.classType == AssemblyClassType.INTEGER:
                            tentativeRegs.append((asmbMember, memberAsmbType))
                        else:
                            raise ValueError()

                    # If there are enough free registers, add them to the register arrays.
                    if len(tentativeRegs) <= regsAvailable:
                        regArgs.extend(tentativeRegs)
                        regsAvailable -= len(tentativeRegs)
                        # Don't use the stack.
                        useStack = False
                
                if useStack:
                    offset = 0
                    for member in argType.members:
                        asmbMember = PseudoMemory(argType, arg.vbeName, offset, self)
                        offset += member.byteSize
                        stackArgs.append((asmbMember, member.toAssemblyType()))

        return (regArgs, stackArgs)
        
    """
    Returns:
    - List of arguments stored in the general registers.
    - True if the return value is returned as a variable in the stack.
    """
    def classifyReturnValue(self, retValue: TACValue) -> tuple[list[tuple[AssemblerOperand, AssemblyType]], bool]:
        asmbType = AssemblyType.fromTAC(retValue.valueType)

        if asmbType.isScalar():
            retAsmbVal = self.fromTACValue(retValue)
            return ([(retAsmbVal, retAsmbVal.assemblyType)], False)

        if asmbType.members[0].classType == AssemblyClassType.MEMORY:
            # The return is stored in the stack.
            return ([], True)
        
        # The return is stored in registers.
        regArgs: list[tuple[AssemblerOperand, AssemblyType]] = []
        offset: int = 0
        for member in asmbType.members:
            asmbMember = PseudoMemory(asmbType, retValue.vbeName, offset, self)
            offset += member.byteSize

            if member.classType == AssemblyClassType.INTEGER:
                regArgs.append((asmbMember, member.toAssemblyType()))
            else:
                raise ValueError()

        return (regArgs, False)

    def firstPass(self):
        REG_ORDER: Final[list[REG]] = [REG.R2, REG.R3, REG.R4, REG.R5, REG.R6, REG.R7]

        # Is the return value passed from the stack?
        self.returnInStack = \
            self.function.funDecl.typeId.returnDeclarator != TypeSpecifier.VOID.toBaseType() and \
            AssemblyType.fromTAC(self.function.funDecl.typeId.returnDeclarator). \
            members[0].classType == AssemblyClassType.MEMORY

        # If the function is variadic, dump all registers into a "Register Save Area".
        if self.function.isVariadic:
            raise ValueError()
        else:
            # Restart the stack to calculate the right offset.
            # If the return value is stored in the stack, its address will be the first stack variable.
            Memory.restartStackVariables(-8 if self.returnInStack else 0)

        # Create the variables used as arguments. These will be stored in the stack in the same 
        # order they appear in the argument list.
        tacArgs = [TACValue(False, arg.type, arg.name) for arg in self.function.arguments]
        # Classify them.
        self.regArgs, self.stackInputArgs = self.classifyArguments(tacArgs, self.returnInStack)

        if self.returnInStack:
            # Store the address of the return value, which is stored in R2 to the stack.
            self.createInst(MOVE, 
                            AssemblyType.LONGWORD,
                            Register(AssemblyType.LONGWORD, REG.R2), 
                            Memory(AssemblyType.LONGWORD, REG.RSB, -8))

        # The order of arguments is: R2 to R7 and then stack (pushed in reversed order).
        # Do not use R2 if the return value is stored in the stack.
        for (value, movAsmbType), reg in zip(self.regArgs, REG_ORDER[(1 if self.returnInStack else 0):]):
            if self.function.isVariadic:
                raise ValueError()

            if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                self.instructions.extend(self.copyBytesFromRegister(reg, value, movAsmbType.size))
            else:
                self.createInst(MOVE, movAsmbType, Register(value.assemblyType, reg), value)

        # The arguments in the stack start at Stack(8). From then on, add in groups of four.
        stackOffset = 8
        for (value, movAsmbType) in self.stackInputArgs:
            if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                self.instructions.extend(
                    self.copyBytes(Memory(AssemblyType.WORD, REG.RSB, stackOffset), value, movAsmbType)
                )
            else:
                # Move to the stack.
                self.createInst(MOVE, 
                                movAsmbType, 
                                Memory(value.assemblyType, REG.RSB, stackOffset), 
                                value)
            # Increment the stack offset.
            stackOffset += 4

        # Convert the function's TAC instructions into assembler instructions.
        for inst in self.function.instructions:
            if isinstance(inst, TACBuiltInFunction):
                self.convertBuiltInTAC(inst)
                continue

            match inst:
                case TACUnary():
                    exp: AssemblerOperand = self.fromTACValue(inst.exp)
                    dest: AssemblerOperand = self.fromTACValue(inst.result)

                    # Move exp to OP1.
                    self.createInst(MOVE, exp.assemblyType, exp, Register(exp.assemblyType, REG.OP1))

                    match inst.operator:
                        case UnaryOperator.NOT:
                            # !(x) is the same as x == 0.
                            self.createInst(CLR, exp.assemblyType, Register(exp.assemblyType, REG.OP2))
                            self.createInst(ALU, ALUOP.CMP, exp.assemblyType)
                            self.createInst(SET, ConditionCode.EQUAL, dest)

                        case UnaryOperator.NEGATION:
                            if inst.result.valueType.isDecimal():
                                # Flip the sign bit by using an XOR operation.
                                negZero = AssemblerStaticConstant.newSimpleConstant(
                                    TypeSpecifier.FLOAT.toBaseType(), "-0.0", False, 4)
                                negZeroData = Data(AssemblyType.LONGWORD, negZero.identifier, 0, self)
                                self.createInst(MOVE, negZeroData.assemblyType, negZeroData, Register(negZeroData.assemblyType, REG.OP2))
                                self.createInst(ALU, ALUOP.XOR, dest.assemblyType, dest)

                            else:
                                # Use the negate ALU operation.
                                self.createInst(ALU, ALUOP.NEG, dest.assemblyType, dest)

                        case UnaryOperator.INCREMENT:
                            if inst.result.valueType.isDecimal():
                                raise ValueError()
                            else:
                                self.createInst(ALU, ALUOP.INC, dest.assemblyType, dest)

                        case UnaryOperator.DECREMENT:
                            if inst.result.valueType.isDecimal():
                                raise ValueError()
                            else:
                                self.createInst(ALU, ALUOP.DEC, dest.assemblyType, dest)

                        case _:
                            raise ValueError(f"Invalid Unary Operation: {inst.operator}")

                case TACBinary():
                    exp1: AssemblerOperand = self.fromTACValue(inst.exp1)
                    exp2: AssemblerOperand = self.fromTACValue(inst.exp2)
                    dest: AssemblerOperand = self.fromTACValue(inst.result)

                    # Move exp1 to OP1.
                    self.createInst(MOVE, exp1.assemblyType, exp1, Register(exp1.assemblyType, REG.OP1))
                    # Move exp2 to OP2.
                    self.createInst(MOVE, exp2.assemblyType, exp2, Register(exp2.assemblyType, REG.OP2))

                    match inst.operator:
                        case BinaryOperator.MODULUS:
                            if inst.result.valueType.isDecimal():
                                raise ValueError()
                            elif inst.result.valueType.isSigned():
                                self.createInst(ALU, ALUOP.SDIV, dest.assemblyType)
                            else:
                                self.createInst(ALU, ALUOP.UDIV, dest.assemblyType)

                            # The modulus is stored in RESH, save it to dest.
                            self.createInst(MOVE, dest.assemblyType, Register(dest.assemblyType, REG.RESH), dest)

                        case BinaryOperator.GREATER_THAN | BinaryOperator.GREATER_OR_EQUAL | \
                             BinaryOperator.LESS_THAN    | BinaryOperator.LESS_OR_EQUAL    | \
                             BinaryOperator.EQUAL        | BinaryOperator.NOT_EQUAL:

                            if inst.result.valueType.isDecimal():
                                raise ValueError()
                            else:
                                self.createInst(ALU, ALUOP.CMP, dest.assemblyType)
                                self.createInst(SET, 
                                                ConditionCode.fromBinaryOperator(inst.operator, inst.exp1.valueType), 
                                                dest)

                        case _:
                            self.createInst(ALU, 
                                            ALUOP.fromBinaryOperator(inst.operator, inst.exp1.valueType), 
                                            dest.assemblyType, dest)

                case TACJump():
                    self.createInst(JMP, inst.target)

                case TACJumpIfValue():
                    cond: AssemblerOperand = self.fromTACValue(inst.condition)
                    val: AssemblerOperand = self.fromTACValue(inst.value)

                    # Move condition to OP1.
                    self.createInst(MOVE, cond.assemblyType, cond, Register(cond.assemblyType, REG.OP1))
                    # Move value to OP2.
                    self.createInst(MOVE, val.assemblyType, val, Register(val.assemblyType, REG.OP2))

                    self.createInst(ALU, ALUOP.CMP, dest.assemblyType)

                    if inst.condition.valueType.isDecimal():
                        raise ValueError()
                    else:
                        self.createInst(BRANCH, ConditionCode.EQUAL, inst.target)

                case TACJumpIfZero():
                    cond: AssemblerOperand = self.fromTACValue(inst.condition)

                    # Move condition to OP1.
                    self.createInst(MOVE, cond.assemblyType, cond, Register(cond.assemblyType, REG.OP1))
                    # Clear OP2.
                    self.createInst(CLR, AssemblyType.LONGWORD, Register(AssemblyType.LONGWORD, REG.OP2))

                    self.createInst(ALU, ALUOP.CMP, dest.assemblyType)

                    if inst.condition.valueType.isDecimal():
                        raise ValueError()
                    else:
                        self.createInst(BRANCH, ConditionCode.EQUAL, inst.target)

                case TACJumpIfNotZero():
                    cond: AssemblerOperand = self.fromTACValue(inst.condition)

                    # Move condition to OP1.
                    self.createInst(MOVE, cond.assemblyType, cond, Register(cond.assemblyType, REG.OP1))
                    # Clear OP2.
                    self.createInst(CLR, AssemblyType.LONGWORD, Register(AssemblyType.LONGWORD, REG.OP2))

                    self.createInst(ALU, ALUOP.CMP, dest.assemblyType)

                    if inst.condition.valueType.isDecimal():
                        raise ValueError()
                    else:
                        self.createInst(BRANCH, ConditionCode.NOT_EQUAL, inst.target)

                case TACCopy():
                    src = self.fromTACValue(inst.src)
                    dst = self.fromTACValue(inst.dst)
                    movInstructions = self.copyBytes(src, dst, AssemblyType.fromTAC(inst.src.valueType))
                    self.instructions.extend(movInstructions)

                case TACLoad():
                    self.createInst(MOVE,
                                    AssemblyType.LONGWORD, 
                                    self.fromTACValue(inst.src),
                                    Register(AssemblyType.LONGWORD, REG.R0))

                    dst = self.fromTACValue(inst.dst)
                    src = Memory(dst.assemblyType, REG.R0, 0)
                    movInstructions = self.copyBytes(src, dst, dst.assemblyType)
                    self.instructions.extend(movInstructions)

                case TACStore():
                    self.createInst(MOVE, 
                                    AssemblyType.LONGWORD,
                                    self.fromTACValue(inst.dst), 
                                    Register(AssemblyType.LONGWORD, REG.R0))
                    
                    src = self.fromTACValue(inst.src)
                    dst = Memory(src.assemblyType, REG.R0, 0)
                    movInstructions = self.copyBytes(src, dst, src.assemblyType)
                    self.instructions.extend(movInstructions)

                case TACGetAddress():
                    self.createInst(OFS, self.fromTACValue(inst.src), self.fromTACValue(inst.dst))

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
                    pointer = self.fromTACValue(inst.pointer)
                    index = self.fromTACValue(inst.index)

                    # Move the base address to R0.
                    self.createInst(MOVE, 
                                    AssemblyType.LONGWORD, 
                                    pointer, 
                                    Register(AssemblyType.LONGWORD, REG.R0))

                    if isinstance(index, Immediate):
                        # The index is a constant and so is the scale. Calculate it during compilation.
                        byteOffset = int(index.value.constantValue) * inst.scale
                        self.createInst(OFS, 
                                        Memory(AssemblyType.LONGWORD, REG.R0, byteOffset), 
                                        self.fromTACValue(inst.dst))
                    else:
                        # The index is a variable, load it into a register.
                        self.createInst(MOVE, 
                                        index.assemblyType, 
                                        index, 
                                        Register(index.assemblyType, REG.R1))

                        self.createInst(OFS, 
                            Indexed(
                                pointer.assemblyType, 
                                Register(AssemblyType.LONGWORD, REG.R0), 
                                Register(index.assemblyType, REG.R1), 
                                inst.scale
                            ), self.fromTACValue(inst.dst))

                case TACLabel():
                    self.createInst(LABEL, inst.identifier)

                case TACSignExtend():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)

                    if isinstance(exp, Register):
                        exp.signExtend = True
                        self.createInst(MOVE, result.assemblyType, exp, result)
                    else:
                        self.createInst(MOVE, exp.assemblyType, exp, Register(exp.assemblyType, REG.R0))
                        self.createInst(MOVE, 
                            result.assemblyType, Register(result.assemblyType, REG.R0, signExtend=True), result)

                case TACTruncate():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)
                    self.createInst(MOVE, result.assemblyType, exp, result)

                case TACZeroExtend():
                    exp = self.fromTACValue(inst.exp)
                    result = self.fromTACValue(inst.result)

                    if isinstance(exp, Register):
                        exp.signExtend = False
                        self.createInst(MOVE, result.assemblyType, exp, result)
                    else:
                        self.createInst(MOVE, exp.assemblyType, exp, Register(exp.assemblyType, REG.R0))
                        self.createInst(MOVE, 
                            result.assemblyType, Register(result.assemblyType, REG.R0, signExtend=False), result)

                case TACFunctionCall() | TACIndirectFunctionCall():
                    returnIntRegs: list[tuple[AssemblerOperand, AssemblyType]] = []
                    self.returnInStack: bool = False

                    # Classify the return value.
                    if inst.returnType != TypeSpecifier.VOID.toBaseType():
                        returnIntRegs, self.returnInStack = self.classifyReturnValue(inst.result)

                    if self.returnInStack:
                        # When the value is returned in the stack, the return value's space is 
                        # reserved on the caller. It's address is stored in R2.
                        retAsmbVal = self.fromTACValue(inst.result)
                        self.createInst(OFS, retAsmbVal, Register(AssemblyType.LONGWORD, REG.R2))

                    # Split between arguments stored in registers and arguments stored in the stack.
                    intRegisterArgs, stackArgs = self.classifyArguments(inst.arguments, self.returnInStack)

                    # The stack needs to be padded so that the function arguments start from a multiple of 16. 
                    # Each cell in the stack is 4 bytes long.
                    if len(stackArgs) % 4 == 0:
                        stackPadding = 0
                    else:
                        # Pad so that it starts from a multiple of 16.
                        stackPadding = 4 - (len(stackArgs) % 4)

                    if stackPadding != 0:
                        # Allocate stack.
                        offs = self.fromTACValue(TACValue(True, TypeSpecifier.INT.toBaseType(), str(stackPadding)))
                        self.createInst(MOVE, 
                                        AssemblyType.LONGWORD, 
                                        Register(AssemblyType.LONGWORD, REG.RSP), 
                                        Register(AssemblyType.LONGWORD, REG.OP1))
                        self.createInst(MOVE, 
                                        offs.assemblyType, offs, Register(offs.assemblyType, REG.OP2))
                        self.createInst(ALU, ALUOP.SUB, AssemblyType.LONGWORD, Register(offs.assemblyType, REG.RSP))

                    # Pass the function's arguments to registers and then the stack.
                    # The order of arguments is R2 to R7 and then stack (pushed in reversed order).
                    # Skip R2 if the return value is saved in the stack, it contains the address of the return value.
                    for (value, movAsmbType), reg in zip(intRegisterArgs, REG_ORDER[(1 if self.returnInStack else 0):]):
                        if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                            # There may be part of a struct/union returned in a register whose byte 
                            # size is not standard, i.e. 3, 5, 6 or 7 bytes.
                            self.instructions.extend(self.copyBytesToRegister(value, reg, movAsmbType.size))
                        else:
                            self.instructions.extend(
                                self.copyBytes(value, Register(value.assemblyType, reg), movAsmbType)
                            )

                    # Push the values in reversed order.
                    for (value, movAsmbType) in reversed(stackArgs):
                        if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                            # There may be part of a struct/union returned in a register whose byte size is not 
                            # standard. For this case, allocate as 4 bytes in the stack. 
                            offs = self.fromTACValue(TACValue(True, TypeSpecifier.LONG.toBaseType(), "4"))
                            self.createInst(MOVE, 
                                            AssemblyType.LONGWORD, 
                                            Register(AssemblyType.LONGWORD, REG.RSP), 
                                            Register(AssemblyType.LONGWORD, REG.OP1))
                            self.createInst(MOVE, 
                                            offs.assemblyType, offs, Register(offs.assemblyType, REG.OP2))
                            self.createInst(ALU, ALUOP.SUB, AssemblyType.LONGWORD, Register(offs.assemblyType, REG.RSP))
                            self.instructions.extend(self.copyBytes(value, Memory(value.assemblyType, REG.RSP, 0), movAsmbType))

                        elif value.assemblyType == AssemblyType.QUADWORD:
                            raise ValueError()
                        elif isinstance(value, (Register, Immediate)) or value.assemblyType == AssemblyType.LONGWORD:
                            # This value can be directly pushed as it is 4 bytes.
                            self.createInst(PSH, value)
                        else:
                            # This value is under 4 bytes so it must be transferred to a register and then pushed. 
                            self.createInst(MOVE, movAsmbType, value, Register(movAsmbType, REG.R0))
                            self.createInst(PSH, Register(AssemblyType.LONGWORD, REG.R0))
                    
                    if inst.isVariadic:
                        raise ValueError()

                    # Emit the call instruction.
                    if isinstance(inst, TACIndirectFunctionCall):
                        funcAddrs = self.fromTACValue(inst.funcAddress)
                        self.createInst(MOVE,
                                        AssemblyType.LONGWORD,
                                        funcAddrs, 
                                        Register(funcAddrs.assemblyType, REG.R0))
                        self.createInst(FUN, Register(funcAddrs.assemblyType, REG.R0))
                    else:
                        self.createInst(FUN, inst.identifier)

                    # Readjust the stack pointer.
                    deallocBytes = 4 * len(stackArgs) + stackPadding
                    if deallocBytes != 0:
                        # Deallocate stack.
                        offs = self.fromTACValue(TACValue(True, TypeSpecifier.INT.toBaseType(), str(deallocBytes)))
                        self.createInst(MOVE, 
                                        AssemblyType.LONGWORD, 
                                        Register(AssemblyType.LONGWORD, REG.RSP), 
                                        Register(AssemblyType.LONGWORD, REG.OP1))
                        self.createInst(MOVE, 
                                        offs.assemblyType, offs, Register(offs.assemblyType, REG.OP2))
                        self.createInst(ALU, ALUOP.ADD, AssemblyType.LONGWORD, Register(offs.assemblyType, REG.RSP))

                    # Retrieve the return value if function is not void and the return value is not 
                    # stored in the stack.
                    if inst.returnType != TypeSpecifier.VOID.toBaseType() and not self.returnInStack:
                        RETURN_REGS: list[REG] = [REG.R0, REG.R1]

                        for (value, movAsmbType), reg in zip(returnIntRegs, RETURN_REGS):
                            if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                                self.instructions.extend(self.copyBytesFromRegister(reg, value, movAsmbType.size))
                            else:
                                self.createInst(MOVE, movAsmbType, Register(value.assemblyType, reg), value)

                case TACReturn():
                    if inst.result.valueType != TypeSpecifier.VOID.toBaseType():
                        self.intRegArgs, self.returnInStack = self.classifyReturnValue(inst.result)
                        if self.returnInStack:
                            # Get the address of where the return value should be stored.
                            self.createInst(MOVE, 
                                            AssemblyType.LONGWORD,
                                            Memory(AssemblyType.LONGWORD, REG.RSB, -8), 
                                            Register(AssemblyType.LONGWORD, REG.R0))
                            # Transfer the return value to this address.
                            transferInsts = self.copyBytes(self.fromTACValue(inst.result), 
                                                          Memory(AssemblyType.QUADWORD, REG.R0, 0), 
                                                          AssemblyType.fromTAC(inst.result.valueType))
                            self.instructions.extend(transferInsts)
                        else:
                            RETURN_REGS = [REG.R0, REG.R1]

                            for (value, movAsmbType), reg in zip(self.intRegArgs, RETURN_REGS):
                                if movAsmbType.baseType == AssemblyBaseType.BYTEARRAY:
                                    self.instructions.extend(self.copyBytesToRegister(value, reg, movAsmbType.size))
                                else:
                                    self.createInst(MOVE, movAsmbType, value, Register(value.assemblyType, reg))

                    self.createInst(RET)

                case _:
                    raise ValueError(f"Unexpected statement {inst} when parsing an AssemblerFunction")

    def convertBuiltInTAC(self, tac: TACBuiltInFunction):
        pass

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
            # Allocate the stack.
            offs = self.fromTACValue(TACValue(True, TypeSpecifier.INT.toBaseType(), str(functionStackAlloc)))
            moveStackToALU = MOVE(AssemblyType.LONGWORD, Register(AssemblyType.LONGWORD, REG.RSP), Register(AssemblyType.LONGWORD, REG.OP1))
            moveOffsToALU = MOVE(offs.assemblyType, offs, Register(offs.assemblyType, REG.OP2))
            stackAllocInstruction = ALU(ALUOP.SUB, AssemblyType.LONGWORD, Register(offs.assemblyType, REG.RSP))
            newInstructions.extend((moveStackToALU, moveOffsToALU, stackAllocInstruction))

        # Fix the instructions.
        for inst in self.instructions:
            newInstructions.extend(inst.thirdPass())

        self.instructions = newInstructions

    def emitCode(self) -> str:
        ret = ""
        if self.function.isGlobal:
            ret =  f"\t.globl {self.identifier}\n"
        
        ret += "\t.section text\n"
        ret += f"{self.identifier}:\n"
        ret += f"\tpsh\t%rsb\n"
        ret += f"\tmov\t%rsp, %rsb\n"

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

class MOVE(AssemblerInstruction):
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
        if (isinstance(self.src, (Immediate, Memory, Data)) and isinstance(self.dst, (Memory, Data))):
            # - MOV cannot have two memory addresses. 
            # - Cannot MOV a constant into memory.
            # Save the src into a temporary register and then pass it to the dst. 
            movToReg = self.createChild(MOVE, self.asmbType, 
                                        self.src, 
                                        Register(self.asmbType, REG.R0))
            movFromReg = self.createChild(MOVE, self.asmbType,
                                        Register(self.asmbType, REG.R0), 
                                        self.dst)
            return [movToReg, movFromReg]

        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.src.assemblyType = self.asmbType
        self.dst.assemblyType = self.asmbType

        # Switch between STO and MOV instructions.
        inst: str
        if isinstance(self.src, Register) and isinstance(self.dst, (Memory, Data)):
            inst = "sto"
        elif isinstance(self.src, (Register, Immediate, Memory, Data)) and isinstance(self.dst, Register):
            inst = "mov"
        else:
            raise ValueError(f"Cannot emit code for this MOV instruction: {self.print()}")

        if self.asmbType == AssemblyType.QUADWORD:
            # Use the stoq/movq instruction for 8 byte values.
            inst += "q"

        return f"\t{inst}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"Mov({self.src}, {self.dst})\n"

class ALUOP(enum.Enum):
    ADD     = enum.auto()
    SUB     = enum.auto()
    SMUL    = enum.auto()
    UMUL    = enum.auto()
    SDIV    = enum.auto()
    UDIV    = enum.auto()
    NEG     = enum.auto()
    AND     = enum.auto()
    OR      = enum.auto()
    XOR     = enum.auto()
    NOT     = enum.auto()
    SHL     = enum.auto()
    SHR     = enum.auto()
    CMP     = enum.auto()
    INC     = enum.auto()
    DEC     = enum.auto()

    @staticmethod
    def fromBinaryOperator(op: BinaryOperator, valueType: DeclaratorType) -> ALUOP:
        if valueType.isDecimal():
            raise ValueError()

        match op:
            case BinaryOperator.MULTIPLICATION:
                if isinstance(valueType, BaseDeclaratorType) and valueType.baseType.isSignedInt():
                    return ALUOP.SMUL
                else:
                    return ALUOP.UMUL

            case BinaryOperator.DIVISION | BinaryOperator.MODULUS:
                if isinstance(valueType, BaseDeclaratorType) and valueType.baseType.isSignedInt():
                    return ALUOP.SDIV
                else:
                    return ALUOP.UDIV

            case BinaryOperator.SUM:
                return ALUOP.ADD

            case BinaryOperator.SUBTRACT:
                return ALUOP.SUB

            case BinaryOperator.BITWISE_LEFT_SHIFT:
                return ALUOP.SHL

            case BinaryOperator.BITWISE_RIGHT_SHIFT:
                return ALUOP.SHR

            case BinaryOperator.BITWISE_AND:
                return ALUOP.AND

            case BinaryOperator.BITWISE_XOR:
                return ALUOP.XOR

            case BinaryOperator.BITWISE_OR:
                return ALUOP.OR

            case _:
                raise ValueError(f"There's no ALU operation for operation {op}")

class ALU(AssemblerInstruction):
    def __init__(self, operation: ALUOP, asmbType: AssemblyType, 
                 dstLow: AssemblerOperand | None = None, dstHigh: AssemblerOperand | None = None,
                 parentAST: AssemblyAST | None = None) -> None:
        self.op = operation
        self.asmbType = asmbType
        self.dstLow = dstLow
        self.dstHigh = dstHigh

        if dstLow is None and dstHigh is not None:
            raise ValueError("Cannot create an ALU operation with just the RESH result.")

        super().__init__(parentAST)

    def secondPass(self):
        self.dstLow = self.convertFromPseudo(self.dstLow)
        self.dstHigh = self.convertFromPseudo(self.dstHigh)

    def thirdPass(self) -> list[AssemblerInstruction]:
        moveLowToReg = isinstance(self.dstLow, (Memory, Data))
        moveHighToReg = isinstance(self.dstHigh, (Memory, Data))

        if moveLowToReg and moveHighToReg:
            aluOp = self.createChild(ALU, self.op, self.asmbType, Register(self.asmbType, REG.R0), Register(self.asmbType, REG.R1))
            moveLow = self.createChild(MOVE, self.asmbType, Register(self.asmbType, REG.R0), self.dstLow)
            moveHigh = self.createChild(MOVE, self.asmbType, Register(self.asmbType, REG.R1), self.dstHigh)
            return [aluOp, moveLow, moveHigh]

        if moveLowToReg:
            aluOp = self.createChild(ALU, self.op, self.asmbType, Register(self.asmbType, REG.R0), self.dstHigh)
            moveLow = self.createChild(MOVE, self.asmbType, Register(self.asmbType, REG.R0), self.dstLow)
            return [aluOp, moveLow]

        if moveHighToReg:
            aluOp = self.createChild(ALU, self.op, self.asmbType, self.dstLow, Register(self.asmbType, REG.R1))
            moveHigh = self.createChild(MOVE, self.asmbType, Register(self.asmbType, REG.R1), self.dstHigh)
            return [aluOp, moveHigh]

        return [self]

    def emitCode(self) -> str:
        ret = f"\t{self.op.name.lower()}"

        if self.dstLow is not None:
            self.dstLow.assemblyType = self.asmbType
            ret += f"\t{self.dstLow.emitCode()}"

        if self.dstHigh is not None:
            self.dstHigh.assemblyType = self.asmbType
            ret += f", {self.dstHigh.emitCode()}"

        ret += "\n"
        return ret

    def print(self) -> str:
        return f"{self.op.name}({self.dstLow}, {self.dstHigh})\n"

# Clear.
class CLR(AssemblerInstruction):
    def __init__(self, asmbType: AssemblyType, dst: AssemblerOperand,
                 parentAST: AssemblyAST | None = None) -> None:
        self.asmbType = asmbType
        self.dst = dst

        super().__init__(parentAST)

    def secondPass(self):
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.dst, (Data, Memory)):
            clrInst = self.createChild(CLR, Register(self.asmbType, REG.R0))
            moveInst = self.createChild(MOVE, self.asmbType, Register(self.asmbType, REG.R0), self.dst)
            return [clrInst, moveInst]

        return [self]

    def emitCode(self) -> str:
        # Set the type to that of the instruction.
        self.dst.assemblyType = self.asmbType

        return f"\tclr\t{self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"CLR({self.dst})\n"

class ConditionCode(enum.Enum):
    EQUAL                   = "eq "
    NOT_EQUAL               = "neq"
    GREATER_SIGNED          = "gs "
    GREATER_EQUAL_SIGNED    = "ges"
    LESS_SIGNED             = "ls "
    LESS_EQUAL_SIGNED       = "les"
    GREATER_UNSIGNED        = "gu "
    GREATER_EQUAL_UNSIGNED  = "geu"
    LESS_UNSIGNED           = "lu "
    LESS_EQUAL_UNSIGNED     = "leu"

    @staticmethod
    def fromBinaryOperator(op: BinaryOperator, valueType: DeclaratorType) -> ConditionCode:
        def matchSignedOperations(op: BinaryOperator) -> ConditionCode:
            match op:
                case BinaryOperator.GREATER_THAN:       return ConditionCode.GREATER_SIGNED
                case BinaryOperator.GREATER_OR_EQUAL:   return ConditionCode.GREATER_EQUAL_SIGNED
                case BinaryOperator.LESS_THAN:          return ConditionCode.LESS_SIGNED
                case BinaryOperator.LESS_OR_EQUAL:      return ConditionCode.LESS_EQUAL_SIGNED
                case BinaryOperator.EQUAL:              return ConditionCode.EQUAL
                case BinaryOperator.NOT_EQUAL:          return ConditionCode.NOT_EQUAL
                case _:
                    raise ValueError(f"Invalid conversion from BinaryOperator {op} to ConditionCode for type {valueType}")
        
        def matchUnsignedOperations(op: BinaryOperator) -> ConditionCode:
            match op:
                case BinaryOperator.GREATER_THAN:       return ConditionCode.GREATER_UNSIGNED
                case BinaryOperator.GREATER_OR_EQUAL:   return ConditionCode.GREATER_EQUAL_UNSIGNED
                case BinaryOperator.LESS_THAN:          return ConditionCode.LESS_UNSIGNED
                case BinaryOperator.LESS_OR_EQUAL:      return ConditionCode.LESS_EQUAL_UNSIGNED
                case BinaryOperator.EQUAL:              return ConditionCode.EQUAL
                case BinaryOperator.NOT_EQUAL:          return ConditionCode.NOT_EQUAL
                case _:
                    raise ValueError(f"Invalid conversion from BinaryOperator {op} to ConditionCode for type {valueType}")

        # Signed operations apply only to signed types.
        if isinstance(valueType, BaseDeclaratorType) and valueType.baseType.isSignedInt():
            return matchSignedOperations(op)

        # The rest, unsigned, decimals, pointers, use unsigned.
        return matchUnsignedOperations(op)
        
class BRANCH(AssemblerInstruction):
    def __init__(self, condition: ConditionCode, identifier: str, parentAST: AssemblyAST | None = None) -> None:
        self.condition = condition
        self.identifier = identifier
        super().__init__(parentAST)

    def emitCode(self) -> str:
        return f"\tb{self.condition.value}\tL{self.identifier}\n"

    def print(self) -> str:
        return f"Branch({self.condition.name}, {self.identifier})\n"

class SET(AssemblerInstruction):
    def __init__(self, condition: ConditionCode, dst: AssemblerOperand, parentAST: AssemblyAST | None = None) -> None:
        self.condition = condition
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.dst, (Data, Memory)):
            setInst = self.createChild(SET, self.condition, Register(self.dst.assemblyType, REG.R0))
            moveInst = self.createChild(MOVE, self.dst.assemblyType, Register(self.dst.assemblyType, REG.R0), self.dst)
            return [setInst, moveInst]

        return [self]


    def emitCode(self) -> str:
        return f"\ts{self.condition.value}\t{self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"Set({self.condition.name}, {self.dst})\n"

class JMP(AssemblerInstruction):
    def __init__(self, identifier: str, parentAST: AssemblyAST | None = None) -> None:
        self.identifier = identifier
        super().__init__(parentAST)

    def emitCode(self) -> str:
        return f"\tjmp \tL{self.identifier}\n"

    def print(self) -> str:
        return f"Jump({self.identifier})\n"

# Offset. 
class OFS(AssemblerInstruction):
    def __init__(self, src: AssemblerOperand, dst: AssemblerOperand, 
                 parentAST: AssemblyAST | None = None) -> None:
        
        if dst.assemblyType != AssemblyType.LONGWORD:
            raise ValueError("OFS expects a LONGWORD as destination")

        self.src = src
        self.dst = dst
        super().__init__(parentAST)

    def secondPass(self):
        self.src = self.convertFromPseudo(self.src)
        self.dst = self.convertFromPseudo(self.dst)

    def thirdPass(self) -> list[AssemblerInstruction]:
        moveSrcToReg = isinstance(self.src, (Memory, Data))
        moveDstToReg = isinstance(self.dst, (Memory, Data))

        if moveSrcToReg and moveDstToReg:
            moveSrc = self.createChild(MOVE, self.src.assemblyType, self.src, Register(self.src.assemblyType, REG.R0))
            ofsOp = self.createChild(OFS, Register(self.src.assemblyType, REG.R0), Register(self.dst.assemblyType, REG.R1))
            moveDst = self.createChild(MOVE, self.dst.assemblyType, Register(self.dst.assemblyType, REG.R1), self.dst)
            return [moveSrc, ofsOp, moveDst]

        if moveSrcToReg:
            moveSrc = self.createChild(MOVE, self.src.assemblyType, self.src, Register(self.src.assemblyType, REG.R0))
            ofsOp = self.createChild(OFS, Register(self.src.assemblyType, REG.R0), self.dst)
            return [moveSrc, ofsOp]

        if moveDstToReg:
            ofsOp = self.createChild(OFS, self.src, Register(self.dst.assemblyType, REG.R1))
            moveDst = self.createChild(MOVE, self.dst.assemblyType, Register(self.dst.assemblyType, REG.R1), self.dst)
            return [ofsOp, moveDst]

        return [self]

    def emitCode(self) -> str:
        if isinstance(self.src, Indexed):
            if self.src.scale in (2, 4, 8, 16, 32, 64, 128):
                return f"\tofs{self.src.scale}\t{self.src.emitCode()}, {self.dst.emitCode()}\n"

        return f"\tofs\t{self.src.emitCode()}, {self.dst.emitCode()}\n"

    def print(self) -> str:
        return f"Offset({self.src}, {self.dst})\n"

class LABEL(AssemblerInstruction):
    GENERAL_LABEL_COUNT: int = 0

    @staticmethod
    def generateLabel(posfix: str) -> str:
        label = f"{LABEL.GENERAL_LABEL_COUNT}_{posfix}"
        LABEL.GENERAL_LABEL_COUNT += 1
        return label

    def __init__(self, identifier: str, parentAST: AssemblyAST | None = None) -> None:
        self.identifier = identifier
        super().__init__(parentAST)

    def emitCode(self) -> str:
        return f"\nL{self.identifier}:\n"

    def print(self) -> str:
        return f"Label({self.identifier})\n"

class PSH(AssemblerInstruction):
    def __init__(self, operand: AssemblerOperand, parentAST: AssemblyAST | None = None) -> None:
        self.operand = operand
        super().__init__(parentAST)

    def secondPass(self):
        self.operand = self.convertFromPseudo(self.operand)

    def thirdPass(self) -> list[AssemblerInstruction]:
        if isinstance(self.operand, (Memory, Data)):
            # - Stack cannot have a memory address. Save the src into temporary register R0 and then push R0.
            movToReg = self.createChild(MOVE, self.operand.assemblyType, self.operand, Register(self.operand.assemblyType, REG.R0))
            pushFromReg = self.createChild(PSH, Register(AssemblyType.LONGWORD, REG.R0))
            return [movToReg, pushFromReg]
        
        return [self]

    def emitCode(self) -> str:
        return f"\tpsh\t{self.operand.emitCode()}\n"
    
    def print(self) -> str:
        return f"Push({self.operand})\n"

class FUN(AssemblerInstruction):
    def __init__(self, callArgument: str|Register, parentAST: AssemblyAST | None = None) -> None:
        self.callArgument = callArgument
        super().__init__(parentAST)

    def emitCode(self) -> str:
        if isinstance(self.callArgument, str):
            # Offset call.
            if self.callArgument in TACFunction.functions:
                return f"\tfun\t{self.callArgument}\n"
            else:
                # If the function is not defined in the code, maybe it's located somewhere else.
                # Add @PLT to link it externally. 
                return f"\tfun\t{self.callArgument}@PLT\n"
            
        elif isinstance(self.callArgument, Register):
            # Indirect call.
            return f"\tfun\t*{self.callArgument.emitCode()}\n"
        
        else:
            raise ValueError()
    
    def print(self) -> str:
        return f'Call({self.callArgument})\n'

class RET(AssemblerInstruction):
    def __init__(self, parentAST: AssemblyAST | None = None) -> None:
        super().__init__(parentAST)

    def emitCode(self) -> str:
        ret  = f"\tmov\t%rsb, %rsp\n"
        ret += f"\tpop\t%rsb\n"
        ret += f"\tret\n"
        return ret

    def print(self) -> str:
        return "Return\n"

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
    R0      = "r0"
    R1      = "r1"
    R2      = "r2"
    R3      = "r3"
    R4      = "r4"
    R5      = "r5"
    R6      = "r6"
    R7      = "r7"
    FL      = "fl"
    RSP     = "rsp"
    RSB     = "rsb"
    PC      = "pc"
    OP1     = "op1"
    OP2     = "op2"
    RESL    = "resl"
    RESH    = "resh"

class Register(AssemblerOperand):
    def __init__(self, assemblyType: AssemblyType, reg: REG, signExtend: bool = False, parentAST: AssemblyAST | None = None) -> None:
        self.reg = reg
        self.signExtend = signExtend
        super().__init__(assemblyType, parentAST)

    def createCopy(self) -> Register:
        return Register(self.assemblyType, self.reg, self.signExtend, self.parent)

    def emitCode(self) -> str:
        ret = f"%{self.reg.value}"

        match self.assemblyType:
            case AssemblyType.LONGWORD:
                pass

            case AssemblyType.WORD:
                if self.signExtend:
                    ret += "'s16"
                else:
                    ret += "'u16"

            case AssemblyType.BYTE:
                if self.signExtend:
                    ret += "'s16"
                else:
                    ret += "'u16"

            case _:
                raise ValueError(f"Cannot emit code for a register with assembly type {self.assemblyType}")

        return ret

    def print(self) -> str:
        return self.reg.value

class Immediate(AssemblerOperand):
    def __init__(self, value: TACValue, parentAST: AssemblyAST | None = None) -> None:
        self.value = value
        self.valueStr = self.value.print()

        if not self.value.isConstant:
            raise ValueError("Cannot create an Immediate operand from a not constant value")

        self.intVal = int(self.valueStr)

        super().__init__(AssemblyType.fromTAC(self.value.valueType), parentAST)

    def createCopy(self) -> Immediate:
        return Immediate(self.value, self.parent)

    def emitCode(self) -> str:
        return f"${self.valueStr}"

    def print(self) -> str:
        return f"Imm({self.value})"
    
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
        self.register = Register(AssemblyType.LONGWORD, register)
        self.offset = offset
        super().__init__(assemblyType, parentAST)

    def createCopy(self) -> Memory:
        return Memory(self.assemblyType, self.register.reg, self.offset, self.parent)

    def emitCode(self) -> str:
        if self.offset == 0:
            return f"({self.register.emitCode()})"
        else:
            return f"({self.register.emitCode()}){self.offset:+}"

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

        ret = Memory(pseudo.assemblyType, REG.RSB, Memory.STACK_OFFSET, pseudo.parent)
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

            base = Memory(pseudo.assemblyType, REG.RSB, Memory.STACK_OFFSET, pseudo.parent)
            Memory.stackVariables[pseudo.name] = base

        base = Memory.stackVariables[pseudo.name]
        return Memory(pseudo.assemblyType, REG.RSB, base.offset + pseudo.offset, pseudo.parent)

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
            return f"(%rip)+{self.identifier}"
        else:
            return f"(%rip)+{self.identifier}{self.offset:+}"

    def print(self) -> str:
        return f"Data({self.identifier})"

# (regA, regB, scale) -> regA + regB * scale
class Indexed(AssemblerOperand):
    def __init__(self, assemblyType: AssemblyType, base: Register, index: Register, scale: int,
                 parentAST: AssemblyAST | None = None) -> None:
        self.base = base
        self.index = index
        self.scale = scale

        if scale < -2147483648 or scale > 2147483647:
             raise ValueError("Scale is out of limits")

        super().__init__(assemblyType, parentAST)

    def createCopy(self) -> Indexed:
        return Indexed(self.assemblyType, self.base, self.index, self.scale, self.parent)

    def emitCode(self) -> str:
        return f"({self.base.emitCode()}, {self.index.emitCode()}, {self.scale})"

    def print(self) -> str:
        return f"Indexed({self.base}, {self.index}, {self.scale})"
