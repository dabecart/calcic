"""
TAC.py

Intermediary representation of the code on an agnostic language called Three Address Code (TAC).
Receives the objects of the parsing stage and interprets them into TAC.

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from .parser import *
from .types import *

from typing import Type, TypeVar
T = TypeVar("T", bound="TAC")

class TAC(ABC):
    def __init__(self, parentTAC: TAC|None = None) -> None:
        super().__init__()
        # AST containing this Assembly AST node.
        self.parent = parentTAC
        self.parse()

    @abstractmethod
    def parse(self):
        pass

    @abstractmethod
    def print(self) -> str:
        pass

    def __str__(self) -> str:
        return self.print()

    def createChild(self, assemblerType: Type[T], *args) -> T:
        return assemblerType(*args, parentTAC=self)
    
    def makeCast(self, innerDeclaratorType: DeclaratorType, innerValue: TACValue, 
                 castDeclaratorType: DeclaratorType, insts: list[TACInstruction]):
        if isinstance(castDeclaratorType, BaseDeclaratorType):
            castType = castDeclaratorType.baseType
        else:
            # Pointers and functions should be taken as ulong.
            castType = TypeSpecifier.ULONG

        if isinstance(innerDeclaratorType, BaseDeclaratorType):
            innerType = innerDeclaratorType.baseType
        else:
            # Pointers and functions should be taken as ulong.
            innerType = TypeSpecifier.ULONG

        # Double casts.
        if innerType.isDecimal() and castType.isDecimal():
            result = self.createChild(TACDecimalToDecimal, innerValue, castDeclaratorType, insts).result
        elif innerType.isDecimal():
            if castType.isSignedInt():
                result = self.createChild(TACDecimalToInt, innerValue, castDeclaratorType, insts).result
            else:
                result = self.createChild(TACDecimalToUInt, innerValue, castDeclaratorType, insts).result
        elif castType.isDecimal():
            if innerType.isSignedInt():
                result = self.createChild(TACIntToDecimal, innerValue, castDeclaratorType, insts).result
            else:
                result = self.createChild(TACUIntToDecimal, innerValue, castDeclaratorType, insts).result
        
        # Integer casts.
        elif castType.getByteSize() == innerType.getByteSize():
            result = self.createChild(TACValue, False, castDeclaratorType)
            self.createChild(TACCopy, innerValue, result, insts)
        elif castType.getByteSize() < innerType.getByteSize():
            result = self.createChild(TACTruncate, innerValue, castDeclaratorType, insts).result
        elif innerType.isSignedInt():
            result = self.createChild(TACSignExtend, innerValue, castDeclaratorType, insts).result
        else:
            result = self.createChild(TACZeroExtend, innerValue, castDeclaratorType, insts).result
        
        return TACBaseOperand(result, castDeclaratorType, insts)
    
    def makeAssignment(self, resultType: DeclaratorType, lValue: TACExpressionResult, 
                       rValue: TACValue, insts: list[TACInstruction]) -> TACBaseOperand:
        match lValue:
            case TACBaseOperand():
                self.createChild(TACCopy, rValue, lValue.value, insts)
                return lValue
            case TACDereferencedPointer():
                self.createChild(TACStore, rValue, lValue.value, insts)
                return TACBaseOperand(rValue, resultType, insts)
            case TACSubObject():
                self.createChild(TACCopyToOffset, rValue, lValue.value, lValue.offset, insts)
                return TACBaseOperand(rValue, resultType, insts)
            case _:
                raise ValueError("Invalid lvalue in the assignment")

    def parseTACExpression(self, exp: Exp, insts: list[TACInstruction]) -> TACExpressionResult:
        match exp:
            case LongConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.LONG.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case ULongConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.ULONG.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case IntConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.INT.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case UIntConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.UINT.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)
            
            case ShortConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.SHORT.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case UShortConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.USHORT.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case CharConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.CHAR.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case UCharConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.UCHAR.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case DoubleConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.DOUBLE.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case FloatConstant():
                val = self.createChild(TACValue, True, TypeSpecifier.FLOAT.toBaseType(), exp)
                return TACBaseOperand(val, exp.typeId, insts)
            
            case String():
                # This string is being used inside an expression. Therefore, the content of the string
                # should be stored in the .rodata section (constants section) first, then use its value
                # as a reference. Start with .L so its hidden.
                constantName: str = exp.context.mangleIdentifier(".Lstr")

                exp.context.constantVariablesMap[constantName] = ConstantVariableContext(
                    name=constantName,
                    idType=exp.typeId,
                    initialization=exp.toConstantsList()
                )

                stringConstant = self.createChild(TACValue, True, exp.typeId, constantName)
                return TACBaseOperand(stringConstant, exp.typeId, insts)
            
            case Variable():
                val = self.createChild(TACValue, False, exp.typeId, exp)
                return TACBaseOperand(val, exp.typeId, insts)

            case Unary():
                if exp.unaryOperator in (UnaryOperator.PRE_INCREMENT, UnaryOperator.PRE_DECREMENT):
                    if isinstance(exp.typeId, BaseDeclaratorType):
                        # If the operation needs an integer promotion, do the cast first, whilst 
                        # keeping track of the preCast result.
                        if exp.needsIntegerPromotion:
                            preCast = self.parseTACExpression(exp.inner, insts)
                            inner = self.makeCast(exp.originalType, preCast.convert(), TypeSpecifier.INT.toBaseType(), insts)
                        else:
                            inner = self.parseTACExpression(exp.inner, insts)
    
                        # Run the Unary operation.
                        unary = self.createChild(TACUnary, exp.unaryOperator, inner.convert(), insts)
                        # Set the value of the original variable to the unary result.
                        if exp.needsIntegerPromotion:
                            castToBeforePromotion = self.makeCast(unary.result.valueType, unary.result, exp.originalType, insts)
                            self.makeAssignment(exp.originalType, preCast, castToBeforePromotion.convert(), insts)
                        else:
                            self.makeAssignment(unary.result.valueType, inner, unary.result, insts)
                        # Return the recently operated expression.
                        return TACBaseOperand(unary.result, exp.typeId, insts)
                    elif isinstance(exp.typeId, (PointerDeclaratorType, ArrayDeclaratorType)):
                        inner = self.parseTACExpression(exp.inner, insts)
                        # Increment/decrement the pointer.
                        preDereference = TACValue(False, exp.typeId)
                        if exp.unaryOperator == UnaryOperator.PRE_INCREMENT:
                            delta = TACValue(True, TypeSpecifier.LONG.toBaseType(), "1", self)
                        else:
                            delta = TACValue(True, TypeSpecifier.LONG.toBaseType(), "-1", self)

                        pointerOp = self.createChild(TACAddToPointer, inner.convert(), 
                            delta, exp.typeId.declarator.getByteSize(), preDereference, insts)
                        # Set the value of the original variable to the unary result.
                        self.makeAssignment(pointerOp.result.valueType, inner, pointerOp.result, insts)
                        # Return the recently operated expression.
                        return TACBaseOperand(pointerOp.result, exp.typeId, insts)
                    else:
                        raise ValueError("not implemented")

                elif exp.unaryOperator in (UnaryOperator.POST_INCREMENT, UnaryOperator.POST_DECREMENT):
                    if isinstance(exp.typeId, BaseDeclaratorType):
                        inner = self.parseTACExpression(exp.inner, insts)
                        previousValue: TACValue = inner.convert()
                        # Save the old value. No need to check integer promotion now as the return 
                        # type after integer promotion is the original type.
                        old = self.createChild(TACValue, False, exp.inner.typeId)
                        self.createChild(TACCopy, previousValue, old, insts)
                        # If the operation needs an integer promotion, do the cast first, whilst 
                        # keeping track of the preCast result.
                        if exp.needsIntegerPromotion:
                            preCast = inner
                            inner = self.makeCast(exp.originalType, preCast.convert(), TypeSpecifier.INT.toBaseType(), insts)
                            previousValue = inner.convert()
                        # Run the Unary operation.
                        unary = self.createChild(TACUnary, exp.unaryOperator, previousValue, insts)
                        # Set the value of the original variable to the unary result.
                        if exp.needsIntegerPromotion:
                            castToBeforePromotion = self.makeCast(unary.result.valueType, unary.result, exp.originalType, insts)
                            self.makeAssignment(exp.originalType, preCast, castToBeforePromotion.convert(), insts)
                        else:
                            self.makeAssignment(unary.result.valueType, inner, unary.result, insts)
                        # Return old.
                        return TACBaseOperand(old, exp.typeId, insts)
                    elif isinstance(exp.typeId, (PointerDeclaratorType, ArrayDeclaratorType)):
                        inner = self.parseTACExpression(exp.inner, insts)
                        # Save the old value.
                        previousValue: TACValue = inner.convert()
                        old = self.createChild(TACValue, False, exp.inner.typeId)
                        self.createChild(TACCopy, previousValue, old, insts)
                        # Increment/decrement the pointer.
                        preDereference = TACValue(False, exp.typeId)
                        if exp.unaryOperator == UnaryOperator.POST_INCREMENT:
                            delta = TACValue(True, TypeSpecifier.LONG.toBaseType(), "1", self)
                        else:
                            delta = TACValue(True, TypeSpecifier.LONG.toBaseType(), "-1", self)
                        pointerOp = self.createChild(TACAddToPointer, inner.convert(), 
                            delta, exp.typeId.declarator.getByteSize(), preDereference, insts)
                        # Set the value of the original variable to the unary result.
                        self.makeAssignment(preDereference.valueType, inner, preDereference, insts)
                        # Return old.
                        return TACBaseOperand(old, exp.typeId, insts)
                    else:
                        raise ValueError("not implemented")
                else:
                    parsedExp = self.parseTACExpression(exp.inner, insts).convert()
                    unary = self.createChild(TACUnary, exp.unaryOperator, parsedExp, insts)
                    return TACBaseOperand(unary.result, exp.typeId, insts)
            
            case Binary():
                binary = TACBinary(exp, instructionsList=insts, parentTAC=self)
                # binary.result contains the value of the binary operation.
                toStoreValue: TACValue = binary.result
                declaratorResult = exp.typeId

                if exp.compoundBinary:
                    # Exp1 may be casted to the common type of exp1 and exp2.
                    if exp.exp1OriginalType != binary.result.valueType:
                        # Do an implicit casting.
                        if not Cast.checkImplicitCast(binary.result.valueType, False, exp.exp1OriginalType):
                            exp.raiseError(f"Cannot implicitly cast from {binary.result.valueType} to {exp.exp1OriginalType}")
                        
                        toStoreValue = self.makeCast(binary.result.valueType, binary.result, exp.exp1OriginalType, insts).convert()

                    # The result of the expression is not the common type of exp1 and exp2 but the 
                    # type of exp1.
                    declaratorResult = exp.exp1OriginalType

                    # Now, assign it to the lvalue.
                    return self.makeAssignment(declaratorResult, binary.exp1Lvalue, toStoreValue, insts)
                
                return TACBaseOperand(toStoreValue, declaratorResult, insts)
            
            case Assignment():
                # First solve the right side of the assignment.
                rValue = self.parseTACExpression(exp.exp2, insts).convert()
                # Then assign to the left side of the assignment (which should be a variable) the 
                # right's result.
                lValue = self.parseTACExpression(exp.exp1, insts)
                return self.makeAssignment(exp.typeId, lValue, rValue, insts)
            
            case TernaryConditional():
                endIfLabel: str = TACLabel.getNewLabelName()
                elseLabel: str = TACLabel.getNewLabelName()
                result = self.createChild(TACValue, False, exp.typeId)
                
                conditionExp: TACValue = self.parseTACExpression(exp.condition, insts).convert()

                self.createChild(TACJumpIfZero, conditionExp, elseLabel, insts)
                thenResult = self.parseTACExpression(exp.thenExp, insts).convert()
                if exp.thenExp.typeId != TypeSpecifier.VOID.toBaseType():
                    # Do not save the result of a void expression.
                    self.createChild(TACCopy, thenResult, result, insts)
                self.createChild(TACJump, endIfLabel, insts)
                
                self.createChild(TACLabel, elseLabel, insts)
                elseResult = self.parseTACExpression(exp.elseExp, insts).convert()
                if exp.elseExp.typeId != TypeSpecifier.VOID.toBaseType():
                    # Do not save the result of a void expression.
                    self.createChild(TACCopy, elseResult, result, insts)
                self.createChild(TACLabel, endIfLabel, insts)

                return TACBaseOperand(result, exp.typeId, insts)
            
            case FunctionCall():
                argValues: list[TACValue] = []
                # Store all the arguments expressions values.
                for argExp in exp.argumentList:
                    argVal = self.parseTACExpression(argExp, insts).convert()
                    argValues.append(argVal)
                # Call the function.
                funcCall = self.createChild(TACFunctionCall, exp.funcIdentifier, exp.typeId, argValues, insts)
                return TACBaseOperand(funcCall.result, exp.typeId, insts)
            
            case Cast():
                if exp.inner.typeId == exp.typeId or exp.typeId == TypeSpecifier.VOID.toBaseType():
                    # No need to do any casting.
                    return self.parseTACExpression(exp.inner, insts)
                
                return self.makeCast(exp.inner.typeId, 
                                     self.parseTACExpression(exp.inner, insts).convert(), 
                                     exp.typeId, insts)
            
            case Dereference():
                preDereference = self.parseTACExpression(exp.inner, insts).convert()
                return TACDereferencedPointer(preDereference, exp.typeId, insts)

            case AddressOf():
                preDereference = self.parseTACExpression(exp.inner, insts)
                match preDereference:
                    case TACBaseOperand():
                        dst = self.createChild(TACValue, False, exp.typeId)
                        self.createChild(TACGetAddress, preDereference.value, dst, insts)
                        return TACBaseOperand(dst, exp.typeId, insts)
                    case TACDereferencedPointer():
                        return TACBaseOperand(preDereference.value, exp.typeId, insts)
                    case TACSubObject():
                        dst = self.createChild(TACValue, False, exp.typeId)
                        # Get the base address of the leftmost struct/union.
                        self.createChild(TACGetAddress, preDereference.value, dst, insts)
                        # Add to this address the offset bytes of the field.
                        delta = TACValue(True, TypeSpecifier.LONG.toBaseType(), str(preDereference.offset), self)
                        self.createChild(TACAddToPointer, dst, delta, 1, dst, insts)
                        return TACBaseOperand(dst, exp.typeId, insts)
                    case _:
                        raise ValueError("Invalid address of TAC conversion")
                    
            case Subscript():
                pointer = self.parseTACExpression(exp.pointer, insts).convert()
                index = self.parseTACExpression(exp.index, insts).convert()

                if not isinstance(pointer.valueType, (PointerDeclaratorType, ArrayDeclaratorType)):
                    raise ValueError("Expected a pointer in a subscript")
                
                # This is saved in a pointer type.
                preDereference = self.createChild(TACValue, False, pointer.valueType)
                self.createChild(TACAddToPointer, pointer, index, exp.typeId.getByteSize(), preDereference, insts)
                # The return dereferenced pointer has the inner type of the pointer.
                return TACDereferencedPointer(preDereference, exp.typeId, insts)
            
            case SizeOf():
                val = TACValue(True, exp.typeId, str(exp.inner.typeId.getByteSize()))
                return TACBaseOperand(val, val.valueType, insts)

            case SizeOfType():
                val = TACValue(True, exp.typeId, str(exp.inner.getByteSize()))
                return TACBaseOperand(val, val.valueType, insts)

            case Dot():
                inner = self.parseTACExpression(exp.leftExp, insts)
                if not isinstance(inner.processedType, BaseDeclaratorType) or \
                   inner.processedType.baseType.name not in ("STRUCT", "UNION"):
                    raise ValueError(f"Invalid type {inner.processedType}")
                
                if inner.processedType.baseType.name == "STRUCT":
                    # Get the byte offset of the member.
                    member = inner.processedType.baseType.getMember(exp.member)
                    if member is None:
                        raise ValueError("Couldn't find the member")

                    match inner:
                        case TACBaseOperand():
                            return TACSubObject(inner.value, member.offset, exp.typeId, insts)
                        case TACSubObject():
                            return TACSubObject(inner.value, inner.offset + member.offset, exp.typeId, insts)
                        case TACDereferencedPointer():
                            # If the offset is 0, no need to add it to the pointer.
                            if member.offset != 0:
                                dstPointer = self.createChild(TACValue, False, PointerDeclaratorType(exp.typeId))
                                delta = TACValue(True, TypeSpecifier.LONG.toBaseType(), str(member.offset), self)
                                self.createChild(TACAddToPointer, inner.value, delta, 1, dstPointer, insts)
                                return TACDereferencedPointer(dstPointer, exp.typeId, insts)
                            else:
                                return TACDereferencedPointer(inner.value, exp.typeId, insts)
                else:
                    # This is an union. Members have offset 0. A 
                    match inner:
                        case TACBaseOperand():
                            return TACSubObject(inner.value, 0, exp.typeId, insts)
                        case TACSubObject():
                            return TACSubObject(inner.value, inner.offset, exp.typeId, insts)
                        case TACDereferencedPointer():
                            return TACDereferencedPointer(inner.value, exp.typeId, insts)
                            
            case Arrow():
                inner = self.parseTACExpression(exp.leftExp, insts).convert()
                if not isinstance(inner.valueType, PointerDeclaratorType) or \
                   not isinstance(inner.valueType.declarator, BaseDeclaratorType) or \
                   inner.valueType.declarator.baseType.name not in ("STRUCT", "UNION"):
                    raise ValueError(f"Invalid type {inner.valueType}")

                if inner.valueType.declarator.baseType.name == "STRUCT":
                    # Get the byte offset of the member.
                    member = inner.valueType.declarator.baseType.getMember(exp.member)
                    if member is None:
                        raise ValueError("Couldn't find the member")

                    # If the offset is 0, no need to add it to the pointer.
                    if member.offset != 0:
                        delta = TACValue(True, TypeSpecifier.LONG.toBaseType(), str(member.offset), self)
                        dstPointer = self.createChild(TACValue, False, PointerDeclaratorType(exp.typeId))
                        self.createChild(TACAddToPointer, inner, delta, 1, dstPointer, insts)
                        return TACDereferencedPointer(dstPointer, exp.typeId, insts)
                    else:
                        return TACDereferencedPointer(inner, exp.typeId, insts)
                else:
                    # This is an union. Offset is zero for all members.
                    return TACDereferencedPointer(inner, exp.typeId, insts)

        raise ValueError(f"Cannot convert from AST {type(exp)} to TAC")
            
    # Parses Statements and Declarations (AST Blocks).
    def parseTACBlockItem(self, blockItem: AST, insts: list[TACInstruction]):
        match blockItem:
            case ReturnStatement():
                self.createChild(TACReturn, blockItem, insts)
            
            case IfStatement():
                endIfLabel: str = TACLabel.getNewLabelName()
                conditionExp: TACValue = self.parseTACExpression(blockItem.condition, insts).convert()

                if blockItem.elseStatement is None:
                    self.createChild(TACJumpIfZero, conditionExp, endIfLabel, insts)
                    self.parseTACBlockItem(blockItem.thenStatement, insts)
                else:
                    elseLabel: str = TACLabel.getNewLabelName()
                    self.createChild(TACJumpIfZero, conditionExp, elseLabel, insts)
                    self.parseTACBlockItem(blockItem.thenStatement, insts)
                    self.createChild(TACJump, endIfLabel, insts)
                    
                    self.createChild(TACLabel, elseLabel, insts)
                    self.parseTACBlockItem(blockItem.elseStatement, insts)

                self.createChild(TACLabel, endIfLabel, insts)

            case ExpressionStatement():
                self.parseTACExpression(blockItem.exp, insts).convert()
            
            case NullStatement():
                # Nothing to generate.
                return

            case CompoundStatement():
                for innerBlock in blockItem.block.blockItems: # type: ignore
                    self.parseTACBlockItem(innerBlock, insts)

            case BreakStatement():
                self.createChild(TACJump, f"break_{blockItem.jumpLabel}", insts)

            case ContinueStatement():
                self.createChild(TACJump, f"continue_{blockItem.jumpLabel}", insts)

            case WhileStatement():
                continueLabel = f"continue_{blockItem.loopTag}"
                breakLabel = f"break_{blockItem.loopTag}"

                self.createChild(TACLabel, continueLabel, insts)
                conditionResult = self.parseTACExpression(blockItem.condition, insts).convert()
                self.createChild(TACJumpIfZero, conditionResult, breakLabel, insts)
                self.parseTACBlockItem(blockItem.body, insts)
                self.createChild(TACJump, continueLabel, insts)
                self.createChild(TACLabel, breakLabel, insts)

            case DoWhileStatement():
                startLabel = f"start_{blockItem.loopTag}"
                continueLabel = f"continue_{blockItem.loopTag}"
                breakLabel = f"break_{blockItem.loopTag}"

                self.createChild(TACLabel, startLabel, insts)
                self.parseTACBlockItem(blockItem.body, insts)
                self.createChild(TACLabel, continueLabel, insts)
                conditionResult = self.parseTACExpression(blockItem.condition, insts).convert()
                self.createChild(TACJumpIfNotZero, conditionResult, startLabel, insts)
                self.createChild(TACLabel, breakLabel, insts)

            case ForStatement():
                startLabel = f"start_{blockItem.loopTag}"
                continueLabel = f"continue_{blockItem.loopTag}"
                breakLabel = f"break_{blockItem.loopTag}"

                if isinstance(blockItem.init, Declaration):
                    self.parseTACBlockItem(blockItem.init, insts)
                elif isinstance(blockItem.init, Exp):
                    self.parseTACExpression(blockItem.init, insts).convert()

                self.createChild(TACLabel, startLabel, insts)
                if blockItem.condition is not None:
                    conditionResult = self.parseTACExpression(blockItem.condition, insts).convert()
                    self.createChild(TACJumpIfZero, conditionResult, breakLabel, insts)
                
                self.parseTACBlockItem(blockItem.body, insts)
                
                self.createChild(TACLabel, continueLabel, insts)
                if blockItem.post is not None:
                    self.parseTACExpression(blockItem.post, insts).convert()
                
                self.createChild(TACJump, startLabel, insts)
                self.createChild(TACLabel, breakLabel, insts)

            case CaseStatement():
                # Convert negative numbers so that the "-" doesn't affect the linker.
                numberLabel = blockItem.value.constValue.replace("-", "_neg")
                labelName = f"case_{numberLabel}_{blockItem.switchLabel}"
                self.createChild(TACLabel, labelName, insts)
                self.parseTACBlockItem(blockItem.statement, insts)

            case DefaultStatement():
                self.createChild(TACLabel, f"default_{blockItem.switchLabel}", insts)
                self.parseTACBlockItem(blockItem.statement, insts)

            case SwitchStatement():
                controlVar = self.parseTACExpression(blockItem.condition, insts).convert()
                # Generate the JumpIfValue instructions.
                for caseSt in blockItem.caseList:
                    # Convert negative numbers so that the "-" doesn't affect the linker.
                    numberLabel = caseSt.value.constValue.replace("-", "_neg")
                    labelName = f"case_{numberLabel}_{blockItem.switchLabel}"
                    self.createChild(
                        TACJumpIfValue, controlVar, TACValue(True, controlVar.valueType, caseSt.value),
                        labelName, insts)

                # If none of the JumpIfValue is taken, jump to the default. If there's no default, 
                # exit the switch by jumping to the break label.
                if blockItem.defaultCase is None:
                    self.createChild(TACJump, f"break_{blockItem.switchLabel}", insts)
                else:
                    self.createChild(TACJump, f"default_{blockItem.switchLabel}", insts)

                self.parseTACBlockItem(blockItem.body, insts)

                self.createChild(TACLabel, f"break_{blockItem.switchLabel}", insts)

            case LabelStatement():
                self.createChild(TACLabel, blockItem.labelName, insts)
                self.parseTACBlockItem(blockItem.statement, insts)

            case GotoStatement():
                self.createChild(TACJump, blockItem.labelName, insts)

            case VariableDeclaration():
                # Discard declarations of static variables (this was already taken care of in the 
                # top level of the program).
                if blockItem.identifier in blockItem.context.staticVariablesMap:
                    return

                if isinstance(blockItem.initialization, SingleInitializer):
                    rightResult = self.parseTACExpression(blockItem.initialization.init, insts).convert()
                    leftVariable = self.createChild(TACValue, False, blockItem.typeId, blockItem.identifier)
                    self.createChild(TACCopy, rightResult, leftVariable, insts)
                elif isinstance(blockItem.initialization, CompoundInitializer):
                    leftVariable = self.createChild(TACValue, False, blockItem.typeId, blockItem.identifier)
                    
                    def declareArray(initialOffset: int, arrayType: DeclaratorType, input: Initializer):
                        if not isinstance(arrayType, ArrayDeclaratorType):
                            raise ValueError(f"Cannot declare array with input type {arrayType}")
                        
                        offset: int = initialOffset
                        offsetStep: int = arrayType.getArrayBaseType().getByteSize()
                    
                        def declareArrayInitializer(subArrayType: DeclaratorType, input: Initializer):
                            nonlocal offset

                            # TODO: this could be optimized by grouping the array's items inside ints or longs.
                            if isinstance(input, CompoundInitializer):
                                for inner in input.init:
                                    if isinstance(subArrayType, BaseDeclaratorType) and subArrayType.baseType.name in "STRUCT":
                                        # Array of structs.
                                        declareStruct(offset, subArrayType, inner)
                                        offset += offsetStep
                                    elif isinstance(subArrayType, BaseDeclaratorType) and subArrayType.baseType.name in "UNION":
                                        # Array of unions.
                                        declareUnion(offset, subArrayType, inner)
                                        offset += offsetStep
                                    elif isinstance(subArrayType, ArrayDeclaratorType):
                                        # Multidimensional array.
                                        declareArrayInitializer(subArrayType.declarator, inner)
                                    else:
                                        # Array of single initializers.
                                        declareArrayInitializer(subArrayType, inner)
                                    
                            elif isinstance(input, SingleInitializer):
                                if isinstance(subArrayType, BaseDeclaratorType) and subArrayType.baseType.name == "STRUCT":
                                    # Array of structs, initialized from another struct. Ex:
                                    # struct example a;
                                    # struct example array[5] = {a, ...}
                                    declareStruct(offset, subArrayType, input)
                                elif isinstance(subArrayType, BaseDeclaratorType) and subArrayType.baseType.name == "UNION":
                                    # Array of unions, initialized from another union.
                                    declareUnion(offset, subArrayType, input)
                                else:
                                    expr = self.parseTACExpression(input.init, insts).convert()
                                    self.createChild(TACCopyToOffset, expr, leftVariable, offset, insts)
                                
                                offset += offsetStep
                            else:
                                raise ValueError()
                            
                        declareArrayInitializer(arrayType.declarator, input)
                    
                    def declareStruct(baseOffset: int, structType: DeclaratorType, input: Initializer):
                        if not isinstance(structType, BaseDeclaratorType) or structType.baseType.name != "STRUCT":
                            raise ValueError(f"Cannot declare struct with input type {structType}")
                        
                        if isinstance(input, SingleInitializer):
                            # Initialize struct from another struct.
                            expr = self.parseTACExpression(input.init, insts).convert()
                            self.createChild(TACCopyToOffset, expr, leftVariable, baseOffset, insts)
                        elif isinstance(input, CompoundInitializer):
                            for memberInfo, memberInit in zip(structType.baseType.getMembers(), input.init):
                                memberOffset = baseOffset + memberInfo.offset
                                
                                if isinstance(memberInfo.type, ArrayDeclaratorType):
                                    # Array inside the struct.
                                    declareArray(memberOffset, memberInfo.type, memberInit)
                                elif isinstance(memberInfo.type, BaseDeclaratorType) and memberInfo.type.baseType.name == "STRUCT":
                                    # Struct inside a struct.
                                    declareStruct(memberOffset, memberInfo.type, memberInit)
                                elif isinstance(memberInfo.type, BaseDeclaratorType) and memberInfo.type.baseType.name == "UNION":
                                    # Union inside a struct.
                                    declareUnion(memberOffset, memberInfo.type, memberInit)
                                elif isinstance(memberInit, SingleInitializer):
                                    # Normal value inside a struct.
                                    expr = self.parseTACExpression(memberInit.init, insts).convert()
                                    self.createChild(TACCopyToOffset, expr, leftVariable, memberOffset, insts)
                                else:
                                    raise ValueError()
                        else:
                            raise ValueError()

                    def declareUnion(baseOffset: int, unionType: DeclaratorType, input: Initializer):
                        if not isinstance(unionType, BaseDeclaratorType) or unionType.baseType.name != "UNION":
                            raise ValueError(f"Cannot declare union with input type {unionType}")
                        
                        if isinstance(input, SingleInitializer):
                            # Initialize union from another union.
                            expr = self.parseTACExpression(input.init, insts).convert()
                            self.createChild(TACCopyToOffset, expr, leftVariable, baseOffset, insts)
                        elif isinstance(input, CompoundInitializer):
                            # Should only be one memberInit.
                            for memberInfo, memberInit in zip(unionType.baseType.getMembers(), input.init):
                                if isinstance(memberInfo.type, ArrayDeclaratorType):
                                    # Array inside the union.
                                    declareArray(baseOffset, memberInfo.type, memberInit)
                                elif isinstance(memberInfo.type, BaseDeclaratorType) and memberInfo.type.baseType.name == "STRUCT":
                                    # Struct inside an union.
                                    declareStruct(baseOffset, memberInfo.type, memberInit)
                                elif isinstance(memberInfo.type, BaseDeclaratorType) and memberInfo.type.baseType.name == "UNION":
                                    # Union inside an union.
                                    declareUnion(baseOffset, memberInfo.type, memberInit)
                                elif isinstance(memberInit, SingleInitializer):
                                    # Normal value inside an union.
                                    expr = self.parseTACExpression(memberInit.init, insts).convert()
                                    self.createChild(TACCopyToOffset, expr, leftVariable, baseOffset, insts)
                                else:
                                    raise ValueError()
                        else:
                            raise ValueError()

                    if isinstance(blockItem.typeId, ArrayDeclaratorType):
                        declareArray(0, blockItem.typeId, blockItem.initialization)
                    elif isinstance(blockItem.typeId, BaseDeclaratorType) and blockItem.typeId.baseType.name == "STRUCT":
                        declareStruct(0, blockItem.typeId, blockItem.initialization)
                    elif isinstance(blockItem.typeId, BaseDeclaratorType) and blockItem.typeId.baseType.name == "UNION":
                        declareUnion(0, blockItem.typeId, blockItem.initialization)
                    else:
                        raise ValueError(f"Cannot initialize {blockItem.typeId} with the current initialization")

            case StructDeclaration():
                # Nothing to generate with struct declarations.
                return

            case UnionDeclaration():
                # Nothing to generate with union declarations.
                return

            case EnumDeclaration():
                # Nothing to generate with enum declarations.
                return

            case FunctionDeclaration():
                # Function declarations can only be defined from an ASTProgram. These declarations 
                # are just function declarations, without their definitions.
                return
            
            case _:
                raise ValueError(f"Cannot convert from AST ({type(blockItem)}) statement to TAC")

class TACValue(TAC):
    _VARIABLE_COUNT = 0

    def __init__(self, isConstant: bool, valueType: DeclaratorType, value: str|Exp|None = None, \
                 parentTAC: TAC|None = None) -> None:
        self.parent = parentTAC
        
        self.isConstant = isConstant
        self.valueType = valueType
        self.inputValue = value

        if self.isConstant:
            if type(value) is str:
                self.constantValue: str = value
            elif isinstance(value, (CharConstant, UCharConstant, 
                                    ShortConstant, UShortConstant, 
                                    IntConstant, UIntConstant, 
                                    LongConstant, ULongConstant, 
                                    DoubleConstant, FloatConstant)):
                self.constantValue: str = value.constValue
            else:
                raise ValueError(f"Invalid value in TACValue: {value}")
        else:
            if isinstance(value, Variable):
                self.vbeName: str = value.identifier
            elif type(value) is str:
                self.vbeName: str = value
            else:
                # Using : to differentiate between parser Variables (which use .) and TAC temporary variables.
                self.vbeName = f"tmp:{TACValue._VARIABLE_COUNT}"
                TACValue._VARIABLE_COUNT += 1

    def parse(self):
        pass

    def print(self) -> str:
        if self.isConstant:
            return self.constantValue
        else:
            return self.vbeName
        
class TACExpressionResult(TAC):
    def __init__(self, value: TACValue, processedType: DeclaratorType, insts: list[TACInstruction], 
                 parentTAC: TAC | None = None) -> None:
        self.value = value
        self.processedType = processedType
        self.insts = insts
        super().__init__(parentTAC)

    def parse(self):
        pass

    @abstractmethod
    def print(self) -> str:
        pass

    # Used to convert rvalues into lvalues.
    def convert(self) -> TACValue:
        match self:
            case TACBaseOperand():
                # Modify the type of the passed value to the processed type.
                self.value.valueType = self.processedType
                return self.value
            case TACDereferencedPointer():
                dst = self.createChild(TACValue, False, self.processedType)
                self.createChild(TACLoad, self.value, dst, self.insts)
                return dst
            case TACSubObject():
                dst = self.createChild(TACValue, False, self.processedType)
                self.createChild(TACCopyFromOffset, self.value, self.offset, dst, self.insts)
                return dst
            case _:
                raise ValueError(f"Cannot convert {self} to lvalue")

class TACBaseOperand(TACExpressionResult):
    def print(self) -> str:
        return f"BaseOperand({self.value})"

class TACDereferencedPointer(TACExpressionResult):
    def print(self) -> str:
        return f"DereferencedPointer({self.value})"
    
class TACSubObject(TACExpressionResult):
    def __init__(self, value: TACValue, offset: int, processedType: DeclaratorType, insts: list[TACInstruction], 
                 parentTAC: TAC | None = None) -> None:
        self.offset = offset
        super().__init__(value, processedType, insts, parentTAC)

    def print(self) -> str:
        return f"SubObject({self.value})"

class TACProgram(TAC):
    def __init__(self, program: Program, parentTAC: TAC | None = None) -> None:
        self.program = program
        super().__init__(parentTAC)

    def parse(self):
        self.topLevel: list[TACTopLevel] = []
        # First initialize all static variables.
        for staticVar in self.program.context.staticVariablesMap.values():
            if staticVar.tentative:
                # Set to zero by default.
                topLevelDecl = self.createChild(
                    TACStaticVariable, staticVar.idType, staticVar.mangledName, staticVar.isGlobal, 
                    [ZeroPaddingInitializer([], None, None, staticVar.idType, staticVar.idType.getByteSize())])
            elif len(staticVar.initialization) == 0:
                # This must be an extern variable initialized somewhere else.
                # Create the instruction so that it is added to the staticVariables dictionary,
                # but do not add it to the topLevel TAC list.
                self.createChild(
                    TACStaticVariable, staticVar.idType, 
                    staticVar.mangledName, staticVar.isGlobal, [])
                continue
            elif staticVar.idType.getTypeQualifiers().const:
                # If the variable is constant, do something similar to above. 
                self.createChild(
                    TACStaticVariable, staticVar.idType, 
                    staticVar.mangledName, staticVar.isGlobal, [])
                # Add its value to the constant variables.
                self.program.context.constantVariablesMap[staticVar.mangledName] = ConstantVariableContext(
                    name=staticVar.mangledName,
                    idType=staticVar.idType,
                    initialization=staticVar.initialization
                )
                continue
            else:
                topLevelDecl = self.createChild(
                    TACStaticVariable, staticVar.idType, staticVar.mangledName, 
                    staticVar.isGlobal, staticVar.initialization)
            
            self.topLevel.append(topLevelDecl)
        
        # Then generate the TAC for the functions.
        for fun in self.program.topLevel: 
            match fun:
                case FunctionDeclaration():
                    # Skip the declarations.
                    if fun.body is None: 
                        continue
                    
                    topLevelDecl = self.createChild(TACFunction, fun)
                case _:
                    continue

            self.topLevel.append(topLevelDecl)

        # Finally, once all instructions have been processed, create the constants section. Push 
        # them on top of everything as they need to be processed first on the assembly stage.
        for const in self.program.context.constantVariablesMap.values():
            topLevelDecl = self.createChild(
                TACConstantVariable, const.idType, const.name, const.initialization)
            self.topLevel.insert(0, topLevelDecl)

    def print(self) -> str:
        ret = ""
        for fun in self.topLevel:
            ret += fun.print()
            ret += "\n"
        return ret

class TACTopLevel(TAC):
    @abstractmethod
    def parse(self):
        pass

    @abstractmethod
    def print(self) -> str:
        pass

class TACStaticVariable(TACTopLevel):
    staticVariables: dict[str, TACStaticVariable] = {}

    def __init__(self, valueType: DeclaratorType, identifier: str, isGlobal: bool,
                 initialization: list[Constant], parentTAC: TAC | None = None) -> None:
        self.valueType = valueType
        self.identifier = identifier
        self.isGlobal = isGlobal
        self.initialization = initialization
        super().__init__(parentTAC)

        # Add the static variable to the list for easy finding on the assembler stage.
        TACStaticVariable.staticVariables[identifier] = self

    def parse(self):
        pass

    def print(self) -> str:
        ret  = f"--- ({self.valueType}) {self.identifier} ---\n"
        toprint = ''.join([str(x) for x in self.initialization])
        ret += f"{toprint}"
        return ret
    
class TACConstantVariable(TACTopLevel):
    constantVariables: dict[str, TACConstantVariable] = {}

    def __init__(self, valueType: DeclaratorType, identifier: str, initialization: list[Constant], 
                 parentTAC: TAC | None = None) -> None:
        self.valueType = valueType
        self.identifier = identifier
        self.initialization = initialization
        super().__init__(parentTAC)

        # Add the static variable to the list for easy finding on the assembler stage.
        TACConstantVariable.constantVariables[identifier] = self

    def parse(self):
        pass

    def print(self) -> str:
        ret  = f"--- ({self.valueType}) {self.identifier} ---\n"
        toprint = ''.join([str(x) for x in self.initialization])
        ret += f"{toprint}"
        return ret

class TACFunction(TACTopLevel):
    functions: dict[str, TACFunction] = {}

    def __init__(self, funDecl: FunctionDeclaration, parentTAC: TAC | None = None) -> None:
        self.funDecl = funDecl
        super().__init__(parentTAC)

        # Add the function to the list for easy finding on the assembler stage.
        TACFunction.functions[funDecl.identifier] = self

    def parse(self):
        self.identifier = self.funDecl.identifier
        self.arguments = self.funDecl.definedArgumentList
        self.instructions: list[TACInstruction] = []
        self.isGlobal = self.funDecl.isGlobal
        
        # Convert the function's body into a list of instructions.
        if self.funDecl.body is None:
            raise ValueError("Unexpected None in a body function")

        for block in self.funDecl.body:
            self.parseTACBlockItem(block, self.instructions)
        
        # Always add a "return 0" at the end of functions. If the function already has a return it
        # will be pruned on the optimization stage.
        ret0 = ReturnStatement([], None, None, 0)
        self.createChild(TACReturn, ret0, self.instructions)

    def print(self) -> str:
        ret = f"--- {self.identifier} ---\n"
        for inst in self.instructions: 
            ret += inst.print()
        return ret

"""
INSTRUCTIONS
"""
class TACInstruction(TAC):
    def __init__(self, instructionsList: list[TACInstruction], 
                 parentTAC: TAC | None = None) -> None:
        self.parent = parentTAC
        self.insts = instructionsList
        # Output or result of the instruction.
        self.result: TACValue = self.parse()
        # Add this instruction to the list once the inner instructions have been parsed.
        self.insts.append(self)

    @abstractmethod
    def parse(self)-> TACValue:
        pass

    @abstractmethod
    def print(self) -> str:
        pass

class TACReturn(TACInstruction):
    def __init__(self, returnAST: ReturnStatement, 
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.returnAST = returnAST
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        if self.returnAST.exp is not None:
            return self.parseTACExpression(self.returnAST.exp, self.insts).convert()
        
        return TACValue(False, TypeSpecifier.VOID.toBaseType())

    def print(self) -> str:
        return f"Return({self.result})\n"

# Converts from int to long.
class TACSignExtend(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        return dest

    def print(self) -> str:
        return f"SignExtend({self.exp}, {self.result})\n"

class TACZeroExtend(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        return dest

    def print(self) -> str:
        return f"ZeroExtend({self.exp}, {self.result})\n"

class TACDecimalToDecimal(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        
        if self.exp.valueType == dest.valueType:
            raise ValueError(f"Converting from the same decimal type: {dest.valueType} in TACDecimalToDecimal")
        
        return dest

    def print(self) -> str:
        return f"DecimalToDecimal({self.exp}, {self.result})\n"

class TACDecimalToInt(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        return dest

    def print(self) -> str:
        return f"DecimalToInt({self.exp}, {self.result})\n"
    
class TACDecimalToUInt(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        return dest

    def print(self) -> str:
        return f"DecimalToUInt({self.exp}, {self.result})\n"
    
class TACIntToDecimal(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        return dest

    def print(self) -> str:
        return f"IntToDecimal({self.exp}, {self.result})\n"
    
class TACUIntToDecimal(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        return dest

    def print(self) -> str:
        return f"UIntToDecimal({self.exp}, {self.result})\n"

# Converts from long to int.
class TACTruncate(TACInstruction):
    def __init__(self, value: TACValue, castType: DeclaratorType,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.exp = value
        self.castType = castType
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the cast result will be stored.
        dest = TACValue(False, self.castType)
        return dest

    def print(self) -> str:
        return f"Truncate({self.exp}, {self.result})\n"

class TACUnary(TACInstruction):
    def __init__(self, operator: UnaryOperator, value: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.operator = operator.convertToSimple()
        self.exp = value
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        # Create a destination temporary value where the result will be stored.
        dest = TACValue(False, self.exp.valueType)
        return dest

    def print(self) -> str:
        return f"Unary({self.operator.name}, {self.exp}, {self.result})\n"
    
class TACBinary(TACInstruction):
    def __init__(self, *args, 
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        if len(args) == 1:
            # ast: Binary.
            self.ASTAsArgument = True
            self.binaryAST: Binary = args[0]
            self.operator = self.binaryAST.binaryOperator
        else:
            # operation, value1, value2
            self.ASTAsArgument = False
            self.operator: BinaryOperator = args[0]
            self.exp1: TACValue = args[1]
            self.exp2: TACValue = args[2]

        self.removeThisFromInsts: bool = False
        super().__init__(instructionsList, parentTAC)

        if self.removeThisFromInsts:
            self.insts.pop()

    def parse(self)-> TACValue:
        if self.ASTAsArgument:
            return self.parseFromASTExpressions()
        else:
            return TACValue(False, self.exp1.valueType)

    def parseFromASTExpressions(self) -> TACValue:
        # Create a temporary destination value where the result of the TACBinary will be stored.
        if self.binaryAST.compoundBinary:
            # Compound binaries end up creating the lvalue's type, but during the evaluation, the 
            # exp1 and exp2 may be casted, its result being the common type.
            # This "dest" value will later be converted to self.binaryAST.typeId on the assignment
            # made when converting AST to TAC. 
            dest = TACValue(False, self.binaryAST.castType)
        else:
            dest = TACValue(False, self.binaryAST.typeId)

        match self.operator:
            case BinaryOperator.AND:
                # To short-circuit the expression, first check exp1 == 0. If true, directly 
                # go to the "falseLabel", where zero is returned.
                # If false, continue to check exp2 == 0. If true, go to "falseLabel" too. 
                # If neither of them fail, then one is returned. After that, it will avoid the 
                # "falseLabel" by jumping to "endLabel". 
                falseLabel: str = TACLabel.getNewLabelName()
                endLabel: str = TACLabel.getNewLabelName()

                self.exp1: TACValue = self.parseTACExpression(self.binaryAST.exp1, self.insts).convert()
                self.createChild(TACJumpIfZero, self.exp1, falseLabel, self.insts)
                
                self.exp2: TACValue = self.parseTACExpression(self.binaryAST.exp2, self.insts).convert()
                self.createChild(TACJumpIfZero, self.exp2, falseLabel, self.insts)

                self.createChild(TACCopy, 
                                 TACValue(True, TypeSpecifier.INT.toBaseType(), "1", self),
                                 dest, self.insts)
                self.createChild(TACJump, endLabel, self.insts)

                self.createChild(TACLabel, falseLabel, self.insts)
                self.createChild(TACCopy, 
                                 TACValue(True, TypeSpecifier.INT.toBaseType(), "0", self), 
                                 dest, self.insts)
                
                self.createChild(TACLabel, endLabel, self.insts)

                # Remove this instruction itself, as there's no single AND or OR instruction: it's 
                # composed of multiple single instructions.
                self.removeThisFromInsts = True

            case BinaryOperator.OR:
                # To short-circuit the expression, first check exp1 != 0. If true, directly 
                # go to the "trueLabel", where one is returned.
                # If false, continue to check exp2 != 0. If true, go to "trueLabel" too. 
                # If neither of them fail, then zero is returned. After that, it will avoid the 
                # "trueLabel" by jumping to "endLabel". 
                trueLabel: str = TACLabel.getNewLabelName()
                endLabel: str = TACLabel.getNewLabelName()

                self.exp1: TACValue = self.parseTACExpression(self.binaryAST.exp1, self.insts).convert()
                self.createChild(TACJumpIfNotZero, self.exp1, trueLabel, self.insts)
                
                self.exp2: TACValue = self.parseTACExpression(self.binaryAST.exp2, self.insts).convert()
                self.createChild(TACJumpIfNotZero, self.exp2, trueLabel, self.insts)

                self.createChild(TACCopy, 
                                 TACValue(True, TypeSpecifier.INT.toBaseType(), "0", self), 
                                 dest, self.insts)
                self.createChild(TACJump, endLabel, self.insts)

                self.createChild(TACLabel, trueLabel, self.insts)
                self.createChild(TACCopy, 
                                 TACValue(True, TypeSpecifier.INT.toBaseType(), "1", self), 
                                 dest, self.insts)
                
                self.createChild(TACLabel, endLabel, self.insts)

                # Remove this instruction itself, as there's no single AND or OR instruction: it's 
                # composed of multiple single instructions.
                self.removeThisFromInsts = True

            case _:
                if self.binaryAST.exp1IsCasted:
                    if not isinstance(self.binaryAST.exp1, Cast):
                        raise ValueError()
                    
                    # Do this cast manually to get the inner expression. This is used in composed
                    # binary operations.
                    castAST = self.binaryAST.exp1
                    self.exp1Lvalue = self.parseTACExpression(castAST.inner, self.insts)
                    self.exp1ToConvert = self.makeCast(castAST.inner.typeId, self.exp1Lvalue.convert(), castAST.typeId, self.insts)
                else:
                    self.exp1Lvalue = self.parseTACExpression(self.binaryAST.exp1, self.insts)
                    self.exp1ToConvert = self.exp1Lvalue

                # Process the expressions first. These values will store the operands.
                self.exp1: TACValue = self.exp1ToConvert.convert()
                self.exp2: TACValue = self.parseTACExpression(self.binaryAST.exp2, self.insts).convert()

                def pointerArithmetic(pointer: TACValue, index: TACValue) -> bool:
                    nonlocal dest

                    if not isinstance(pointer.valueType, (PointerDeclaratorType, ArrayDeclaratorType)) or \
                       not isinstance(index.valueType, BaseDeclaratorType) or \
                       not self.operator in (BinaryOperator.SUM, BinaryOperator.SUBTRACT):
                        return False
                    
                    if self.operator == BinaryOperator.SUBTRACT:
                        # Negate the index.
                        neg = self.createChild(TACUnary, UnaryOperator.NEGATION, index, self.insts)
                        index = neg.result

                    inst = self.createChild(TACAddToPointer, 
                                            pointer, index, pointer.valueType.declarator.getByteSize(),
                                            dest, self.insts)
                    dest = inst.result
                    return True
                
                def pointerSubtraction(pointer1: TACValue, pointer2: TACValue) -> bool:
                    nonlocal dest

                    if not isinstance(pointer1.valueType, (PointerDeclaratorType, ArrayDeclaratorType)) or \
                       not isinstance(pointer2.valueType, (PointerDeclaratorType, ArrayDeclaratorType)) or \
                       self.operator != BinaryOperator.SUBTRACT:
                        return False
                    
                    # Get the size of the referenced object.
                    pointer1Size = pointer1.valueType.declarator.getByteSize()

                    # Subtract the pointers (as integers).
                    pointer1.valueType = TypeSpecifier.ULONG.toBaseType()
                    pointer2.valueType = TypeSpecifier.ULONG.toBaseType()
                    diff = TACBinary(BinaryOperator.SUBTRACT, pointer1, pointer2, instructionsList=self.insts, parentTAC=self)
                    # Divide the difference by the size of pointer1.
                    div = TACBinary( 
                            BinaryOperator.DIVISION, 
                            diff.result, 
                            TACValue(True, TypeSpecifier.ULONG.toBaseType(), str(pointer1Size), self),
                            instructionsList=self.insts,
                            parentTAC=self
                    )
                    dest = div.result
                    return True

                pointerFuncHandlers = (
                    lambda: pointerArithmetic(self.exp1, self.exp2),
                    lambda: pointerArithmetic(self.exp2, self.exp1),
                    lambda: pointerSubtraction(self.exp1, self.exp2),
                    lambda: pointerSubtraction(self.exp2, self.exp1),
                )

                # Executes all handlers and short-circuits if one of the pointer operations matches.
                for handler in pointerFuncHandlers:
                    if handler():
                        # This is no longer a TACBinary.
                        self.removeThisFromInsts = True
                        break

        return dest

    def print(self) -> str:
        return f"Binary({self.operator.name}, {self.exp1}, {self.exp2}, {self.result})\n"
    
class TACCopy(TACInstruction):
    def __init__(self, src: TACValue, dst: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.src = src
        self.dst = dst
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        return self.dst

    def print(self) -> str:
        return f"Copy({self.src}, {self.dst})\n"
    
class TACGetAddress(TACInstruction):
    def __init__(self, src: TACValue, dst: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.src = src
        self.dst = dst
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        return self.dst

    def print(self) -> str:
        return f"GetAddress({self.src}, {self.dst})\n"

class TACLoad(TACInstruction):
    def __init__(self, srcPointer: TACValue, dst: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.src = srcPointer
        self.dst = dst
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        return self.dst

    def print(self) -> str:
        return f"Load({self.src}, {self.dst})\n"
    
class TACStore(TACInstruction):
    def __init__(self, src: TACValue, dstPointer: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.src = src
        self.dst = dstPointer
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        return self.dst

    def print(self) -> str:
        return f"Store({self.src}, {self.dst})\n"

class TACAddToPointer(TACInstruction):
    def __init__(self, pointer: TACValue, index: TACValue, scale: int, dst: TACValue,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        if not isinstance(pointer.valueType, (PointerDeclaratorType, ArrayDeclaratorType)) or \
           not isinstance(index.valueType, BaseDeclaratorType):
            raise ValueError(f"Invalid input types {pointer.valueType} or {index.valueType} in TACAddToPointer")

        self.pointer = pointer
        self.index = index
        self.scale = scale
        self.dst = dst
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        return self.dst

    def print(self) -> str:
        return f"AddToPointer({self.pointer} + {self.index}*{self.scale}, {self.dst})\n"
    
class TACCopyToOffset(TACInstruction):
    def __init__(self, src: TACValue, dst: TACValue, byteOffset: int,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.src = src
        self.dst = dst
        self.byteOffset = byteOffset
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        return self.dst

    def print(self) -> str:
        return f"CopyToOffset({self.src}, {self.dst} + {self.byteOffset})\n"
    
class TACCopyFromOffset(TACInstruction):
    def __init__(self, src: TACValue, byteOffset: int, dst: TACValue, 
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.src = src
        self.byteOffset = byteOffset
        self.dst = dst
        super().__init__(instructionsList, parentTAC)

    def parse(self)-> TACValue:
        return self.dst

    def print(self) -> str:
        return f"CopyFromOffset({self.src} + {self.byteOffset}, {self.dst})\n"

class TACJump(TACInstruction):
    def __init__(self, target: str,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.target = target
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> None:
        return None

    def print(self) -> str:
        return f"Jump({self.target})\n"

class TACJumpIfValue(TACInstruction):
    def __init__(self, condition: TACValue, value: TACValue, target: str,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.condition = condition
        self.value = value
        self.target = target
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> None:
        return None

    def print(self) -> str:
        return f"JumpIfValue({self.condition}, {self.value}, {self.target})\n"    

class TACJumpIfZero(TACInstruction):
    def __init__(self, condition: TACValue, target: str,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.condition = condition
        self.target = target
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> None:
        return None

    def print(self) -> str:
        return f"JumpIfZero({self.condition}, {self.target})\n"
    
class TACJumpIfNotZero(TACInstruction):
    def __init__(self, condition: TACValue, target: str,
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.condition = condition
        self.target = target
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> None:
        return None

    def print(self) -> str:
        return f"JumpIfNotZero({self.condition}, {self.target})\n"

class TACLabel(TACInstruction):
    LABEL_COUNT: int = 0
    
    def __init__(self, identifier: str, 
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.identifier = identifier
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> None:
        return None

    def print(self) -> str:
        return f"Label({self.identifier})\n"
    
    @staticmethod
    def getNewLabelName() -> str:
        # Autogenerate a new label identifier.
        identifier: str = f"label{TACLabel.LABEL_COUNT}"
        TACLabel.LABEL_COUNT += 1
        return identifier
    
class TACFunctionCall(TACInstruction):
    def __init__(self, identifier: str, returnType: DeclaratorType, arguments: list[TACValue],
                 instructionsList: list[TACInstruction], parentTAC: TAC | None = None) -> None:
        self.identifier = identifier
        self.returnType = returnType
        self.arguments = arguments
        super().__init__(instructionsList, parentTAC)

    def parse(self) -> TACValue:
        # After the function execution, this is where the return value will be stored.
        return TACValue(False, self.returnType)

    def print(self) -> str:
        argList = ', '.join([arg.print() for arg in self.arguments])
        return f"FunctionCall: {self.identifier}({argList})\n"
