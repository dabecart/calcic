"""
parser.py

Interprets the tokens given by the lexing stage. Applies the grammar and semantic rules of C99 and 
generates an Abstract Syntax Tree (AST) of the file. 

calcic. Written by @dabecart, 2026.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import enum
import copy
import math
import struct
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import ClassVar, TypeVar, overload
from .lexer import Token 
from .types import *

# Used to generate the verbose output.
PADDING_INCREMENT: int = 2

ASTType      = TypeVar("ASTType", bound="AST")
TExp         = TypeVar("TExp", bound="Exp")
TStmt        = TypeVar("TStmt", bound="Statement")
TBlockItem   = TypeVar("TBlockItem", bound="BlockItem")
TDeclaration = TypeVar("TDeclaration", bound="Declaration")
TInitializer = TypeVar("TInitializer", bound="Initializer")

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CONTEXT: To handle the state of variables and functions during the compilation.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

# Context information for each identifier.
class IdentifierType(enum.Enum):
    VARIABLE            = enum.auto()
    FUNCTION            = enum.auto()
    STATIC_VARIABLE     = enum.auto()
    CONSTANT_VARIABLE   = enum.auto()
    TYPEDEF             = enum.auto()

@dataclass
class IdentifierContext:
    identifierType: IdentifierType
    originalName: str
    mangledName: str
    alreadyDeclared: bool = True

@dataclass
class VariableIdentifier:
    idType: DeclaratorType
    constValue: Constant|None = None

@dataclass
class FunctionIdentifier:
    name: str
    arguments: list[ParameterInformation]
    returnType: DeclaratorType
    storageClass: StorageClass|None
    isGlobal: bool
    alreadyDefined: bool = False

@dataclass
class StaticVariableContext:
    storageClass: StorageClass|None
    idType: DeclaratorType
    mangledName: str
    # True if it has external linkage.
    isGlobal: bool
    tentative: bool
    initialization: list[Constant]

@dataclass
class ConstantVariableContext:
    name: str
    idType: DeclaratorType
    initialization: list[Constant]

@dataclass
class TypedefContext:
    originalName: str
    idType: DeclaratorType

# Used to manage structs, unions, enums and typedefs.
class TaggableType(enum.Enum):
    STRUCT  = "struct"
    UNION   = "union"
    ENUM    = "enum"

@dataclass(order=True)
class TagContext:
    tagType: TaggableType
    creationOrder: int
    originalName: str
    mangledName: str
    # To verify wether the tag element can be shadowed or not.
    alreadyDeclared: bool = True

@dataclass
class SwitchContext:
    name: str
    # Type of the variable controlling the switch.
    controlType: DeclaratorType
    cases: list[CaseStatement]          = field(default_factory=list)
    defaultCase: DefaultStatement|None  = None

@dataclass
class LabelContext:
    originalName: str
    mangledName: str
    declared: bool = False
    # The token after the goto.
    gotoToken: Token|None = None

@dataclass
class Context:
    # --- IDENTIFIERS ---
    # Stores the information about ordinary identifiers used by variables, functions and typedefs. 
    # Key is the original name.
    identifierMap: dict[str, IdentifierContext]              = field(default_factory=dict)
    # Stores the attributes of functions. Key is the mangled name.
    variablesMap: dict[str, VariableIdentifier]              = field(default_factory=dict)
    # Stores the attributes of functions. Key is the original name (function names aren't mangled).
    functionMap: dict[str, FunctionIdentifier]               = field(default_factory=dict)
    # Stores the attributes of variables which are stored on the .data section. Key is the original 
    # name.
    staticVariablesMap: dict[str, StaticVariableContext]     = field(default_factory=dict)
    # Stores the attributes of variables which are stored on the .rodata section.
    constantVariablesMap: dict[str, ConstantVariableContext] = field(default_factory=dict)
    # Contains the declarators of the typedef, keyed by the mangled identifier.
    typedefMap: dict[str, TypedefContext]                    = field(default_factory=dict)

    # --- TAGS ---
    # Stores the information about the tags used by structs, unions and enums. Key is the mangled 
    # name.
    tagsMap: dict[str, TagContext]                           = field(default_factory=dict)
    # Contains the type of the struct, keyed by the mangled identifier. 
    structMap: dict[str, TypeSpecifier]                      = field(default_factory=dict)
    # Contains the type of the union, keyed by the mangled identifier. 
    unionMap: dict[str, TypeSpecifier]                       = field(default_factory=dict)
    # Contains the type of the enum, keyed by the mangled identifier. 
    enumMap: dict[str, TypeSpecifier]                        = field(default_factory=dict)

    loopTracking: list[str]                     = field(default_factory=list)
    switchTracking: list[SwitchContext]         = field(default_factory=list)
    labelTracking: dict[str,LabelContext]       = field(default_factory=dict)
    insideAFunction: bool                       = False
    # The function the context is currently in.
    insideFunctionName: str                     = ""

    # Use it to assign the break to a loop or a switch.
    _switchOrLoop: list[str]                    = field(default_factory= lambda:[""])

    @contextmanager
    def newContext(self):
        # When entering the new scope, create a copy of the variable map with the same keys but with
        # all variables set to alreadyDeclared = False. Variables can be redeclared inside a new
        # scope. 
        previousIDMap = self.identifierMap
        self.identifierMap = copy.deepcopy(self.identifierMap)
        for id in self.identifierMap.values():
            id.alreadyDeclared = False
        
        # Do the same with the tags map.
        previousTagsMap = self.tagsMap
        self.tagsMap = copy.deepcopy(self.tagsMap)
        for singleTag in self.tagsMap.values():
            singleTag.alreadyDeclared = False

        yield

        # Restore the variable map.
        self.identifierMap.clear()
        self.identifierMap.update(previousIDMap)
        # Restore the tags map.
        self.tagsMap.clear()
        self.tagsMap.update(previousTagsMap)

    IDENTIFIER_COUNTER: ClassVar[int] = 0
    def mangleIdentifier(self, originalName: str) -> str:
        mangledName= f"{originalName}.{Context.IDENTIFIER_COUNTER}"
        Context.IDENTIFIER_COUNTER += 1
        return mangledName
    
    def addVariableIdentifier(self, 
                              originalIdentifier: str, mangledIdentifier: str, 
                              varType: DeclaratorType, constantValue: Constant|None = None):
        idContext = IdentifierContext(
            identifierType=IdentifierType.VARIABLE,
            originalName=originalIdentifier,
            mangledName=mangledIdentifier
        )
        # Add variable to context.
        self.identifierMap[originalIdentifier] = idContext
        # Add variable to the variables map.
        self.variablesMap[mangledIdentifier] = VariableIdentifier(
            idType=varType,
            constValue=constantValue
        )

    def addFunctionIdentifier(self, func: FunctionDeclaration):
        idContext = IdentifierContext(
            identifierType=IdentifierType.FUNCTION,
            originalName=func.identifier,
            mangledName=func.identifier
        )
        # Add the function identifier to the current scope.
        self.identifierMap[func.identifier] = idContext
        # Do not add the function information yet, as the function may not be defined.

    def addTypedefIdentifier(self, originalIdentifier: str, mangledIdentifier: str, varType: DeclaratorType):
        idContext = IdentifierContext(
            identifierType=IdentifierType.TYPEDEF,
            originalName=originalIdentifier,
            mangledName=mangledIdentifier
        )
        # Add the function identifier to the current scope.
        self.identifierMap[originalIdentifier] = idContext
        # Add the typedef information to the map.
        self.typedefMap[mangledIdentifier] = TypedefContext(
            originalName=originalIdentifier,
            idType=varType
        )
    
    LOOP_COUNTER: ClassVar[int] = 0
    @contextmanager
    def enterLoop(self):
        loopLabel = f"loop.{Context.LOOP_COUNTER}"
        Context.LOOP_COUNTER += 1
        self.loopTracking.append(loopLabel)
        self._switchOrLoop.append("loop")

        yield loopLabel
        
        self.loopTracking.pop()
        self._switchOrLoop.pop()

    SWITCH_COUNTER: ClassVar[int] = 0
    @contextmanager
    def enterSwitch(self, controlType: DeclaratorType):
        switchLabel = f"switch.{Context.SWITCH_COUNTER}"
        Context.SWITCH_COUNTER += 1
        newSwitch = SwitchContext(switchLabel, controlType)

        self.switchTracking.append(newSwitch)
        self._switchOrLoop.append("switch")
        
        yield switchLabel
        
        self.switchTracking.pop()
        self._switchOrLoop.pop()

    def getCurrentSwitch(self) -> SwitchContext:
        return self.switchTracking[-1]

    def getSwitchOrLoop(self) -> str:
        return self._switchOrLoop[-1]

    LABEL_COUNTER: ClassVar[int] = 0
    def mangleLabelName(self, originalName: str) -> str:
        ret = LabelContext(
            originalName=originalName, 
            mangledName= f"{originalName}.{Context.LABEL_COUNTER}"
        )
        
        Context.LABEL_COUNTER += 1

        # Include it in the label tracking map.
        self.labelTracking[originalName] = ret
        return ret.mangledName

    @contextmanager
    def enterFunction(self, functionName: str):
        if functionName not in self.functionMap:
            raise ValueError(f"Function {functionName} has not been declared yet")

        self.insideAFunction = True
        self.insideFunctionName = functionName
        self.loopTracking.clear()
        self.switchTracking.clear()
        self.labelTracking.clear()
        self._switchOrLoop = [""]

        # A function also yields a new context.
        with self.newContext():
            yield

        self.insideAFunction = False
        self.insideFunctionName = ""
        for label in self.labelTracking.values():
            if not label.declared:
                if label.gotoToken is None:
                    raise ValueError(f"Missing declaration of label {label.originalName}")
                else:
                    raise ValueError(f"{label.gotoToken.getPosition()} Missing declaration of label {label.originalName}")

    # Use it to return the most recent struct, union or enum.
    STRUCT_ORDER: int = 0
    UNION_ORDER: int = 0
    ENUM_ORDER: int = 0
    TYPEDEF_ORDER: int = 0
    def addStruct(self, structure: StructDeclaration):
        structContext = TagContext(
            tagType=TaggableType.STRUCT,
            creationOrder=Context.STRUCT_ORDER,
            originalName=structure.originalIdentifier,
            mangledName=structure.identifier
        )
        Context.STRUCT_ORDER += 1

        # Add struct to context using the mangled identifier.
        self.tagsMap[structure.identifier] = structContext
        # Add type to the structure map.
        self.structMap[structure.identifier] = structure.typeId

    # Use it to return the most recent union.
    def addUnion(self, union: UnionDeclaration):
        unionContext = TagContext(
            tagType=TaggableType.UNION,
            creationOrder=Context.UNION_ORDER,
            originalName=union.originalIdentifier,
            mangledName=union.identifier,
        )
        Context.UNION_ORDER += 1

        # Add union to context using the mangled identifier.
        self.tagsMap[union.identifier] = unionContext
        # Add type to the union map.
        self.unionMap[union.identifier] = union.typeId

    def addEnum(self, enum: EnumDeclaration):
        enumContext = TagContext(
            tagType=TaggableType.ENUM,
            creationOrder=Context.ENUM_ORDER,
            originalName=enum.originalIdentifier,
            mangledName=enum.identifier,
        )
        Context.ENUM_ORDER += 1

        # Add enum to context using the mangled identifier.
        self.tagsMap[enum.identifier] = enumContext
        # Add type to the enum map.
        self.enumMap[enum.identifier] = enum.typeId

    # Finds an object named "originalName" of type "objectType" in the tags map.
    def getTagObjectFromOriginalName(self, tagType: TaggableType, originalName: str) -> TagContext|None:
        ret: list[TagContext] = []
        for cntx in self.tagsMap.values():
            if cntx.tagType == tagType and cntx.originalName == originalName:
                ret.append(cntx)

        if len(ret) > 0:
            # Order the list and get the most recent struct using the "creationOrder" index.
            return sorted(ret)[-1]
        
        return None
    
    def getStructFromOriginalName(self, originalName: str) -> TagContext|None:
        return self.getTagObjectFromOriginalName(TaggableType.STRUCT, originalName)

    def getUnionFromOriginalName(self, originalName: str) -> TagContext|None:
        return self.getTagObjectFromOriginalName(TaggableType.UNION, originalName)

    def getEnumFromOriginalName(self, originalName: str) -> TagContext|None:
        return self.getTagObjectFromOriginalName(TaggableType.ENUM, originalName)

    # Returns the conflicting type with the same tag name.
    def getConflictsForTaggableType(self, tagType: TaggableType, tagName: str) -> tuple[TagContext|None, TypeSpecifier|None]:
        toCheckTypes: set[TaggableType] = set([t for t in TaggableType]) - {tagType}
        for t in toCheckTypes:
            matched = self.getTagObjectFromOriginalName(t, tagName)
            if matched:
                if matched.tagType == TaggableType.STRUCT:
                    return (matched, self.structMap[matched.mangledName])
                elif matched.tagType == TaggableType.UNION:
                    return (matched, self.unionMap[matched.mangledName])
                elif matched.tagType == TaggableType.ENUM:
                    return (matched, self.enumMap[matched.mangledName])
                else:
                    raise ValueError()
        return (None, None)

    def completeStruct(self, structure: StructDeclaration):
        # Modify the parameters inside the context.
        ctx = self.getStructFromOriginalName(structure.originalIdentifier)
        if ctx is None:
            raise ValueError(f"Could not find the struct {structure.originalIdentifier} in the current context")

        # Get the type from the map and update the values.
        typeCtx = self.structMap[ctx.mangledName]
        typeCtx.byteSize = structure.byteSize
        typeCtx.alignment = structure.alignment
        typeCtx.members = structure.membersToParamInfo()

    def completeUnion(self, union: UnionDeclaration):
        # Modify the parameters inside the context.
        ctx = self.getUnionFromOriginalName(union.originalIdentifier)
        if ctx is None:
            raise ValueError(f"Could not find the union {union.originalIdentifier} in the current context")

        # Get the type from the map and update the values.
        typeCtx = self.unionMap[ctx.mangledName]
        typeCtx.byteSize = union.byteSize
        typeCtx.alignment = union.alignment
        typeCtx.members = union.membersToParamInfo()

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AST BASE CLASS
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
class AST(ABC):
    def __init__(self, tokens: list[Token], context: Context|None = None, parentAST: AST|None = None, *args) -> None:
        super().__init__()
        self.tokens = tokens
        self.context = context if context is not None else Context()
        # AST containing this AST node.
        self.parent = parentAST
        # Last parsed token inside this AST.
        self.lastToken: None|Token = None
        self.parse(*args)

    @abstractmethod
    def parse(self, *args):
        pass

    @abstractmethod
    def print(self, padding: int) -> str:
        pass

    # Returns the last token found on the current AST. If it does not exist it searches recursively
    # through the parents.
    def getLastChild(self) -> Token|None:
        if self.lastToken is not None:
            if isinstance(self.lastToken, Token):
                return self.lastToken
            raise ValueError("Found an invalid type in the getLastChild function")
        if self.parent is not None:
            return self.parent.getLastChild()
        
        return None
    
    def pop(self) -> Token:
        tok = self.tokens.pop(0)
        self.lastToken = tok
        return tok

    def raiseError(self, msg: str):
        lastToken = self.getLastChild()
        if lastToken is not None:
            raise ValueError(f"{lastToken.getPosition()} {msg}")
        else:
            raise ValueError(msg)
        
    def raiseWarning(self, msg: str):
        lastToken = self.getLastChild()
        if lastToken is not None:
            print(f"{lastToken.getPosition()} Warning: {msg}")
        else:
            print(f"Warning: {msg}")

    def peek(self, index: int = 0) -> Token:
        if index >= len(self.tokens):
            raise ValueError(f"Cannot peek at {index}")
        return self.tokens[index]

    def expect(self, *expectedIDs: str) -> Token:
        # If there are no remaining tokens on the list, go to the parent, get the line location and print the error.
        if not self.tokens:
            lastToken = self.getLastChild()
            if lastToken is not None:
                raise ValueError(f"{lastToken.getPosition()} Expected {' or '.join(expectedIDs)} after {lastToken.value}")
            else:
                raise ValueError(f"Expected {' or '.join(expectedIDs)} but no token was found")

        tok = self.peek()
        if tok.id not in expectedIDs:
            raise ValueError(
                f"{tok.getPosition()} Expected {' or '.join(expectedIDs)} but found {tok.id}"
            )

        self.pop()
        return tok
    
    @overload
    def createChild(self, childType: type[TStmt], *args) -> TStmt: ...
    @overload
    def createChild(self, childType: type[TExp], *args) -> TExp: ...
    @overload
    def createChild(self, childType: type[TBlockItem], *args) -> TBlockItem: ...
    @overload
    def createChild(self, childType: type[TDeclaration], *args) -> TDeclaration: ...
    @overload
    def createChild(self, childType: type[TInitializer], *args) -> TInitializer: ...
    @overload
    def createChild(self, childType: type[ASTType], *args) -> ASTType: ...

    def createChild(self, childType: type[AST], *args):
        if childType is Statement:
            ret = self._parseStatement()
        elif childType is Exp:
            ret = self._parseExpression()
        elif childType is BlockItem:
            ret = self._parseBlockItem()
        elif childType is Declaration:
            ret = self._parseDeclaration()
        elif childType is Initializer:
            ret = self._parseInitializer(*args)
        else:
            ret = childType(self.tokens, self.context, self, *args)
        # Once created, set the last token of the parent to be the last token parsed by the child.
        if ret.lastToken is not None:
            self.lastToken = ret.lastToken
        return ret

    def _parseStatement(self) -> Statement:
        tok = self.peek()

        match tok.id:
            case "return":
                return self.createChild(ReturnStatement)
            case "if":
                return self.createChild(IfStatement)
            case ";":
                return self.createChild(NullStatement)
            case "{":
                return self.createChild(CompoundStatement)
            case "break":
                return self.createChild(BreakStatement)
            case "continue":
                return self.createChild(ContinueStatement)
            case "while":
                return self.createChild(WhileStatement)
            case "do":
                return self.createChild(DoWhileStatement)
            case "for":
                return self.createChild(ForStatement)
            case "switch":
                return self.createChild(SwitchStatement)
            case "case":
                return self.createChild(CaseStatement)
            case "default":
                return self.createChild(DefaultStatement)
            case "goto":
                return self.createChild(GotoStatement)
            case "identifier":
                if self.peek(1).id == ":":
                    return self.createChild(LabelStatement)

        # Try to parse an expression statement.
        return self.createChild(ExpressionStatement)

    def _parseExpression(self, minPrecedence: int = 0) -> Exp:
        # Parse a single factor.
        leftTerm = self._parseFactor()

        tok = self.peek()
        # While the next token is an operator and has a bigger precedence than the previous 
        # operator...
        while (op := BinaryOperator.toBinaryOperator(tok)) is not None \
            and (opPrecedence := op.getPrecedence()) >= minPrecedence:
            # Consume the token from the list.
            tok = self.pop()
            if op == BinaryOperator.ASSIGNMENT:
                # Right-associative.
                # Get the right term of the expression.
                rightTerm = self._parseExpression(opPrecedence)
                # Create the assignment leftTerm = rightTerm.
                leftTerm = self.createChild(Assignment, leftTerm, rightTerm)
            elif op.isCompound():
                # Right-associative.
                # Get the right term of the expression.
                rightTerm = self._parseExpression(opPrecedence)
                # Operate the current left term with the right term. In the TAC stage, this value
                # will be stored in the leftTerm.
                leftTerm = self.createChild(Binary, op, leftTerm, rightTerm) 
            elif op == BinaryOperator.CONDITIONAL:
                # Right-associative but funky (is a ternary operator).
                ifExp = self._parseExpression(0)
                self.expect(":")
                thenExp = self._parseExpression(opPrecedence)
                leftTerm = self.createChild(TernaryConditional, leftTerm, ifExp, thenExp)
            else:
                # Left-associative.
                # Get the right term of the expression.
                rightTerm = self._parseExpression(opPrecedence + 1)
                # Create a binary expression.
                leftTerm = self.createChild(Binary, op, leftTerm, rightTerm)
            # Go to the next token.
            tok = self.peek()

        return leftTerm
    
    def _parseFactor(self) -> Exp:
        tok = self.peek()

        if tok.id == "(" and self.isTokenACastIdentifierType(self.peek(1)):
            return self.createChild(Cast)
        
        return self._parseUnaryExpression()

    def _parseUnaryExpression(self) -> Exp:
        tok = self.peek()

        if (preOp := UnaryOperator.toPreUnaryOperator(tok)) is not None:
            # Pop the pre-unary.
            self.pop()

            # &*<exp> is a special case, where the AddressOf and Dereference are skipped.
            if preOp == UnaryOperator.ADDRESS_OF and self.peek().id == "*":
                self.pop()
                return self._parseFactor()

            match preOp:
                case UnaryOperator.DEREFERENCE:
                    return self.createChild(Dereference)

                case UnaryOperator.ADDRESS_OF:
                    return self.createChild(AddressOf)

                case _:
                    innerFactor = self._parseFactor()
                    unaryChild = self.createChild(Unary, preOp, innerFactor)
                    if unaryChild.unaryOperator == UnaryOperator.PLUS:
                        # This does nothing on the result, return the inner factor.
                        return innerFactor
                    return unaryChild

        if tok.id == "sizeof":
            if self.peek(1).id == "(" and self.isTokenAnIdentifierType(self.peek(2)):
                return self.createChild(SizeOfType)
            else:
                return self.createChild(SizeOf)

        return self._parsePostfixExpression()
    
    def _parsePostfixExpression(self) -> Exp:
        primary: Exp = self._parsePrimaryExpression()
        
        ret: Exp = primary
        while True:
            postTok = self.peek()
            if (postOp := UnaryOperator.toPostUnaryOperator(postTok)) is not None:
                # Pop the post-unary.
                self.pop()
                ret = self.createChild(Unary, postOp, ret)
            elif postTok.id == "[":
                # This is a subscript.
                ret = self.createChild(Subscript, ret)
            elif postTok.id == ".":
                # Structure/union dot.
                ret = self.createChild(Dot, ret)
            elif postTok.id == "->":
                # Pointer structure/union arrow.
                ret = self.createChild(Arrow, ret)
            else:
                # No postfix.
                break
        
        return ret

    def _parsePrimaryExpression(self) -> Exp:
        tok = self.peek()
        ret: Exp|None = None
        match tok.id:
            case "constant":
                # A number (without any ending identifier such as L or ul) can be considered an int 
                # or a long.
                try:
                    intVal = tok.parseIntegerToken()
                except Exception as e:
                    self.raiseError(str(e))
                if intVal >= 0x80000000:
                    ret = self.createChild(LongConstant)
                else:
                    ret = self.createChild(IntConstant)
            
            case "unsigned_constant":
                # A number ending in u can be either considered an uint or ulong.
                try:
                    intVal = tok.parseIntegerToken()
                except Exception as e:
                    self.raiseError(str(e))
                if intVal >= 0x100000000:
                    ret = self.createChild(ULongConstant)
                else:
                    ret = self.createChild(UIntConstant)

            case "long_constant":
                ret = self.createChild(LongConstant)

            case "unsigned_long_constant":
                ret = self.createChild(ULongConstant)

            case "double_constant":
                ret = self.createChild(DoubleConstant)

            case "float_constant":
                ret = self.createChild(FloatConstant)

            case "character":
                # Characters such as 'e' are taken as integers in C.
                ret = self.createChild(IntConstant)

            case "string":
                ret = self.createChild(String)

            case "identifier":
                if self.peek(1).id == "(":
                    ret = self.createChild(FunctionCall)
                elif tok.value in self.context.identifierMap is not None:
                    # Remove the token.
                    self.pop()
                    
                    ctx = self.context.identifierMap[tok.value]
                    if ctx.identifierType == IdentifierType.VARIABLE and \
                       (constVal := self.context.variablesMap[ctx.mangledName].constValue) is not None:
                        # If it's a constant value, return it directly.
                        ret = constVal
                    else:
                        ret = self.createChild(Variable, tok.value)

            case "(":
                self.expect("(")
                ret = self._parseExpression()
                self.expect(")")

        if ret is None:
            self.raiseError(f"Unexpected token {tok} when parsing a Factor")

        return ret

    def isTokenAnIdentifierType(self, tok: Token) -> bool:
        basicSpecifier = tok.id in Token.SPECIFIER
        typedefSpecifier = tok.id == "identifier" and \
                           tok.value in self.context.identifierMap and \
                           self.context.identifierMap[tok.value].identifierType == IdentifierType.TYPEDEF
        return basicSpecifier or typedefSpecifier

    def isTokenACastIdentifierType(self, tok: Token) -> bool:
        basicSpecifier = tok.id in Token.CAST_SPECIFIER
        typedefSpecifier = tok.id == "identifier" and \
                           tok.value in self.context.identifierMap and \
                           self.context.identifierMap[tok.value].identifierType == IdentifierType.TYPEDEF
        return basicSpecifier or typedefSpecifier

    def _parseBlockItem(self) -> BlockItem:
        # A block can either be a declaration or a statement.
        peekToken = self.peek()
        if self.isTokenAnIdentifierType(peekToken) or peekToken.id == "typedef":
            return self.createChild(Declaration)
        else:
            return self.createChild(Statement)

    def _parseDeclaration(self) -> Declaration:
        peekId = self.peek().id
        if peekId == "typedef":
            ret = self.createChild(TypedefDeclaration)
            self.expect(";")
        else:
            storageClass, declType, typeDeclaration = self.getStorageClassAndDeclaratorType()

            # If we find a semicolon, what we parsed was a simple type definition (struct, union or 
            # enum).
            if self.peek().id == ";":
                self.pop()
                if typeDeclaration is None:
                    self.raiseError("Expected a type declaration")
                ret = typeDeclaration
            else:
                declarator = self.createChild(TopDeclarator)
                info: DeclaratorInformation = declarator.process(declType)

                if isinstance(info.type, FunctionDeclaratorType):
                    ret = self.createChild(FunctionDeclaration, storageClass, info)
                else:
                    ret = self.createChild(VariableDeclaration, storageClass, info)
        
        return ret

    def _parseInitializer(self, *args) -> Initializer:
        if isinstance(args[0], ArrayDeclaratorType):
            return self.createChild(CompoundInitializer, *args)
        elif isinstance(args[0], BaseDeclaratorType) and args[0].baseType.name in ("STRUCT", "UNION"):
            # Structs/unions can be initialized from {} or from single initializers (such as 
            # function calls or copies from another struct).
            if self.peek().id == "{":
                return self.createChild(CompoundInitializer, *args)
            else:
                return self.createChild(SingleInitializer, *args)
        else:
            return self.createChild(SingleInitializer, *args)

    # - expectsStorageClass: True when a storage class can be accepted.
    # - creatingVariable: It is used to differentiate between struct/union/enum/typedef declarations 
    # and variable declarations. This impacts the shadowing logic.
    def getStorageClassAndDeclaratorType(self, expectsStorageClass = True) -> tuple[StorageClass|None, DeclaratorType, Declaration|None]:
        storageClass: StorageClass|None = None
        typeQualifiers: TypeQualifier = TypeQualifier()
        typeDeclaration: Declaration|None = None
        idType: TypeSpecifier

        idTypeAlreadyDefined: bool = False
        typeSet: set[str] = set()

        # Parse qualifiers and common types.
        while (tok := self.peek()).id in Token.SPECIFIER:
            if tok.id in Token.TYPE_SPECIFIER:
                # Type specifiers cannot be repeated (there's long long, but I haven't implemented 
                # it).
                if tok.id in typeSet:
                    self.raiseError(f"Already defined as {tok.id}")

                # Taggable types: TYPE{struct, union, enum} <identifier>
                if tok.id in ("struct", "union", "enum"):
                    if idTypeAlreadyDefined:
                        self.raiseError("Conflicting types")
                    idTypeAlreadyDefined = True

                    # Parse a declaration.
                    if tok.id == "struct":
                        typeDeclaration = self.createChild(StructDeclaration)
                        # The type is the same as the declaration type.
                        idType = typeDeclaration.typeId
                    elif tok.id == "union":
                        typeDeclaration = self.createChild(UnionDeclaration)
                        # The type is the same as the declaration type.
                        idType = typeDeclaration.typeId
                    elif tok.id == "enum":
                        # Enums are a bit more finicky to work with, as they cannot be redeclared 
                        # without having its initializer list.
                        tagType = TaggableType(tok.id)
                        typeIdentifier = self.peek(1).value
                        objectType = self.context.getTagObjectFromOriginalName(tagType, typeIdentifier)     
                        newDeclaration = objectType is None or self.peek(2).id == "{"
                        
                        if newDeclaration:
                            typeDeclaration = self.createChild(EnumDeclaration)
                            # The type is the same as the declaration type.
                            idType = typeDeclaration.typeId
                        else:
                            # To shut Pylance...
                            if objectType is None: raise ValueError()
                            idType = self.context.enumMap[objectType.mangledName]
                            # Pop the keyword and the identifier.
                            self.pop()
                            self.pop()
                    else:
                        raise ValueError()
                    
                else:
                    # Pop the type when it's not a struct, union or enum.
                    self.pop()

                typeSet.add(tok.id)
                
            elif tok.id in Token.TYPE_QUALIFIER:
                match tok.id:
                    case "const":
                        # Duplicated const keywords are valid.
                        typeQualifiers.const = True
                    case _:
                        self.raiseError(f"Invalid specifier: {tok.id}")
                
                self.pop()

            elif tok.id in Token.STORAGE_QUALIFIER:
                if not expectsStorageClass:
                    self.raiseError("Invalid storage class")
                if storageClass is not None:
                    self.raiseError(f"Variable already defined as {storageClass.value}")
                match tok.id:
                    case "static":
                        storageClass = StorageClass.STATIC
                    case "extern":
                        storageClass = StorageClass.EXTERN

                self.pop()

            else:
                self.raiseError(f"Unexpected qualifier: {tok.id}")

        if len(typeSet) == 0:
            # It could be that the next token is the identifier of a typedef type.
            peekToken = self.peek()
            if peekToken.id == "identifier" and \
               peekToken.value in self.context.identifierMap and \
               (ctx := self.context.identifierMap[peekToken.value]).identifierType == IdentifierType.TYPEDEF:
                # Found a typedef alias, pop the alias.
                self.pop()

                # Return the original type and apply the qualifiers to the type.
                aliasedType = self.context.typedefMap[ctx.mangledName].idType.copy()
                if len(typeQualifiers.toSet()) > 0:
                    if isinstance(aliasedType, (BaseDeclaratorType, PointerDeclaratorType)):
                        aliasedType.qualifiers = typeQualifiers
                    else:
                        self.raiseError(f"Cannot use {typeQualifiers}with type {aliasedType}")
                
                return (storageClass, aliasedType, typeDeclaration)

            # If it's none of that, raise an error.
            tok = self.pop()
            self.raiseError(f"Expected a type, not {tok.value}")

        if {"signed", "unsigned"} <= typeSet:
            self.raiseError("Variable declared as signed and unsigned at the same time")

        if typeSet in ({"unsigned", "long"}, {"unsigned", "long", "int"}):
            idType = TypeSpecifier.ULONG
        elif typeSet in ({"long"}, {"signed", "long"}, {"long", "int"}, {"signed", "long", "int"}):
            idType = TypeSpecifier.LONG
        elif typeSet in ({"unsigned", "int"}, {"unsigned"}):
            idType = TypeSpecifier.UINT
        elif typeSet in ({"int"}, {"signed", "int"}, {"signed"}):
            idType = TypeSpecifier.INT
        elif typeSet in ({"unsigned", "short"}, {"unsigned", "short", "int"}):
            idType = TypeSpecifier.USHORT
        elif typeSet in ({"short"}, {"signed", "short"}, {"short", "int"}, {"signed", "short", "int"}):
            idType = TypeSpecifier.SHORT
        elif typeSet == {"unsigned", "short"}:
            idType = TypeSpecifier.USHORT
        elif typeSet == {"char"}:
            idType = TypeSpecifier.CHAR
        elif typeSet == {"signed", "char"}:
            idType = TypeSpecifier.SIGNED_CHAR
        elif typeSet == {"unsigned", "char"}:
            idType = TypeSpecifier.UCHAR
        elif typeSet == {"double"}:
            idType = TypeSpecifier.DOUBLE
        elif typeSet == {"float"}:
            idType = TypeSpecifier.FLOAT
        elif typeSet == {"void"}:
            idType = TypeSpecifier.VOID
        elif typeSet in ({"struct"}, {"union"}, {"enum"}):
            # Value already set a few lines above.
            pass
        else:
            self.raiseError(f"Invalid combination of type specifiers: {', '.join(typeSet)}")

        return (storageClass, idType.toBaseType(typeQualifiers), typeDeclaration)
    
    def parseConstantFromType(self, constantType: DeclaratorType, strict: bool = False) -> list[Constant]:
        if isinstance(constantType, BaseDeclaratorType):
            baseConstType = constantType.baseType

            # Structs and unions cannot be constructed with the common "Initializer" class as they 
            # initialize members with single "zeroed" objects, and here we use ZeroPaddingInitializer.
            if baseConstType.name == "STRUCT":
                # Get the members array.
                members = baseConstType.getMembers()

                # Iterate over the members.
                ret: list[Constant] = []

                memoryOffset: int = 0
                memberIndex: int = 0

                self.expect("{")
                parsed = self.parseConstantFromType(members[0].type, strict)
                memoryOffset += members[0].type.getByteSize()
                ret.extend(parsed)

                while self.peek().id == ",":
                    memberIndex += 1
                    self.pop()

                    if self.peek().id == "}":
                        break

                    if memberIndex >= len(members):
                        self.raiseError(f"Too many initializers for {constantType}")

                    paddingBytes = members[memberIndex].offset - memoryOffset
                    if paddingBytes > 0:
                        padding = self.createChild(ZeroPaddingInitializer, 
                                                    TypeSpecifier.VOID.toBaseType(), paddingBytes)
                        ret.append(padding)

                    ret.extend(self.parseConstantFromType(members[memberIndex].type, strict))
                    memoryOffset += members[memberIndex].type.getByteSize() + paddingBytes

                finalPadding = baseConstType.getByteSize() - memoryOffset
                if finalPadding > 0:
                    padding = self.createChild(ZeroPaddingInitializer, 
                                                TypeSpecifier.VOID.toBaseType(), finalPadding)
                    ret.append(padding)

                self.expect("}")
                return ret
            
            if baseConstType.name == "UNION":
                members = baseConstType.getMembers()
                self.expect("{")
                ret: list[Constant] = self.parseConstantFromType(members[0].type, strict)
                self.expect("}")

                # Even though to initialize an union you parse a constant from the first type in 
                # the declaration, its size must be taken into account. An union of an int and 
                # double is initialized with an int, but its size is that of the double.
                bytePadding: int = baseConstType.getByteSize() - members[0].type.getByteSize()
                if bytePadding > 0:
                    padding = self.createChild(ZeroPaddingInitializer, 
                                                TypeSpecifier.VOID.toBaseType(), bytePadding)
                    ret.append(padding)

                return ret

            # Static evaluation, parse the initializer.
            initializer = self.createChild(Initializer, constantType)
            if strict and isinstance(initializer, SingleInitializer):
                if baseConstType.isInteger() and not initializer.precastType.isInteger():
                    self.raiseError("Expected an integer constant")
            constValues: list[StaticEvalValue] = initializer.staticEval()

            match baseConstType.name:
                case "LONG":
                    constClass = LongConstant
                case "ULONG":
                    constClass = ULongConstant
                case "INT":
                    constClass = IntConstant
                case "UINT":
                    constClass = UIntConstant
                case "SHORT":
                    constClass = ShortConstant
                case "USHORT":
                    constClass = UShortConstant
                case "CHAR" | "SIGNED_CHAR":
                    constClass = CharConstant
                case "UCHAR":
                    constClass = UCharConstant
                case "DOUBLE":
                    constClass = DoubleConstant
                case "FLOAT":
                    constClass = FloatConstant
                case _:
                    self.raiseError("Invalid constant type")
            
            return [self.createChild(constClass, constValues[0].value)]

        elif isinstance(constantType, PointerDeclaratorType):
            if self.peek().id == "string":
                # A char* can be initialized with a string. The string is stored as a constant and 
                # a pointer to said constant is stored in the variable. Be mindful that char is 
                # different to signed char and unsigned char, in a semantic sense.
                if not isinstance(constantType.declarator, BaseDeclaratorType) or \
                   not constantType.declarator.baseType == TypeSpecifier.CHAR:
                    self.raiseError("Cannot initialize a non char pointer with a string")

                strAST: String = self.createChild(String)
                # Start with .L so that the linker "hides" this constant.
                constantName: str = self.context.mangleIdentifier(".Lstr")

                self.context.constantVariablesMap[constantName] = ConstantVariableContext(
                    name=constantName,
                    idType=strAST.typeId,
                    initialization=strAST.toConstantsList()
                )
                
                cnst = self.createChild(PointerInitializer, constantType, constantName, 0)
            else:
                # Static evaluation, parse the initializer.
                initializer = self.createChild(Initializer, constantType)
                constValues: list[StaticEvalValue] = initializer.staticEval()
                if len(constValues) > 1:
                    self.raiseError("Expected a single initializer")

                constVal = constValues[0]
                if constVal.valType != StaticEvalType.POINTER:
                    raise ValueError()

                if constVal.value == "":
                    # Integer has been converted into pointer.
                    cnst = self.createChild(ULongConstant, constVal.pointerOffset)
                else:
                    # Another variable has been used to create this pointer.
                    cnst = self.createChild(PointerInitializer, 
                                            constantType, 
                                            constVal.value, 
                                            constVal.pointerOffset)
            return [cnst]
        
        elif isinstance(constantType, ArrayDeclaratorType):
            # This list contains a flat array of the variables.
            ret: list[Constant] = []

            def processCompoundInitializer(typ: ArrayDeclaratorType):
                currentLevel: list[Constant] = []
                # Number of children. Do not mix with len(currentLevel), remember memory is linear
                # for constant arrays: if an inner children is an array, the number of items added
                # to currentLevel are the size of the array, but that's only one child more of the 
                # top or current level array.
                currentLevelChildren: int

                if self.peek().id == "string":
                    if not isinstance(typ.declarator, BaseDeclaratorType) or \
                       not typ.declarator.baseType.isCharacter():
                        self.raiseError("Cannot initialize a non char array with a string")

                    stringValue = self.createChild(String).value
                    chConsts = [CharConstant([], None, None, str(ord(ch))) for ch in stringValue]
                    currentLevel.extend(chConsts)
                    
                    currentLevelChildren = len(stringValue)

                else:
                    self.expect("{")

                    # Fill with constants. Must be at least one.
                    if isinstance(typ.declarator, ArrayDeclaratorType):
                        # Recursive calls to initialize inner arrays.
                        processCompoundInitializer(typ.declarator)
                    else:
                        currentLevel.extend(self.parseConstantFromType(typ.declarator))

                    currentLevelChildren = 1

                    while self.peek().id == ",":
                        self.pop()
                        if self.peek().id == "}":
                            break

                        if isinstance(typ.declarator, ArrayDeclaratorType):
                            # Recursive calls to initialize inner arrays.
                            processCompoundInitializer(typ.declarator)
                        else:
                            currentLevel.extend(self.parseConstantFromType(typ.declarator))

                        currentLevelChildren += 1

                    self.expect("}")

                if currentLevelChildren > typ.size:
                    self.raiseError(f"Too many initializers: expected {typ.size}, received {len(currentLevel)}")

                # Append ZeroPaddings until the array is filled. 
                if currentLevelChildren < typ.size:
                    padBytes = (typ.size - currentLevelChildren) * typ.declarator.getByteSize()
                    padding = self.createChild(ZeroPaddingInitializer, typ.declarator, padBytes)
                    currentLevel.append(padding)

                # Add to the return list.
                ret.extend(currentLevel)

            processCompoundInitializer(constantType)
            return ret

        else:
            self.raiseError(f"Cannot parse constant of type {constantType}")

    def __str__(self) -> str:
        return self.print(padding=0)

class DeclaratorAST(AST):
    @abstractmethod
    def parse(self, *args):
        pass

    @abstractmethod
    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        pass

    def print(self, padding: int) -> str:
        return ""

class SimpleDeclarator(DeclaratorAST):
    def parse(self, *args):
        nextTok = self.expect("identifier", "(")

        if nextTok.id == "identifier":
            self.isIdentifier = True
            self.id = nextTok.value
        else:
            self.isIdentifier = False
            self.declarator = self.createChild(TopDeclarator)
            self.expect(")")

    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        if self.isIdentifier:
            return DeclaratorInformation(self.id, baseType, [])
        else:
            return self.declarator.process(baseType)

class DirectDeclarator(DeclaratorAST):
    def parse(self, *args):
        self.simple = self.createChild(SimpleDeclarator)
        self.isFunctionDeclarator = False
        self.isArrayDeclarator = False

        nextTok = self.peek()
        if nextTok.id == "(":
            self.isFunctionDeclarator = True
            self.paramDeclarators: list[FunctionParameterDeclarator] = []

            self.pop()

            if self.peek().id == "void" and self.peek(1).id == ")":
                self.pop()
            elif self.peek().id == ")":
                # C99 says that a function with no arguments needs 'void' as argument.
                pass
            else:
                while True:
                    param = self.createChild(FunctionParameterDeclarator)
                    self.paramDeclarators.append(param)
                    if self.peek().id == ",":
                        self.pop()
                    else:
                        break
            
            self.expect(")")
        elif nextTok.id == "[":
            self.isArrayDeclarator = True
            self.arrayDimensions: list[int] = []

            while self.peek().id == "[":
                self.pop()

                if self.peek().id == "]":
                    self.raiseError("Variable length arrays not implemented")

                arrayDim = self.parseConstantFromType(TypeSpecifier.LONG.toBaseType(), strict=True)[0]
                dim = int(arrayDim.constValue)
                if dim <= 0:
                    self.raiseError("Array dimension must be greater than zero")
                
                self.expect("]")

                self.arrayDimensions.insert(0, dim)

    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        if self.isFunctionDeclarator:
            simpleDecl = self.simple.process(baseType)
            if isinstance(simpleDecl.type, (BaseDeclaratorType, PointerDeclaratorType)):
                funcParams: list[ParameterInformation] = []
                for param in self.paramDeclarators:
                    paramInfo: DeclaratorInformation = param.process(param.declType)
                    if isinstance(paramInfo.type, FunctionDeclaratorType):
                        self.raiseError("Function pointers aren't supported as parameters")
                    funcParams.append(ParameterInformation(paramInfo.type, paramInfo.name))
                
                funcDecl = FunctionDeclaratorType(funcParams, simpleDecl.type)
                return DeclaratorInformation(simpleDecl.name, funcDecl, funcDecl.params)
            else:
                self.raiseError("Not implemented")

        elif self.isArrayDeclarator:
            if not baseType.isComplete():
                self.raiseError("Cannot create array of an incomplete type")
            
            arrayDecl = ArrayDeclaratorType(baseType, self.arrayDimensions[0])
            for dim in self.arrayDimensions[1:]:
                arrayDecl = ArrayDeclaratorType(arrayDecl, dim)
            return self.simple.process(arrayDecl)

        else:
            return self.simple.process(baseType)

class FunctionParameterDeclarator(DeclaratorAST):
    def parse(self, *args):
        _, self.declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
        self.decl = self.createChild(TopDeclarator)

    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        info = self.decl.process(baseType)
        info.type = info.type.decay()
        if info.type == TypeSpecifier.VOID.toBaseType():
            self.raiseError("A parameter must not have void type")
        return info

class TopDeclarator(DeclaratorAST):
    def parse(self, *args):
        if self.peek().id == "*":
            self.pop()
            self.isPointer = True

            self.isConstantPointer = False
            while self.peek().id == "const":
                self.isConstantPointer = True
                self.pop()

            self.decl = self.createChild(TopDeclarator)
        else:
            self.decl = self.createChild(DirectDeclarator)
            self.isPointer = False

    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        if self.isPointer:
            baseType = PointerDeclaratorType(baseType, TypeQualifier(self.isConstantPointer))
        ret = self.decl.process(baseType)
        return ret 

# An abstract declarator is the declarator used inside casts. The main differente with normal 
# Declarators is that they don't have an identifier or name inside.
class AbstractDeclaratorAST(AST):
    @abstractmethod
    def parse(self, *args):
        pass

    @abstractmethod
    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        pass

    def print(self, padding: int) -> str:
        return ""

class TopAbstractDeclarator(AbstractDeclaratorAST):
    def parse(self, *args):
        if self.peek().id == "*":
            self.pop()

            self.isPointer = True
            self.isConstantPointer = False
            while self.peek().id == "const":
                self.isConstantPointer = True
                self.pop()

            try:
                self.decl = self.createChild(TopAbstractDeclarator)
            except:
                self.decl = None
        else:
            self.decl = self.createChild(BaseAbstractDeclarator)
            self.isPointer = False

    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        if self.isPointer:
            baseType = PointerDeclaratorType(baseType, TypeQualifier(self.isConstantPointer))
            if self.decl is None:
                return DeclaratorInformation("", baseType, [])
        
        if self.decl is not None:
            return self.decl.process(baseType)
        self.raiseError("No declaration inside TopAbstractDeclarator")

class BaseAbstractDeclarator(AbstractDeclaratorAST):
    def parse(self, *args):
        self.arrayDimensions: list[int] = []
        self.inner = None

        if self.peek().id == "(":
            self.pop()
            self.inner = self.createChild(TopAbstractDeclarator)
            self.expect(")")

            # There could be [], but they are optional.
            while self.peek().id == "[":
                self.pop()

                arrayDim = self.parseConstantFromType(TypeSpecifier.LONG.toBaseType(), strict=True)[0]
                dim = int(arrayDim.constValue)
                if dim <= 0:
                    self.raiseError("Array dimension must be greater than zero")

                self.expect("]")

                self.arrayDimensions.insert(0, dim)

        else:
            # There must be at least one [].
            while True:
                self.expect("[")

                arrayDim = self.parseConstantFromType(TypeSpecifier.LONG.toBaseType(), strict=True)[0]
                dim = int(arrayDim.constValue)
                if dim <= 0:
                    self.raiseError("Array dimension must be greater than zero")

                self.expect("]")

                self.arrayDimensions.insert(0, dim)

                if self.peek().id != "[":
                    break

    def process(self, baseType: DeclaratorType) -> DeclaratorInformation:
        retType = baseType

        for dim in self.arrayDimensions:
            if not retType.isComplete():
                self.raiseError("Cannot create array of an incomplete type")
            retType = ArrayDeclaratorType(retType, dim)

        if self.inner is not None:
            info = self.inner.process(retType)
            retType = info.type

        return DeclaratorInformation("", retType, [])

class Program(AST):
    def parse(self, *args):
        self.topLevel: list[Declaration] = []
        while len(self.tokens) > 0:
            self.topLevel.append(self.createChild(Declaration))

    def print(self, padding: int) -> str:
        ret = ""
        for func in self.topLevel:
            ret += func.print(padding)
        return ret

# Grouping class. A block can either be a Declaration or a Statement.
class BlockItem(AST):
    @abstractmethod
    def parse(self, *args):
        pass

    @abstractmethod
    def print(self, padding: int) -> str:
        pass


"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
STATEMENTS
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

class Statement(BlockItem):
    @abstractmethod
    def parse(self, *args):
        pass

    @abstractmethod
    def print(self, padding: int) -> str:
        pass

class ReturnStatement(Statement):
    def parse(self, *args):
        # None in the case of void functions.
        self.exp: Exp|None = None

        if len(args) > 0:
            self.exp = self.createChild(IntConstant, args[0])
        else:
            if not self.context.insideAFunction:
                self.raiseError("Return is outside a function")

            # Enclosing function.
            encFuncCtx = self.context.functionMap[self.context.insideFunctionName]
            
            self.expect("return")

            if encFuncCtx.returnType != TypeSpecifier.VOID.toBaseType():
                self.exp = self.createChild(Exp).preconvertExpression()

                # Check if the return value needs a cast. It will depend on the return type of the 
                # enclosing function.
                if encFuncCtx.returnType != self.exp.typeId:
                    self.exp = self.createChild(Cast, encFuncCtx.returnType, self.exp, True).preconvertExpression()

            self.expect(";")

    def print(self, padding: int) -> str:
        pad = " " * padding
        if self.exp is not None:
            ret  = f'{pad}Return(\n'
            ret += self.exp.print(padding + PADDING_INCREMENT)
            ret += f'{pad})\n'
        else:
            ret  = f'{pad}Return\n'
        return ret

class IfStatement(Statement):
    def parse(self, *args):
        self.expect("if")
        self.expect("(")

        self.condition = self.createChild(Exp).preconvertExpression()
        if not self.condition.typeId.isScalar():
            self.raiseError(f"Expected a scalar expression, received {self.condition.typeId}")
        
        self.expect(")")

        self.thenStatement = self.createChild(Statement)

        if self.peek().id == "else":
            self.expect("else")
            self.elseStatement = self.createChild(Statement)
        else:
            self.elseStatement = None

    def print(self, padding: int) -> str:
        pad  = " " * padding
        ret  = f'{pad}If(\n'
        ret += f'{pad}- Condition:\n'
        ret += self.condition.print(padding + PADDING_INCREMENT)
        ret += f'{pad}- Then:\n'
        ret += self.thenStatement.print(padding + PADDING_INCREMENT)

        if self.elseStatement is not None:
            ret += f'{pad}- Else:\n'
            ret += self.elseStatement.print(padding + PADDING_INCREMENT)

        ret += f'{pad})\n'
        return ret

class ExpressionStatement(Statement):
    def parse(self, *args):
        self.exp = self.createChild(Exp).preconvertExpression()
        self.expect(";")

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}Statement: Expression(\n'
        ret += self.exp.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret
    
class NullStatement(Statement):
    def parse(self, *args):
        self.expect(";")

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f"{pad}Statement: Null\n"
    
class CompoundStatement(Statement):
    def parse(self, *args):
        self.block: Block = self.createChild(Block)

    def print(self, padding: int) -> str:
        return self.block.print(padding)
    
class BreakStatement(Statement):
    def parse(self, *args):
        self.expect("break")
        if len(self.context.loopTracking) == 0 and len(self.context.switchTracking) == 0:
            # This break is not within a loop.
            self.raiseError("Unexpected 'break' out of loop or switch")
        
        match self.context.getSwitchOrLoop():
            case "loop":
                self.jumpLabel = self.context.loopTracking[-1]
            case "switch":
                self.jumpLabel = self.context.getCurrentSwitch().name
            case _:
                self.raiseError(f"Unexpected context ({self.context._switchOrLoop})")

        self.expect(";")
    
    def print(self, padding: int) -> str:
        pad = " " * padding
        return f"{pad}Break\n"

class ContinueStatement(Statement):
    def parse(self, *args):
        self.expect("continue")
        if len(self.context.loopTracking) == 0:
            # This continue is not within a loop.
            self.raiseError("Unexpected 'continue' out of loop")
        self.jumpLabel = self.context.loopTracking[-1]

        self.expect(";")
    
    def print(self, padding: int) -> str:
        pad = " " * padding
        return f"{pad}Continue\n"

class WhileStatement(Statement):
    def parse(self, *args):
        self.expect("while")
        self.expect("(")

        self.condition = self.createChild(Exp).preconvertExpression()
        if not self.condition.typeId.isScalar():
            self.raiseError(f"Expected a scalar expression, received {self.condition.typeId}")

        self.expect(")")
        
        with self.context.enterLoop() as self.loopTag:
            self.body = self.createChild(Statement)

    def print(self, padding: int) -> str:
        pad  = " " * padding
        ret  = f'{pad}While(\n'
        ret += f'{pad}- Condition:\n'
        ret += self.condition.print(padding + PADDING_INCREMENT)
        ret += f'{pad}- Body:\n'
        ret += self.body.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret

class DoWhileStatement(Statement):
    def parse(self, *args):
        self.expect("do")
        with self.context.enterLoop() as self.loopTag:
            self.body = self.createChild(Statement)

        self.expect("while")
        self.expect("(")

        self.condition = self.createChild(Exp).preconvertExpression()
        if not self.condition.typeId.isScalar():
            self.raiseError(f"Expected a scalar expression, received {self.condition.typeId}")

        self.expect(")")
        self.expect(";")

    def print(self, padding: int) -> str:
        pad  = " " * padding
        ret  = f'{pad}DoWhile(\n'
        ret += f'{pad}- Body:\n'
        ret += self.body.print(padding + PADDING_INCREMENT)
        ret += f'{pad}- Condition:\n'
        ret += self.condition.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret

class ForStatement(Statement):
    def parse(self, *args):
        self.expect("for")
        self.expect("(")

        # When parsing the header, enter a new context.
        with self.context.newContext():
            tok = self.peek()
            if self.isTokenAnIdentifierType(tok):
                self.init = self.createChild(VariableDeclaration)
                if self.init.storageClass is not None:
                    self.raiseError(f"Cannot have storage class")
            elif tok.id == ";":
                self.init = None
                self.expect(";")
            else:
                self.init = self.createChild(Exp).preconvertExpression()
                self.expect(";")

            tok = self.peek()
            if tok.id == ";":
                self.condition = None
            else:
                self.condition = self.createChild(Exp).preconvertExpression()
                if not self.condition.typeId.isScalar():
                    self.raiseError(f"Expected a scalar expression, received {self.condition.typeId}")
        
            self.expect(";")

            tok = self.peek()
            match tok.id:
                # TODO: this is temporal. 
                case ")":
                    self.post = None
                case _:
                    self.post = self.createChild(Exp).preconvertExpression()
            self.expect(")")

            with self.context.enterLoop() as self.loopTag:
                self.body = self.createChild(Statement)

    def print(self, padding: int) -> str:
        pad  = " " * padding
        ret  = f'{pad}For(\n'
        if self.init is None:
            ret += f'{pad}- Init: None\n'
        else:
            ret += f'{pad}- Init:\n'
            ret += self.init.print(padding + PADDING_INCREMENT)

        if self.condition is None:
            ret += f'{pad}- Condition: None\n'
        else:
            ret += f'{pad}- Condition:\n'
            ret += self.condition.print(padding + PADDING_INCREMENT)

        if self.post is None:
            ret += f'{pad}- Post: None\n'
        else:
            ret += f'{pad}- Post:\n'
            ret += self.post.print(padding + PADDING_INCREMENT)

        ret += f'{pad}- Body:\n'
        ret += self.body.print(padding + PADDING_INCREMENT)

        ret += f'{pad})\n'
        return ret

class CaseStatement(Statement):
    def parse(self, *args):
        self.expect("case")
        if len(self.context.switchTracking) == 0:
            self.raiseError("Unmatched 'case' outside switch")
        
        self.value = self.parseConstantFromType(
            self.context.getCurrentSwitch().controlType,
            strict=True)[0]
        
        if self.value.constValue in [case.value.constValue for case in self.context.getCurrentSwitch().cases]:
            self.raiseError(f"Duplicated 'case' with value {self.value.constValue}")
        
        self.switchLabel = self.context.getCurrentSwitch().name
        # Add this case to the current switch list.
        self.context.getCurrentSwitch().cases.append(self)

        self.expect(":")

        # C17 requires an statement after a label.
        self.statement = self.createChild(Statement)
    
    def print(self, padding: int) -> str:
        pad  = " " * padding
        ret  = f'{pad}Case(\n'
        ret += self.value.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret
    
class DefaultStatement(Statement):
    def parse(self, *args):
        self.expect("default")
        if len(self.context.switchTracking) == 0:
            self.raiseError("Unmatched 'default' outside switch")
        
        if self.context.getCurrentSwitch().defaultCase is not None:
            self.raiseError(f"Multiple 'default' in the same switch")

        self.switchLabel = self.context.getCurrentSwitch().name
        # Add this default to the current switch.
        self.context.getCurrentSwitch().defaultCase = self

        self.expect(":")

        # C17 requires an statement after a label.
        self.statement = self.createChild(Statement)

    def print(self, padding: int) -> str:
        pad  = " " * padding
        return f'{pad}Default\n'

class SwitchStatement(Statement):
    def parse(self, *args):
        self.expect("switch")
        self.expect("(")
        self.condition = self.createChild(Exp).preconvertExpression().checkIntegerPromotion()

        if not isinstance(self.condition.typeId, BaseDeclaratorType) or \
           not self.condition.typeId.baseType.isInteger():
            self.raiseError("Switch control variable must be an integer")

        self.expect(")")

        with self.context.enterSwitch(self.condition.typeId) as self.switchLabel:
            self.body = self.createChild(Statement)
            
            # The context stores information about all the cases and default.
            self.caseList = self.context.getCurrentSwitch().cases
            self.defaultCase = self.context.getCurrentSwitch().defaultCase

    def print(self, padding: int) -> str:
        pad  = " " * padding
        ret  = f'{pad}Switch(\n'
        ret += f'{pad}- Condition:\n'
        ret += self.condition.print(padding + PADDING_INCREMENT)
        ret += f'{pad}- Body:\n'
        ret += self.body.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret

class LabelStatement(Statement):
    def parse(self, *args):
        labelName = self.expect("identifier").value
        if labelName in self.context.labelTracking:
            if self.context.labelTracking[labelName].declared:
                self.raiseError(f"Duplicated label {labelName}")
            else:
                self.labelName = self.context.labelTracking[labelName].mangledName
                self.context.labelTracking[labelName].declared = True
        else:
            self.labelName = self.context.mangleLabelName(labelName)
            self.context.labelTracking[labelName].declared = True

        self.expect(":")

        # C17 requires an statement after a label.
        self.statement = self.createChild(Statement)

    def print(self, padding: int) -> str:
        pad  = " " * padding
        return f'{pad}Label({self.labelName})\n'

class GotoStatement(Statement):
    def parse(self, *args):
        self.expect("goto")
        
        labelNameToken = self.expect("identifier")
        labelName = labelNameToken.value
        if labelName in self.context.labelTracking:
            self.labelName = self.context.labelTracking[labelName].mangledName
        else:
            self.labelName = self.context.mangleLabelName(labelName)
            # But do not set it as declared.
            self.context.labelTracking[labelName].gotoToken = labelNameToken

        self.expect(";")

    def print(self, padding: int) -> str:
        pad  = " " * padding
        return f'{pad}Goto({self.labelName})\n'


"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DECLARATIONS
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

class Declaration(BlockItem):
    @abstractmethod
    def parse(self, *args):
        pass
    
    @abstractmethod
    def print(self, padding: int) -> str:
        pass

class FunctionDeclaration(Declaration):
    def parse(self, *args):
        if len(args) > 0:
            self.storageClass: StorageClass | None = args[0]
            info: DeclaratorInformation = args[1]
        else:
            self.storageClass, declType, _ = self.getStorageClassAndDeclaratorType()
            declarator = self.createChild(TopDeclarator)
            info: DeclaratorInformation = declarator.process(declType)

        if not isinstance(info.type, FunctionDeclaratorType):
            self.raiseError("Expected a function declaration")

        self.typeId: FunctionDeclaratorType = info.type
        self.identifier: str = info.name
        self.returnType: DeclaratorType = info.type.returnDeclarator

        if isinstance(self.returnType, FunctionDeclaratorType):
            self.raiseError("A function cannot return another function")
        if isinstance(self.returnType, ArrayDeclaratorType):
            self.raiseError("A function cannot return an array")

        if self.context.insideAFunction and self.storageClass == StorageClass.STATIC:
            self.raiseError("A block-scope function may only have extern storage class")

        self.isGlobal = self.storageClass != StorageClass.STATIC

        if self.identifier in self.context.functionMap:
            funcInMap = self.context.functionMap[self.identifier]
            if funcInMap.isGlobal and not self.isGlobal:
                self.raiseError("Static function declaration follows non-static")
            if funcInMap.returnType != self.returnType:
                self.raiseError(f"Function should return {funcInMap.returnType}")
            # A function can be first defined as static and, on the next definition, if it has no 
            # storage class, it should be considered as static too (even though it would normally be
            # considered extern by default).
            self.isGlobal = funcInMap.isGlobal

        if self.identifier in self.context.identifierMap:
            idCtx = self.context.identifierMap[self.identifier]
            if idCtx.alreadyDeclared and idCtx.identifierType != IdentifierType.FUNCTION:
                self.raiseError(f"{self.identifier} already declared in this scope")
            
            if self.identifier in self.context.staticVariablesMap:
                self.raiseError(f"{self.identifier} already declared in this scope")
        
        # Functions can be declared (remember, defined is not the same as declared) more than once 
        # in the same scope. 
        self.context.addFunctionIdentifier(self)

        # Parse the function's argument list.
        self.argumentList: list[ParameterInformation] = info.type.params
        # Names of the variables after being mangled.
        self.definedArgumentList : list[ParameterInformation] = []
        
        # Can't have two variables passed to a function with the same name.
        argNames: list[str] = []
        for arg in self.argumentList:
            if arg.name in argNames:
                self.raiseError(f"Redefinition of argument {arg.name}")
            argNames.append(arg.name)

        if self.identifier in self.context.functionMap:
            # Check the arguments of the function.
            ctxArgs = self.context.functionMap[self.identifier].arguments
            if len(ctxArgs) != len(self.argumentList):
                self.raiseError(f"Function {self.identifier} already declared with different arguments")

            # Check the types of the arguments. The names and qualifiers do not need to be the same. 
            for ctxVar, var in zip(ctxArgs, self.argumentList):
                if ctxVar.type.unqualify() != var.type.unqualify():
                    self.raiseError(f"{var.name} should be {ctxVar.type}, not {var.type}")
        else:
            # Add the function information to the current context.
            self.context.functionMap[self.identifier] = FunctionIdentifier(
                name=self.identifier,
                arguments=self.argumentList,
                returnType=self.returnType,
                storageClass=self.storageClass,
                isGlobal=self.isGlobal,
                alreadyDefined=False
            )

        self.body: list[AST]|None = None

        if self.peek().id == "{":
            if self.context.insideAFunction:
                self.raiseError("Can't define a function inside another function")

            if info.type.returnDeclarator != TypeSpecifier.VOID.toBaseType() and \
               not info.type.returnDeclarator.isComplete():
                self.raiseError("Cannnot define a function with an incomplete return type")

            # Check if the function is already defined.
            if self.context.functionMap[self.identifier].alreadyDefined:
                self.raiseError(f"Function {self.identifier} is already defined")
            else:
                self.context.functionMap[self.identifier].alreadyDefined = True
                # Set the right argument list with the function definition's qualifiers. Remember,
                # qualifiers can be dropped on function prototypes, but they "stay" on function 
                # definitions.
                self.context.functionMap[self.identifier].arguments = self.argumentList

            # Check that there are no incomplete types passed as arguments.
            for arg in self.argumentList:
                if not arg.type.isComplete():
                    self.raiseError(f"Type of argument {arg.name} is not complete")

            self.expect("{")

            with self.context.enterFunction(self.identifier):
                # Include the arguments of the function to the new context.
                for funcArg in self.argumentList:
                    mangledName = self.context.mangleIdentifier(funcArg.name)
                    mangledParam = ParameterInformation(funcArg.type, mangledName)
                    self.definedArgumentList.append(mangledParam)
                    self.context.addVariableIdentifier(funcArg.name, mangledName, funcArg.type)

                self.body = []
                while self.peek().id != "}":
                    self.body.append(self._parseBlockItem())
            
            self.expect("}")
        else:
            self.expect(";")

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}Function(\n'
        ret += f"{pad}- Return: {self.typeId.returnDeclarator}\n"
        ret += f"{pad}- Identifier: {self.identifier}\n"

        if self.body is not None:
            argsStr = [f"{arg.type} {arg.name}" for arg in self.definedArgumentList]
            ret += f"{pad}- Parameters: {', '.join(argsStr)}\n"
            ret += f"{pad}- Body:\n"
            for inst in self.body:
                ret += inst.print(padding + PADDING_INCREMENT)
        else:
            argsStr = [f"{arg.type} {arg.name}" for arg in self.argumentList]
            ret += f"{pad}- Parameters: {', '.join(argsStr)}\n"

        ret += f'{pad})\n'
        return ret

class VariableDeclaration(Declaration):
    def parse(self, *args):
        if len(args) > 0:
            self.storageClass: StorageClass | None = args[0]
            info: DeclaratorInformation = args[1]
        else:
            self.storageClass, declType, _ = self.getStorageClassAndDeclaratorType()
            declarator = self.createChild(TopDeclarator)
            info: DeclaratorInformation = declarator.process(declType)

        if isinstance(info.type, FunctionDeclaratorType):
            self.raiseError("Expected a variable declaration")

        if self.storageClass != StorageClass.EXTERN and not info.type.isComplete():
            self.raiseError("Cannot define a variable with an incomplete type")

        self.originalIdentifier = info.name
        self.typeId = info.type
        self.initialization: Initializer|list[Constant] = []
        tentative: bool = False
        self.isGlobal: bool = True

        # Check if there's a variable already declared but with different type.
        if self.originalIdentifier in self.context.identifierMap:
            idCtx = self.context.identifierMap[self.originalIdentifier]
            if idCtx.identifierType != IdentifierType.VARIABLE and idCtx.alreadyDeclared:
               self.raiseError(f"{self.originalIdentifier} was already defined in this scope")
           
            if idCtx.alreadyDeclared and \
               (ctxType := self.context.variablesMap[idCtx.mangledName].idType) != self.typeId:
                self.raiseError(f"{self.originalIdentifier} already defined as {ctxType}, not {self.typeId}")

        if self.context.insideAFunction:
            # This declaration is a Block Scope Variable Declaration.
            if self.originalIdentifier in self.context.identifierMap:
                idCtx = self.context.identifierMap[self.originalIdentifier]
                
                if self.originalIdentifier in self.context.staticVariablesMap:
                    varCtx = self.context.staticVariablesMap[self.originalIdentifier]
                    # You can declare a variable with external linkage multiple times in the same block.
                    if idCtx.alreadyDeclared and not (varCtx.isGlobal and self.storageClass == StorageClass.EXTERN):
                        self.raiseError(f"{self.originalIdentifier} has already been declared in this scope")
                
                elif idCtx.alreadyDeclared:
                    self.raiseError(f"{self.originalIdentifier} has already been declared in this scope")

            if self.storageClass == StorageClass.EXTERN:
                if self.originalIdentifier in self.context.functionMap:
                    self.raiseError(f"{self.originalIdentifier} already declared as a function")

                if self.peek().id == "=":
                    self.raiseError(f"Initializer on local extern variable declaration {self.originalIdentifier}")
                
                if self.originalIdentifier in self.context.staticVariablesMap:
                    varCtx = self.context.staticVariablesMap[self.originalIdentifier]
                    if varCtx.idType != self.typeId:
                        self.raiseError(f"{self.originalIdentifier} already defined as {varCtx.idType}, not {self.typeId}")

                    self.identifier = varCtx.mangledName
                else:
                    # By default, take it as an external linkage variable.
                    self.identifier = self.originalIdentifier
                
                # Add the variable to the identifiers context.
                self.context.addVariableIdentifier(self.originalIdentifier, self.identifier, self.typeId)

                # This is a global static variable (accessible from other compilation units). Add it
                # to the context.
                if self.originalIdentifier not in self.context.staticVariablesMap:
                    self.context.staticVariablesMap[self.originalIdentifier] = StaticVariableContext(
                        mangledName=self.identifier,
                        idType=self.typeId,
                        storageClass=self.storageClass,
                        isGlobal=True,
                        tentative=False,
                        initialization=[]
                    )

            elif self.storageClass == StorageClass.STATIC:
                if self.peek().id == "=":
                    self.pop()
                    self.initialization = self.parseConstantFromType(self.typeId)
                else:
                    self.initialization = [self.createChild(ZeroPaddingInitializer, self.typeId, self.typeId.getByteSize())]

                # In different blocks you can define a new static variable.
                self.identifier = self.context.mangleIdentifier(self.originalIdentifier)
                self.isGlobal = False
                # Add the variable to the identifiers context.
                self.context.addVariableIdentifier(self.originalIdentifier, self.identifier, self.typeId)

                # This is a local static variable (accessible only from this compilation unit). Add 
                # it to the context.
                self.context.staticVariablesMap[self.identifier] = StaticVariableContext(
                    mangledName=self.identifier,
                    idType=self.typeId,
                    storageClass=self.storageClass,
                    isGlobal=False,
                    tentative=False,
                    initialization=self.initialization
                )
            else:
                # Automatic variables (normal variables in a block scope).
                self.identifier = self.context.mangleIdentifier(self.originalIdentifier)
                self.isGlobal = False
                # Add the variable to the identifiers context.
                self.context.addVariableIdentifier(self.originalIdentifier, self.identifier, self.typeId)

                if self.peek().id == "=":
                    self.pop()
                    self.initialization = self.createChild(Initializer, self.typeId)
                else:
                    # Initialization is already []
                    pass

        else:
            # This is a File Scope Variable Declaration.
            if self.peek().id == "=":
                self.pop()
                self.initialization = self.parseConstantFromType(self.typeId)
            else:
                if self.storageClass == StorageClass.EXTERN:
                    # Initialization is already []
                    pass
                else:
                    # The value can be declared later on.
                    tentative = True

            self.isGlobal = self.storageClass != StorageClass.STATIC

            if self.originalIdentifier in self.context.staticVariablesMap:
                varCtx: StaticVariableContext = self.context.staticVariablesMap[self.originalIdentifier]

                self.identifier = varCtx.mangledName
                if varCtx.idType != self.typeId:
                    self.raiseError(f"{self.identifier} already defined as {varCtx.idType}")

                if self.storageClass == StorageClass.EXTERN:
                    self.isGlobal = varCtx.isGlobal
                elif varCtx.isGlobal != self.isGlobal:
                    self.raiseError("Conflicting variable linkage")

                if varCtx.initialization and not varCtx.tentative:
                    # If the context declaration is a constant...
                    if self.initialization and not tentative:
                        # ... and the initialization is a constant, raise error.
                        self.raiseError("Conflicting file scope variable definitions")
                    
                    self.initialization = varCtx.initialization
                    tentative = False
                elif varCtx.tentative and not(self.initialization and not tentative):
                    # If the context declaration is set as tentative and the current initialization
                    # is not a constant, set as tentative.
                    tentative = True
            else:
                if self.isGlobal:
                    self.identifier = self.originalIdentifier
                else:
                    self.identifier = self.context.mangleIdentifier(self.originalIdentifier)

            self.context.staticVariablesMap[self.originalIdentifier] = StaticVariableContext(
                mangledName=self.identifier,
                idType=self.typeId,
                storageClass=self.storageClass,
                isGlobal=self.isGlobal,
                tentative=tentative,
                initialization=self.initialization # type: ignore
            )

            # Add the variable to the identifiers context.
            self.context.addVariableIdentifier(self.originalIdentifier, self.identifier, self.typeId)

        self.expect(";")

    def print(self, padding: int) -> str:
        pad = " " * padding
        if self.initialization is None:
            return f"{pad}VariableDecl: {self.typeId} {self.identifier}\n"
        else:
            ret  = f"{pad}VariableDecl: {self.typeId} {self.identifier} = (\n"
            if isinstance(self.initialization, AST):
                ret += self.initialization.print(padding + PADDING_INCREMENT)
            else:
                for init in self.initialization:
                    ret += f"{pad}  {init}"
            ret += f"{pad})\n"
            return ret

class StructMemberDeclaration(AST):
    def parse(self, *args):
        _, declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
        declarator = self.createChild(TopDeclarator)
        info: DeclaratorInformation = declarator.process(declType)

        self.typeId = info.type
        self.name = info.name
        # The offset gets written in StructDeclaration.
        self.offset: int = -1

        if not self.typeId.isComplete():
            self.raiseError("Cannot use incomplete type inside a struct declaration")

        self.expect(";")

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}{self.typeId} {self.name}'

class StructDeclaration(Declaration):
    ANONYMOUS_STRUCT_COUNT: int = 0

    def parse(self):
        self.members: list[StructMemberDeclaration] = []
        self.byteSize: int = -1
        self.alignment: int = -1

        self.expect("struct")

        self.anonymous = self.peek().id != "identifier"
        if self.anonymous:
            self.originalIdentifier = f"struct.{StructDeclaration.ANONYMOUS_STRUCT_COUNT}"
            StructDeclaration.ANONYMOUS_STRUCT_COUNT += 1
        else:
            self.originalIdentifier = self.pop().value

        # If it's a declaration, it can shadow other tagged elements.  
        canShadowOthers = self.peek().id in (";", "{")

        # Check that there's no unions, enums or typedefs with this name already defined.
        conflictCtx, conflictingType = self.context.getConflictsForTaggableType(TaggableType.STRUCT, 
                                                                   self.originalIdentifier)
        if conflictCtx is not None and conflictingType is not None:
            # If there's a conflict with a type which is already declared in the scope, always raise
            # an error.
            if conflictCtx.alreadyDeclared:
                self.raiseError(f"{conflictingType} already declared in this scope")

            if not canShadowOthers:
                # Check if the current type exists.
                ctx = self.context.getStructFromOriginalName(self.originalIdentifier)
                if ctx is None:
                    # If the current type does not exist and you're not shadowing the previous type,
                    # then there's a conflict. 
                    self.raiseError(f"This conflicts with previous type {conflictingType}")
                # If it exists, then you're just creating a new variable with the current type.

        ctx = self.context.getStructFromOriginalName(self.originalIdentifier)
        if ctx is None or (canShadowOthers and not ctx.alreadyDeclared):
            # If the struct is new, or it is shadowing another struct with the same name...
            # Create a new mangled identifier.
            self.identifier = self.context.mangleIdentifier(self.originalIdentifier)
            # Create the type.
            self.typeId = TypeSpecifier.STRUCT(self.originalIdentifier, self.identifier, 
                                               self.byteSize, self.alignment, 
                                               self.membersToParamInfo())
            # Add it to the context map only if it has not been declared in the current scope.
            self.context.addStruct(self)
        else:
            # The struct is referring to a previously declared struct, get its type and identifier.
            self.identifier = ctx.mangledName
            self.typeId = self.context.structMap[ctx.mangledName]

        # Parse the members.
        if self.anonymous and self.peek().id != "{":
            self.raiseError("Anonymous struct needs to have a declaration list")

        if self.peek().id == "{":
            self.pop()

            # The struct is going to be defined.
            self.byteSize = 0

            # Must always have at least one member.
            while True:
                decl = self.createChild(StructMemberDeclaration)
                declAlignment = decl.typeId.getAlignment()
                decl.offset = declAlignment * math.ceil(self.byteSize / declAlignment)

                # The name of the declaration must not be repeated.
                for mem in self.members:
                    if mem.name == decl.name:
                        self.raiseError(f"Duplicated member with name {decl.name}")
                self.members.append(decl)

                # The alignment of the structure must be the greatest alignment of it members.
                if declAlignment > self.alignment:
                    self.alignment = declAlignment

                # Add the size of the member plus the offset added between the current member and the
                # previous one.
                self.byteSize += decl.typeId.getByteSize() + (decl.offset - self.byteSize)

                if self.peek().id == "}":
                    break
            
            # Round up the byteSize depending on the struct's alignment.
            self.byteSize = self.alignment * math.ceil(self.byteSize / self.alignment)
            self.expect("}")

        ctx = self.context.getStructFromOriginalName(self.originalIdentifier)
        if ctx is None:
            self.raiseError("This should not be None")
        
        if ctx.alreadyDeclared and len(self.members) > 0:
            ctxType = self.context.structMap[ctx.mangledName]
            if len(ctxType.getMembers()) > 0:
                self.raiseError(f"{ctxType} already defined in this scope")
            else:
                # Set the new members of the struct.
                self.context.completeStruct(self)

    def membersToParamInfo(self) -> list[ParameterInformation]:
        return [
            ParameterInformation(member.typeId, member.name, member.offset)
            for member in self.members
        ]

    def print(self, padding: int) -> str:
        pad = " " * padding
        if len(self.members) > 0:
            ret  = f'{pad}Struct({self.typeId} = {{\n'
            for member in self.members:
                ret += f'{pad}  - {member.typeId} {member.name}\n'
            ret += f'{pad}}}\n'
        else:
            ret =f'{pad}Struct({self.typeId})\n'
        return ret

class UnionMemberDeclaration(AST):
    def parse(self, *args):
        _, declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
        declarator = self.createChild(TopDeclarator)
        info: DeclaratorInformation = declarator.process(declType)

        self.typeId = info.type
        self.name = info.name

        if not self.typeId.isComplete():
            self.raiseError("Cannot use incomplete type inside an enum declaration")

        self.expect(";")

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}{self.typeId} {self.name}'

class UnionDeclaration(Declaration):
    ANONYMOUS_UNION_COUNT: int = 0

    def parse(self):
        self.members: list[UnionMemberDeclaration] = []
        self.byteSize: int = -1
        self.alignment: int = -1

        self.expect("union")

        self.anonymous = self.peek().id != "identifier"
        if self.anonymous:
            self.originalIdentifier = f"union.{UnionDeclaration.ANONYMOUS_UNION_COUNT}"
            UnionDeclaration.ANONYMOUS_UNION_COUNT += 1
        else:
            self.originalIdentifier = self.pop().value
        
        # If it's a declaration, it can shadow other tagged elements.  
        canShadowOthers = self.peek().id in (";", "{")

        # Check that there's no struct, enums or typedefs with this name already defined.
        conflictCtx, conflictingType = self.context.getConflictsForTaggableType(TaggableType.UNION, 
                                                                                self.originalIdentifier)
        if conflictCtx is not None and conflictingType is not None:
            # If there's a conflict with a type which is already declared in the scope, always raise
            # an error.
            if conflictCtx.alreadyDeclared:
                self.raiseError(f"{conflictingType} already declared in this scope")

            if not canShadowOthers:
                # Check if the current type exists.
                ctx = self.context.getUnionFromOriginalName(self.originalIdentifier)
                if ctx is None:
                    # If the current type does not exist and you're not shadowing the previous type,
                    # then there's a conflict. 
                    self.raiseError(f"This conflicts with previous type {conflictingType}")
                # If it exists, then you're just creating a new variable with the current type.

        ctx = self.context.getUnionFromOriginalName(self.originalIdentifier)
        if ctx is None or (canShadowOthers and not ctx.alreadyDeclared):
            # If the union is new, or it is being shadowed by another union with the same name...
            # Create a new mangled identifier.
            self.identifier = self.context.mangleIdentifier(self.originalIdentifier)
            # Create the type.
            self.typeId = TypeSpecifier.UNION(self.originalIdentifier, self.identifier, 
                                               self.byteSize, self.alignment, 
                                               self.membersToParamInfo())
            # Add it to the context map only if it has not been declared in the current scope.
            self.context.addUnion(self)
        else:
            # The union is referring to a previously declared union, get its type and identifier.
            self.identifier = ctx.mangledName
            self.typeId = self.context.unionMap[ctx.mangledName]

        # Parse the members.
        if self.anonymous and self.peek().id != "{":
            self.raiseError("Anonymous union needs to have a declaration list")

        if self.peek().id == "{":
            self.pop()

            # The union is going to be defined. Its size is the size of the biggest element in the 
            # union.
            self.byteSize = 0

            # Must always have at least one member.
            while True:
                decl = self.createChild(UnionMemberDeclaration)
                declAlignment = decl.typeId.getAlignment()
                declSize = decl.typeId.getByteSize()

                # The name of the declaration must not be repeated.
                for mem in self.members:
                    if mem.name == decl.name:
                        self.raiseError(f"Duplicated member with name {decl.name}")
                self.members.append(decl)

                # The alignment of the union must be the greatest alignment of it members.
                if declAlignment > self.alignment:
                    self.alignment = declAlignment

                # The size of the union is the size of the biggest member.
                if declSize > self.byteSize:
                    self.byteSize = declSize

                if self.peek().id == "}":
                    break
            
            # Round up the byteSize depending on the union's alignment.
            self.byteSize = self.alignment * math.ceil(self.byteSize / self.alignment)
            self.expect("}")

        ctx = self.context.getUnionFromOriginalName(self.originalIdentifier)
        if ctx is None:
            self.raiseError("This should not be None")
        
        if ctx.alreadyDeclared and len(self.members) > 0:
            ctxType = self.context.unionMap[ctx.mangledName]
            if len(ctxType.getMembers()) > 0:
                self.raiseError(f"{ctxType} already defined in this scope")
            else:
                # Set the new members of the union.
                self.context.completeUnion(self)

    def membersToParamInfo(self) -> list[ParameterInformation]:
        # Offsets in union are all 0.
        return [
            ParameterInformation(member.typeId, member.name, 0)
            for member in self.members
        ]

    def print(self, padding: int) -> str:
        pad = " " * padding
        if len(self.members) > 0:
            ret  = f'{pad}Union({self.typeId} = {{\n'
            for member in self.members:
                ret += f'{pad}  - {member.typeId} {member.name}\n'
            ret += f'{pad}}}\n'
        else:
            ret =f'{pad}Union({self.typeId})\n'
        return ret

class EnumMemberDeclaration(AST):
    def parse(self, *args):
        # An enum member is always a const int.
        self.typeId = TypeSpecifier.INT.toBaseType(TypeQualifier(const=True))
        self.name = self.expect("identifier").value
        self.mangledName = self.context.mangleIdentifier(self.name)

        # If the value is not set on the member declaration, it will be calculated in the 
        # EnumDeclaration.
        self.value: Constant
        self.hasSetValue: bool = False
        if self.peek().id == "=":
            self.pop()
            self.value = self.parseConstantFromType(TypeSpecifier.INT.toBaseType(), True)[0]
            self.hasSetValue = True

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}{self.typeId} {self.name}'

class EnumDeclaration(Declaration):
    ANONYMOUS_ENUM_COUNT: int = 0

    def parse(self):
        # An enum is always defined.
        self.isDefined = True

        self.expect("enum")

        if self.peek().id != "identifier":
            # Anonymous enum.
            self.originalIdentifier = f"enum.{EnumDeclaration.ANONYMOUS_ENUM_COUNT}"
            EnumDeclaration.ANONYMOUS_ENUM_COUNT += 1
        else:
            self.originalIdentifier = self.pop().value

        self.members: list[EnumMemberDeclaration] = []

        # Create a new mangled name.
        self.identifier = self.context.mangleIdentifier(self.originalIdentifier)

        # Check that there's no struct, unions or typedefs with this name already defined.
        conflictCtx, conflictingType = self.context.getConflictsForTaggableType(TaggableType.ENUM, 
                                                                                self.originalIdentifier)
        if conflictCtx is not None and conflictingType is not None and conflictCtx.alreadyDeclared:
            # We can shadow the conflicting type if it is not yet declared in the same scope.
            self.raiseError(f"{conflictingType} already declared in this scope")

        # Check that there's no enum with this name already defined. Contrary to structs and unions, 
        # enums cannot be declared first and defined later.
        prevEnum = self.context.getEnumFromOriginalName(self.originalIdentifier)
        if prevEnum is not None and prevEnum.alreadyDeclared:
            self.raiseError(f"{self.context.enumMap[prevEnum.mangledName]} already declared in this scope")

        # If the enum is new, or it is being shadowed by another enum with the same name...
        # Create a new mangled identifier.
        self.identifier = self.context.mangleIdentifier(self.originalIdentifier)
        # Create the type.
        self.typeId = TypeSpecifier.ENUM(self.originalIdentifier, self.identifier)
        # Add it to the context map only if it has not been declared in the current scope.
        self.context.addEnum(self)

        # Parse the members. An enum cannot be forward declared in C99. This is different to structs
        # and unions.
        self.expect("{")

        # The enum is going to be defined. It must always have at least one member.
        # The initial value of the enum is always zero, unless set in the member.
        nextEnumValue: int = 0
        while True:
            decl = self.createChild(EnumMemberDeclaration)

            # The name of the declaration must not be repeated.
            for mem in self.members:
                if mem.name == decl.name:
                    self.raiseError(f"Duplicated member with name {decl.name}")

            # Set the value of the enum if it does not have one.
            if decl.hasSetValue:
                nextEnumValue = int(decl.value.constValue)
            else:
                decl.value = IntConstant([], None, None, nextEnumValue)
            # Increment for the next enum value.
            nextEnumValue += 1

            self.members.append(decl)

            if decl.name in self.context.identifierMap and \
                self.context.identifierMap[decl.name].alreadyDeclared:
                self.raiseError(f"{decl.name} already declared in this scope")

            # Add the enum member as a new "constant variable" to the context.
            self.context.addVariableIdentifier(decl.name, decl.mangledName, decl.typeId, decl.value)

            # End of enum.
            if self.peek().id == "}":
                break

            # End of enum with trailing comma.
            if self.peek().id == "," and self.peek(1).id == "}":
                self.pop()
                break

            # If there are more members, a comma is expected.
            self.expect(",")

        self.expect("}")

    def print(self, padding: int) -> str:
        pad = " " * padding
        if len(self.members) > 0:
            ret  = f'{pad}Enum({self.typeId} = {{\n'
            for member in self.members:
                ret += f'{pad}  - {member.name} = {member.value}\n'
            ret += f'{pad}}}\n'
        else:
            ret =f'{pad}Enum({self.typeId})\n'
        return ret
    
class TypedefDeclaration(Declaration):
    def parse(self, *args):
        self.expect("typedef")
        _, declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
        declarator = self.createChild(TopDeclarator)
        info: DeclaratorInformation = declarator.process(declType)

        self.originalIdentifier = info.name
        self.typeId = info.type
        # Set the alias of the type.
        self.typeId.setAlias(self.originalIdentifier)

        # Check that there's no conflict with other identifiers in the same scope.
        if self.originalIdentifier in self.context.identifierMap and \
           self.context.identifierMap[self.originalIdentifier].alreadyDeclared:
            self.raiseError(f"{self.originalIdentifier} already declared in this scope")

        self.identifier = self.context.mangleIdentifier(self.originalIdentifier)
        self.context.addTypedefIdentifier(self.originalIdentifier, self.identifier, self.typeId)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f"{pad}Typedef {self.originalIdentifier} ({self.typeId})"

# A block is a list of BlockItems enclosed by {}.
class Block(AST):
    def parse(self, *args):
        self.expect("{")

        with self.context.newContext():
            self.blockItems: list[AST] = []
            while self.peek().id != "}":
                self.blockItems.append(self._parseBlockItem())
        
        self.expect("}")

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret = f"{pad}{{\n"
        for block in self.blockItems:
            ret += block.print(padding + PADDING_INCREMENT)
        ret += f"{pad}}}\n"
        return ret

class StaticEvalType(enum.Enum):
    INTEGER = enum.auto()
    FLOAT = enum.auto()
    POINTER = enum.auto()
    VARIABLE = enum.auto()

@dataclass
class StaticEvalValue:
    valType: StaticEvalType
    value: str
    # Byte offset from the base of the pointer.
    pointerOffset: int = 0

    def getIntegerValue(self, referenceAST: AST) -> int:
        if self.valType in (StaticEvalType.POINTER, StaticEvalType.VARIABLE):
            referenceAST.raiseError(f"Cannot get integer value during constant folding")
        try:
            return int(self.value)
        except:
            return int(float(self.value))

    def getFloatValue(self, referenceAST: AST) -> float:
        if self.valType in (StaticEvalType.POINTER, StaticEvalType.VARIABLE):
            referenceAST.raiseError(f"Cannot get float value during constant folding")
        return float(self.value)
    
class Exp(AST):
    def __init__(self, tokens: list[Token], 
                 context: Context | None = None, parentAST: AST | None = None, *args) -> None:
        self.typeId: DeclaratorType = None # type: ignore
        super().__init__(tokens, context, parentAST, *args)
        if self.typeId is None:
            self.raiseError("Expression type is None")

    @abstractmethod
    def parse(self, *args):
        pass

    @abstractmethod
    def staticEval(self) -> StaticEvalValue:
        pass

    @abstractmethod
    def print(self, padding: int) -> str:
        pass

    def isLvalue(self) -> bool:
        return isinstance(self, (Variable, Dereference, Subscript, String, Arrow)) or \
            (isinstance(self, Dot) and self.getFirstDotOperand().isLvalue())
    
    def isLvalueAssignable(self) -> bool:
        # To be assignable, it must be a lvalue and not a constant.
        return self.isLvalue() and \
            not isinstance(self.typeId, (ArrayDeclaratorType, FunctionDeclaratorType)) and \
            not self.typeId.getTypeQualifiers().const
    
    def isNullPointer(self) -> bool:
        if isinstance(self, (CharConstant, UCharConstant, 
                             ShortConstant, UShortConstant, 
                             IntConstant, UIntConstant, 
                             LongConstant, ULongConstant)):
            return int(self.constValue) == 0
        return False
    
    def preconvertExpression(self) -> Exp:
        # If self is not an array, return it normally.
        ret = self

        # If an array is inputted in an expression, add an AddressOf operation beforehand. 
        if isinstance(self.typeId, ArrayDeclaratorType):
            # This will return pointer to array...
            ret = self.createChild(AddressOf, self)
            # In our case, we want to "decay" the array. We're not really taking the address of the
            # array (which would result in a pointer to array), but in the TAC, this instruction is
            # used to get the base address of the array.
            ret.typeId = self.typeId.decay()
        
        # This expression is going to be evaluated, it cannot be an incomplete type (except void).
        if ret.typeId != TypeSpecifier.VOID.toBaseType() and not ret.typeId.isComplete():
            self.raiseError(f"Incomplete type {ret.typeId} is not allowed")

        # When converting from lValue to rValue the qualifiers (const, volatile) of a variable get 
        # stripped. This does not happen for pointers!
        if isinstance(ret.typeId, BaseDeclaratorType):
            newDecl = ret.typeId.copy()
            newDecl.qualifiers = TypeQualifier()
            ret.typeId = newDecl

        return ret
    
    def checkIntegerPromotion(self) -> Exp:
        # Check if the expression needs integer promotion.
        if isinstance(self.typeId, BaseDeclaratorType) and self.typeId.baseType.needsIntegerPromotion():
            return self.createChild(Cast, TypeSpecifier.INT.toBaseType(), self, True)
        
        return self

    def getCommonType(self, exp1: Exp, exp2: Exp) -> DeclaratorType|None:
        if isinstance(exp1.typeId, (PointerDeclaratorType, ArrayDeclaratorType)) or \
           isinstance(exp2.typeId, (PointerDeclaratorType, ArrayDeclaratorType)):
            if exp1.typeId == exp2.typeId:
                return exp1.typeId
            elif exp1.isNullPointer():
                return exp2.typeId
            elif exp2.isNullPointer():
                return exp1.typeId
            elif isinstance(exp1.typeId, (PointerDeclaratorType, ArrayDeclaratorType)) and \
                exp1.typeId.declarator == TypeSpecifier.VOID.toBaseType() and \
                isinstance(exp2.typeId, PointerDeclaratorType):
                # The operations on pointers always return non const pointers.
                return PointerDeclaratorType(TypeSpecifier.VOID.toBaseType())
            elif isinstance(exp2.typeId, (PointerDeclaratorType, ArrayDeclaratorType)) and \
                exp2.typeId.declarator == TypeSpecifier.VOID.toBaseType() and \
                isinstance(exp1.typeId, PointerDeclaratorType):
                return PointerDeclaratorType(TypeSpecifier.VOID.toBaseType())
            
            # Arrays can be converted to pointers.
            if isinstance(exp1.typeId, ArrayDeclaratorType) and isinstance(exp2.typeId, PointerDeclaratorType) and \
               exp1.typeId.decay() == exp2.typeId:
                return exp2.typeId

            if isinstance(exp2.typeId, ArrayDeclaratorType) and isinstance(exp1.typeId, PointerDeclaratorType) and \
               exp2.typeId.decay() == exp1.typeId:
                return exp1.typeId

            return None
        elif isinstance(exp1.typeId, BaseDeclaratorType) and isinstance(exp2.typeId, BaseDeclaratorType):
            type1: TypeSpecifier = exp1.typeId.baseType
            type2: TypeSpecifier = exp2.typeId.baseType
            retType: TypeSpecifier

            # Integer promotions.
            if type1.needsIntegerPromotion():
                type1 = TypeSpecifier.INT
            if type2.needsIntegerPromotion():
                type2 = TypeSpecifier.INT

            if type1 == type2:
                retType = type1
            elif type1.name == "STRUCT" or type2.name == "STRUCT":
                # No common type between different structs.
                return None
            elif type1.name == "UNION" or type2.name == "UNION":
                # No common type between different unions.
                return None
            
            elif type1 == TypeSpecifier.DOUBLE or type2 == TypeSpecifier.DOUBLE:
                retType = TypeSpecifier.DOUBLE
            
            elif type1 == TypeSpecifier.FLOAT or type2 == TypeSpecifier.FLOAT:
                retType = TypeSpecifier.FLOAT

            elif type1.getByteSize() == type2.getByteSize():
                # If both types are the same, choose the unsigned type.
                if type1.isSignedInt():
                    retType = type2
                else:
                    retType = type1
                
            elif type1.getByteSize() > type2.getByteSize():
                retType = type1
            else:
                retType = type2

            return retType.toBaseType()
        else:
            return None

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CONSTANTS: Different kind of C constants of basic types.
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

class Constant(Exp):
    def __init__(self, tokens: list[Token], context: Context | None = None, 
                 parentAST: AST | None = None, *args) -> None:
        self.constValue: str
        super().__init__(tokens, context, parentAST, *args)

    @abstractmethod
    def parse(self, *args):
        pass
    
    def staticEval(self) -> StaticEvalValue:
        if isinstance(self.typeId, PointerDeclaratorType):
            return StaticEvalValue(StaticEvalType.POINTER, self.constValue)
        if self.typeId.isDecimal():
            return StaticEvalValue(StaticEvalType.FLOAT, self.constValue)
        return StaticEvalValue(StaticEvalType.INTEGER, self.constValue)

    @abstractmethod
    def print(self, padding: int) -> str:
        pass

    def _parseIntValue(self):
        if self.peek().id in ("double_constant", "float_constant"):
            intVal = math.floor(float(self.pop().value))
        else:
            intValToken = self.expect(
                "constant", "long_constant", 
                "unsigned_constant", "unsigned_long_constant", "character"
            )
            try:
                intVal = intValToken.parseIntegerToken()
            except Exception as e:
                self.raiseError(str(e))
        return intVal

class CharConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.CHAR.toBaseType()
        
        if len(args) > 0:
            intVal = int(args[0])
        else:
            intVal = self._parseIntValue()

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}Char = {self.constValue}\n'

class UCharConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.UCHAR.toBaseType()
        
        if len(args) > 0:
            intVal = int(args[0])
        else:
            if self.peek().id in ("double_constant", "float_constant"):
                intVal = math.floor(float(self.pop().value))
            else:
                intVal = self._parseIntValue()

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}UChar = {self.constValue}\n'

class ShortConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.SHORT.toBaseType()
        
        if len(args) > 0:
            intVal = int(args[0])
        else:
            if self.peek().id in ("double_constant", "float_constant"):
                intVal = math.floor(float(self.pop().value))
            else:
                intVal = self._parseIntValue()

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}Short = {self.constValue}\n'

class UShortConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.USHORT.toBaseType()
        
        if len(args) > 0:
            intVal = int(args[0])
        else:
            if self.peek().id in ("double_constant", "float_constant"):
                intVal = math.floor(float(self.pop().value))
            else:
                intVal = self._parseIntValue()

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}UShort = {self.constValue}\n'

class IntConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.INT.toBaseType()
        
        if len(args) > 0:
            intVal = int(args[0])
        else:
            intVal = self._parseIntValue()

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}Int = {self.constValue}\n'

class UIntConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.UINT.toBaseType()
        
        if len(args) > 0:
            intVal = int(args[0])
        else:
            intVal = self._parseIntValue()

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}UInt = {self.constValue}\n'

class LongConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.LONG.toBaseType()

        if len(args) > 0:
            intVal = int(args[0])
        else:
            if self.peek().id in ("double_constant", "float_constant"):
                intVal = math.floor(float(self.pop().value))
            else:
                intVal = self._parseIntValue()

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}Long = {self.constValue}\n'

class ULongConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.ULONG.toBaseType()

        if len(args) > 0:
            intVal = int(args[0])
        else:
            if self.peek().id in ("double_constant", "float_constant"):
                intVal = math.floor(float(self.pop().value))
            else:
                intValToken = self.expect(
                    "constant", "long_constant", 
                    "unsigned_constant", "unsigned_long_constant", "character"
                )
                try:
                    intVal = intValToken.parseIntegerToken()
                except Exception as e:
                    self.raiseError(str(e))

        intVal, warn = StaticEvaluation.parseValue(self.typeId.baseType, intVal)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(intVal)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}ULong = {self.constValue}\n'

class DoubleConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.DOUBLE.toBaseType()

        if len(args) > 0:
            doubleValue = float(args[0])
        else:
            if self.peek().id in ("double_constant", "float_constant"):
                doubleValue = self.pop().parseDecimalToken()
            else:
                intValToken = self.expect(
                    "constant", "long_constant", 
                    "unsigned_constant", "unsigned_long_constant", "character"
                )
                try:
                    intVal = intValToken.parseIntegerToken()
                except Exception as e:
                    self.raiseError(str(e))    
                doubleValue = float(intVal)

        doubleValue, warn = StaticEvaluation.parseValue(self.typeId.baseType, doubleValue)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(doubleValue)

        self.hex = f"{struct.unpack('<Q', struct.pack('<d', doubleValue))[0]:#0{16}x}"

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}Double = {self.constValue} (0x{self.hex})\n'

class FloatConstant(Constant):
    def parse(self, *args):
        self.typeId = TypeSpecifier.FLOAT.toBaseType()

        if len(args) > 0:
            doubleValue = float(args[0])
        else:
            if self.peek().id in ("double_constant", "float_constant"):
                doubleValue = self.pop().parseDecimalToken()
            else:
                intValToken = self.expect(
                    "constant", "long_constant", 
                    "unsigned_constant", "unsigned_long_constant", "character"
                )
                try:
                    intVal = intValToken.parseIntegerToken()
                except Exception as e:
                    self.raiseError(str(e))
                doubleValue = float(intVal)

        floatValue, warn = StaticEvaluation.parseValue(self.typeId.baseType, doubleValue)
        if warn:
            warn.rise(self.raiseWarning, self.raiseError)
        self.constValue = str(floatValue)

        self.hex = f"{struct.unpack('<I', struct.pack('<f', floatValue))[0]:#0{8}x}"

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}Float = {self.constValue} ({self.hex})\n'

# This is a special type of constant.
class String(Exp):
    def parse(self, *args):
        self.value: str = self.expect("string").value
        
        # A string may be composed of multiple string tokens.
        while (tok := self.peek()).id == "string":
            self.pop()
            self.value += tok.value

        # Add +1 for the null terminator.
        self.typeId = ArrayDeclaratorType(TypeSpecifier.CHAR.toBaseType(), len(self.value) + 1)

    def staticEval(self) -> StaticEvalValue:
        # TODO: change to pointer name
        return StaticEvalValue(StaticEvalType.POINTER, self.value)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}String = "{self.value}"\n'
    
    def toConstantsList(self) -> list[Constant]:
        chars: list[Constant] = [CharConstant([], None, None, str(ord(ch))) for ch in self.value]
        # Add the null terminator.
        chars.append(CharConstant([], None, None, "0"))
        return chars

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# v These constants are used to initialize static variables v
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Used to pad an array with zeros.
class ZeroPaddingInitializer(Constant):
    def parse(self, typeId: DeclaratorType, byteCount: int):
        self.typeId = typeId
        self.byteCount = byteCount

    def staticEval(self) -> StaticEvalValue:
        raise ValueError()

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}ZeroPaddingInit({self.byteCount})\n'
    
# Used to initialize a pointer with the address of another static object.
class PointerInitializer(Constant):
    def parse(self, typeId: DeclaratorType, pointerName: str, byteOffset: int):
        self.typeId = typeId
        self.constValue = pointerName
        self.offset = byteOffset

    def staticEval(self) -> StaticEvalValue:
        raise ValueError()

    def print(self, padding: int) -> str:
        pad = " " * padding
        if self.offset == 0:
            return f'{pad}PointerInit({self.constValue})\n'
        else:
            return f'{pad}PointerInit({self.constValue} + {self.offset})\n'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# END of constants are used to initialize static variables.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EXPRESSIONS
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
class Variable(Exp):
    def parse(self, variableName: str):
        # This variable is inside the identifierMap, verified at _parsePrimaryExpression.
        ctxVar = self.context.identifierMap[variableName]
        
        # Get the type of variable from the identifier.
        if ctxVar.mangledName not in self.context.variablesMap:
            self.raiseError("Internal error")

        self.typeId = self.context.variablesMap[ctxVar.mangledName].idType
        self.originalIdentifier: str = variableName
        self.identifier: str = ctxVar.mangledName

    def staticEval(self) -> StaticEvalValue:
        # ISO C99 6.7.8, identifiers with static storage duration must be initialized with a 
        # constant expression. Section 6.6 specifies that no memory content must be read during 
        # evaluations.
        # For the moment, we'll return a "variable" type value which should be converted to a 
        # "pointer" by an AddressOf expression.
        return StaticEvalValue(StaticEvalType.VARIABLE, self.identifier)

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}Variable: {self.typeId} {self.identifier}\n'
    
class Cast(Exp):
    @staticmethod
    def checkImplicitCast(srcType: DeclaratorType, isSourceNullPointer: bool, castType: DeclaratorType) -> bool:
        # Following C99 6.5.16.1, it can only be implicitly casted if:
        # - Both types are arithmetical types.
        bothArith = castType.isArithmetic() and srcType.isArithmetic()
        
        # - One operand is a pointer to an object or incomplete type and the other is a pointer to a
        # *qualified or unqualified version of void*, and the type *pointed to by the cast type* has 
        # all the qualifiers of the type type *pointed to by the source type*.
        isSourcePointer = isinstance(srcType, PointerDeclaratorType)
        isCastPointer = isinstance(castType, PointerDeclaratorType)

        isSourceVoidPointer = isSourcePointer and srcType.declarator.unqualify() == TypeSpecifier.VOID.toBaseType()
        isCastVoidPointer = isCastPointer and castType.declarator.unqualify() == TypeSpecifier.VOID.toBaseType()
        hasAllQualifiers = isSourcePointer and isCastPointer and (castType.declarator.getTypeQualifiers().contains(srcType.declarator.getTypeQualifiers()))

        toVoidPointerType = isSourcePointer and not isSourceVoidPointer and isCastVoidPointer and hasAllQualifiers
        fromVoidPointerType = isSourceVoidPointer and isCastPointer and not isCastVoidPointer and hasAllQualifiers
        voidPointerCheck = toVoidPointerType or fromVoidPointerType

        # - The cast type is a pointer and the source is a null pointer constant.
        fromNullPointer = isCastPointer and isSourceNullPointer

        # - Both operands are pointers to qualified or unqualified versions of compatible types, and
        # the type pointed to by the cast type has all the qualifiers of the type pointed to by the
        # source type.
        validPointers = isSourcePointer and isCastPointer and \
            (srcType.declarator.unqualify() == castType.declarator.unqualify()) and \
            hasAllQualifiers
        
        ret = bothArith or voidPointerCheck or fromNullPointer or validPointers
        return ret 

    def validateCast(self):
        if self.isImplicit and not Cast.checkImplicitCast(self.inner.typeId, self.inner.isNullPointer(), self.typeId):
            self.raiseError(f"Cannot implicitly cast from ({self.inner.typeId}) to ({self.typeId})")

        # ISO rule C99 6.5.4.2.
        if not self.typeId.isScalar() and self.typeId.unqualify() != TypeSpecifier.VOID.toBaseType():
            self.raiseError(f"Cannot cast from {self.inner.typeId} to {self.typeId}")

        # Check if the types are the same but differ on their qualifiers. In this case, the cast is 
        # always permitted.
        if self.inner.typeId.unqualify() == self.typeId.unqualify():
            return

        # Cannot cast a decimal to a pointer.
        decToPointer = isinstance(self.inner.typeId, BaseDeclaratorType) and \
                       self.inner.typeId.isDecimal() and \
                       isinstance(self.typeId, PointerDeclaratorType)
        # Cannot cast a pointer to a decimal.
        pointerToDec = isinstance(self.typeId, BaseDeclaratorType) and \
                       self.typeId.isDecimal() and \
                       isinstance(self.inner.typeId, PointerDeclaratorType)
        # Cannot cast to an array.
        toArray = isinstance(self.typeId, ArrayDeclaratorType)
        # Cannot cast a void to any other type different than void.
        fromVoid = self.inner.typeId == TypeSpecifier.VOID.toBaseType() and self.typeId != TypeSpecifier.VOID.toBaseType()
        # Cannot cast from struct/union to scalar.
        castStructOrUnionToScalar = self.typeId.isScalar() and \
            isinstance(self.inner.typeId, BaseDeclaratorType) and \
            self.inner.typeId.baseType.name in ("STRUCT", "UNION")

        if decToPointer or pointerToDec or toArray or fromVoid or castStructOrUnionToScalar:
            self.raiseError(f"Cannot cast from ({self.inner.typeId}) to ({self.typeId})")

    def parse(self, *args):
        if len(args) > 0:
            # Parse from args.
            self.typeId: DeclaratorType = args[0]
            # These inner types will be already pre-preconverted.
            self.inner: Exp = args[1]
            self.isImplicit: bool = len(args) > 2 and args[2]
        else:
            # Parse from tokens.
            self.expect("(")
            _, declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
            if self.peek().id != ")":
                declarator = self.createChild(TopAbstractDeclarator)
                info: DeclaratorInformation = declarator.process(declType)
                self.typeId = info.type
            else:
                self.typeId = declType
            self.expect(")")

            self.inner: Exp = self._parseFactor().preconvertExpression()
            self.isImplicit = False

        self.validateCast()

    def staticEval(self) -> StaticEvalValue:
        innerValue = self.inner.staticEval()

        if isinstance(self.typeId, BaseDeclaratorType):
            if self.typeId.baseType.isDecimal():
                operand = innerValue.getFloatValue(self.inner)
                evalType = StaticEvalType.FLOAT
            else:
                operand = innerValue.getIntegerValue(self.inner)
                evalType = StaticEvalType.INTEGER
            
            retValue, warning = StaticEvaluation.parseValue(self.typeId.baseType, operand)
            if warning:
                warning.rise(self.raiseWarning, self.raiseError)
            return StaticEvalValue(evalType, str(retValue))
            
        if isinstance(self.typeId, PointerDeclaratorType):
            if self.inner.typeId.isInteger():
                # Integer to pointer.
                return StaticEvalValue(StaticEvalType.POINTER, "", innerValue.getIntegerValue(self.inner))
            if innerValue.valType == StaticEvalType.POINTER:
                # Pointer from one type to another. 
                return innerValue

        self.raiseError(f"Static evaluation of cast from {self.inner.typeId} to {self.typeId} not implemented")

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}Cast ({self.typeId}) (\n'
        ret += self.inner.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret

class UnaryOperator(enum.Enum):
    BITWISE_COMPLEMENT  = '~'
    PLUS                = '+'
    NEGATION            = '-'
    NOT                 = "!"
    INCREMENT           = "++"
    DECREMENT           = "--"
    DEREFERENCE         = "*"
    ADDRESS_OF          = "&"

    PRE_INCREMENT       = "++x"
    PRE_DECREMENT       = "--x"
    POST_INCREMENT      = "x++"
    POST_DECREMENT      = "x--"

    def convertToSimple(self) -> UnaryOperator:
        match self:
            case UnaryOperator.PRE_INCREMENT | UnaryOperator.POST_INCREMENT: 
                return UnaryOperator.INCREMENT
            case UnaryOperator.PRE_DECREMENT |UnaryOperator.POST_DECREMENT:
                return UnaryOperator.DECREMENT
            case _:
                return self

    @staticmethod
    def toPreUnaryOperator(token: Token) -> UnaryOperator|None:
        try:
            op = UnaryOperator(token.id)
        except:
            return None
        
        match op:
            case UnaryOperator.INCREMENT:   
                return UnaryOperator.PRE_INCREMENT
            case UnaryOperator.DECREMENT:   
                return UnaryOperator.PRE_DECREMENT
            case UnaryOperator.BITWISE_COMPLEMENT | UnaryOperator.NEGATION | UnaryOperator.PLUS | \
                 UnaryOperator.NOT | UnaryOperator.DEREFERENCE | UnaryOperator.ADDRESS_OF:
                return op
            case _:
                return None

    @staticmethod
    def toPostUnaryOperator(token: Token) -> UnaryOperator|None:
        try:
            op = UnaryOperator(token.value)
        except:
            return None
        
        match op:
            case UnaryOperator.INCREMENT:   
                return UnaryOperator.POST_INCREMENT
            case UnaryOperator.DECREMENT:   
                return UnaryOperator.POST_DECREMENT
            case _:
                return None

class Unary(Exp):
    def parse(self, operator: UnaryOperator, inner: Exp):
        self.unaryOperator: UnaryOperator = operator
        isLvalueAssignable = inner.isLvalueAssignable()
        self.inner: Exp = inner.preconvertExpression()

        if not self.inner.typeId.isScalar():
            self.raiseError(f"Expected a scalar expression, received {self.inner.typeId}")

        if isinstance(self.inner.typeId, PointerDeclaratorType):
            if self.unaryOperator in (UnaryOperator.BITWISE_COMPLEMENT, UnaryOperator.NEGATION, UnaryOperator.PLUS):
                self.raiseError("Invalid operation for pointer type")
            if not self.inner.typeId.declarator.isComplete():
                self.raiseError(f"Invalid operation on incomplete type {self.inner.typeId}")

        if self.unaryOperator in (UnaryOperator.PRE_INCREMENT,  UnaryOperator.PRE_DECREMENT, 
                                  UnaryOperator.POST_INCREMENT, UnaryOperator.POST_DECREMENT):
            if not isLvalueAssignable:
                self.raiseError(f"Expression must be a modifiable lvalue")

            # For the ++ and -- operators, we need to know wether a cast was done due to integer 
            # promotion. If a cast was made, the result to store must be recasted back to the original 
            # type. This final cast is carried out in the TAC stage.
            self.originalType = self.inner.typeId
            self.needsIntegerPromotion = self.inner != self.inner.checkIntegerPromotion()
        
        else:
            self.inner = self.inner.checkIntegerPromotion()

        if self.unaryOperator == UnaryOperator.NOT:
            self.typeId = TypeSpecifier.INT.toBaseType()
        else:
            # Note: for elements that need integer promotion, the result of the ++ or -- is the 
            # original type, not int.
            self.typeId = self.inner.typeId

        if self.unaryOperator == UnaryOperator.BITWISE_COMPLEMENT:
            if not isinstance(self.inner.typeId, BaseDeclaratorType) or \
               not self.inner.typeId.baseType.isInteger():
                self.raiseError("Cannot bitwise-complement a non-integer number")

    def staticEval(self) -> StaticEvalValue:
        innerValue = self.inner.staticEval()

        if self.unaryOperator == UnaryOperator.NOT and innerValue.valType == StaticEvalType.POINTER:
            # A pointer to a static variable is never null.
            retVal = 0
            warning = EVAL_OK
        else:
            if self.typeId.isDecimal():
                operand = innerValue.getFloatValue(self.inner)
            else:
                operand = innerValue.getIntegerValue(self.inner)
            retVal, warning = StaticEvaluation.evalDecl(
                self.unaryOperator.value, self.inner.typeId, self.typeId, operand)

        if warning:
            warning.rise(self.raiseWarning, self.raiseError)
            
        return StaticEvalValue(
            StaticEvalType.INTEGER if isinstance(retVal, int) else StaticEvalType.FLOAT, 
            str(retVal)
        )
        
    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}{self.typeId} Unary: {self.unaryOperator.name}(\n'
        ret += self.inner.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret
    
class BinaryOperator(enum.Enum):
    # In order of precedence.
    MULTIPLICATION      = "*"
    DIVISION            = "/"
    MODULUS             = "%"
    SUM                 = '+'
    SUBTRACT            = '-'
    BITWISE_LEFT_SHIFT  = "<<"
    BITWISE_RIGHT_SHIFT = ">>"
    GREATER_THAN        = ">"
    GREATER_OR_EQUAL    = ">="
    LESS_THAN           = "<"
    LESS_OR_EQUAL       = "<="
    EQUAL               = "=="
    NOT_EQUAL           = "!="
    BITWISE_AND         = "&"
    BITWISE_XOR         = "^"
    BITWISE_OR          = "|"
    AND                 = "&&"
    OR                  = "||"
    CONDITIONAL         = "?" # For ternary conditional.
    ASSIGNMENT          = "="

    COMPOUND_MULTIPLICATION      = "*="
    COMPOUND_DIVISION            = "/="
    COMPOUND_MODULUS             = "%="
    COMPOUND_SUM                 = '+='
    COMPOUND_SUBTRACT            = '-='
    COMPOUND_BITWISE_LEFT_SHIFT  = "<<="
    COMPOUND_BITWISE_RIGHT_SHIFT = ">>="
    COMPOUND_BITWISE_AND         = "&="
    COMPOUND_BITWISE_XOR         = "^="
    COMPOUND_BITWISE_OR          = "|="

    LOGIC_LEFT_SHIFT            = "logic<<"
    ARITHMETIC_LEFT_SHIFT       = "arith<<"
    LOGIC_RIGHT_SHIFT           = "logic>>"
    ARITHMETIC_RIGHT_SHIFT      = "arith>>"

    def isCompound(self) -> bool:
        return self.name.startswith("COMPOUND_")
    
    def isComparison(self) -> bool:
        return self in (BinaryOperator.GREATER_THAN, BinaryOperator.GREATER_OR_EQUAL, \
                        BinaryOperator.LESS_THAN, BinaryOperator.LESS_OR_EQUAL, \
                        BinaryOperator.EQUAL, BinaryOperator.NOT_EQUAL)

    def getCompoundOperator(self) -> BinaryOperator:
        if not self.isCompound(): 
            raise ValueError(f"{self} is not a compound operator")
        
        operationName = self.name.split("COMPOUND_")[1]
        return BinaryOperator[operationName]

    @staticmethod
    def toBinaryOperator(token: Token) -> BinaryOperator|None:
        try:
            op = BinaryOperator(token.value)
        except:
            return None
        return op
    
    def getBaseOperator(self) -> BinaryOperator:
        if self.isCompound():
            return BinaryOperator(self.value.replace("=", ""))
        if "logic" in self.value:
            return BinaryOperator(self.value.replace("logic", ""))
        if "arith" in self.value:
            return BinaryOperator(self.value.replace("arith", ""))
        return self
    
    # Based on https://en.cppreference.com/w/c/language/operator_precedence.html
    def getPrecedence(self) -> int:
        match self:
            case BinaryOperator.MULTIPLICATION:                 return 130 
            case BinaryOperator.DIVISION:                       return 130 
            case BinaryOperator.MODULUS:                        return 130 
            case BinaryOperator.SUM:                            return 120 
            case BinaryOperator.SUBTRACT:                       return 120 
            case BinaryOperator.BITWISE_LEFT_SHIFT:             return 110 
            case BinaryOperator.BITWISE_RIGHT_SHIFT:            return 110 
            case BinaryOperator.GREATER_THAN:                   return 100 
            case BinaryOperator.GREATER_OR_EQUAL:               return 100 
            case BinaryOperator.LESS_THAN:                      return 100 
            case BinaryOperator.LESS_OR_EQUAL:                  return 100 
            case BinaryOperator.EQUAL:                          return 90 
            case BinaryOperator.NOT_EQUAL:                      return 90 
            case BinaryOperator.BITWISE_AND:                    return 80 
            case BinaryOperator.BITWISE_XOR:                    return 70 
            case BinaryOperator.BITWISE_OR:                     return 60 
            case BinaryOperator.AND:                            return 50 
            case BinaryOperator.OR:                             return 40 
            case BinaryOperator.CONDITIONAL:                    return 30
            case BinaryOperator.ASSIGNMENT:                     return 20
            case BinaryOperator.COMPOUND_MULTIPLICATION:        return 20
            case BinaryOperator.COMPOUND_DIVISION:              return 20
            case BinaryOperator.COMPOUND_MODULUS:               return 20
            case BinaryOperator.COMPOUND_SUM:                   return 20
            case BinaryOperator.COMPOUND_SUBTRACT:              return 20
            case BinaryOperator.COMPOUND_BITWISE_LEFT_SHIFT:    return 20
            case BinaryOperator.COMPOUND_BITWISE_RIGHT_SHIFT:   return 20
            case BinaryOperator.COMPOUND_BITWISE_AND:           return 20
            case BinaryOperator.COMPOUND_BITWISE_XOR:           return 20
            case BinaryOperator.COMPOUND_BITWISE_OR:            return 20
            case _:
                raise ValueError("Invalid Binary operator to get its precedence")

class Binary(Exp):
    def validateExp(self, inputOp: BinaryOperator, exp1: Exp, exp2: Exp, isExp1Assignable: bool):
        INT_ONLY_OPERATORS = (BinaryOperator.BITWISE_AND, BinaryOperator.BITWISE_OR, \
                              BinaryOperator.BITWISE_XOR, BinaryOperator.BITWISE_LEFT_SHIFT, \
                              BinaryOperator.BITWISE_RIGHT_SHIFT, BinaryOperator.MODULUS)

        if not exp1.typeId.isComplete():
            self.raiseError(f"Expected a scalar expression, received {exp1.typeId}")
        if not exp2.typeId.isComplete():
            self.raiseError(f"Expected a scalar expression, received {exp2.typeId}")

        if isExp1Pointer := isinstance(exp1.typeId, (PointerDeclaratorType, ArrayDeclaratorType)):
            isExp1PointerComplete = exp1.typeId.declarator.isComplete()
        else:
            isExp1PointerComplete = True

        if isExp2Pointer := isinstance(exp2.typeId, (PointerDeclaratorType, ArrayDeclaratorType)):
            isExp2PointerComplete = exp2.typeId.declarator.isComplete()
        else:
            isExp2PointerComplete = True

        op = inputOp
        if inputOp.isCompound():
            if not isExp1Assignable:
                self.raiseError("Left expression in compound operation must be lvalue")
            op = inputOp.getCompoundOperator()

        # Only arithmetic types can be multiplied and divided.
        if op in (BinaryOperator.MULTIPLICATION, BinaryOperator.DIVISION):
            if not exp1.typeId.isArithmetic() or not exp2.typeId.isArithmetic():
                self.raiseError("Expressions must have arithmetic type")

        # These operations only expect integers to work with in both its inputs.
        elif op in INT_ONLY_OPERATORS and \
           (not exp1.typeId.isInteger() or not exp2.typeId.isInteger()):
            self.raiseError("Invalid operation on non-integer type")

        # Sum can operate with arithmetic types or combining an integer with a pointer type.
        elif op == BinaryOperator.SUM:
            if isExp1Pointer and isExp2Pointer:
                self.raiseError("One of the expressions must have integer type")

            if not isExp1PointerComplete:
                self.raiseError(f"Expected a pointer to a complete type, received {exp1.typeId}")
            if not isExp2PointerComplete:
                self.raiseError(f"Expected a pointer to a complete type, received {exp2.typeId}")

            if (isExp1Pointer and not exp2.typeId.isInteger()) or \
               (isExp1Pointer and inputOp.isCompound() and isExp2Pointer):
                self.raiseError("Second expression must have integer type")
            
            if isExp2Pointer and not exp1.typeId.isInteger():
                self.raiseError("First expression must have integer type")

        # Subtract can operate with arithmetic types, with a pointer first and integer second, or 
        # with two pointers of the same type.
        elif op == BinaryOperator.SUBTRACT:
            if isExp1Pointer and isExp2Pointer and exp1.typeId != exp2.typeId:
                self.raiseError("Both expressions must have the same pointer type")

            if not isExp1PointerComplete:
                self.raiseError(f"Expected a pointer to a complete type, received {exp1.typeId}")
            if not isExp2PointerComplete:
                self.raiseError(f"Expected a pointer to a complete type, received {exp2.typeId}")

            if isExp1Pointer and not isExp2Pointer and not exp2.typeId.isInteger():
                self.raiseError("Second expression must have integer type")

            if not isExp1Pointer and isExp2Pointer:
                self.raiseError("Second expression cannot be a pointer")
            
        # Comparisons can only happen between scalars. AND and OR can only operate with scalars.
        elif op.isComparison() or op in (BinaryOperator.AND, BinaryOperator.OR):
            if not exp1.typeId.isScalar() or not exp2.typeId.isScalar():
                self.raiseError(f"Operand types {exp1.typeId} and {exp2.typeId} are incompatible")

    def parse(self, op: BinaryOperator, exp1: Exp, exp2: Exp):
        isLvalueAssignable = exp1.isLvalueAssignable()

        self.exp1 = exp1.preconvertExpression()
        self.exp2 = exp2.preconvertExpression()
        self.validateExp(op, self.exp1, self.exp2, isLvalueAssignable)
        
        self.compoundBinary = op.isCompound()
        self.binaryOperator = op.getCompoundOperator() if self.compoundBinary else op
        # Used in compound operations, in the TAC phase.
        self.exp1OriginalType = exp1.typeId
        self.exp1IsCasted = False
        # On compound operations, the values may be casted to their common type to be operated. This
        # intermediary type is stored in castType.
        self.castType: DeclaratorType

        if self.binaryOperator in (BinaryOperator.AND, BinaryOperator.OR):
            # No need to to perform type conversions for AND and OR.
            self.typeId = BaseDeclaratorType(TypeSpecifier.INT, TypeQualifier())

        elif self.binaryOperator in (BinaryOperator.BITWISE_LEFT_SHIFT, BinaryOperator.BITWISE_RIGHT_SHIFT):
            exp1Promoted = self.exp1.checkIntegerPromotion()
            if self.exp1 != exp1Promoted:
                self.exp1 = exp1Promoted.preconvertExpression()
                self.exp1IsCasted = True
            exp2Promoted = self.exp2.checkIntegerPromotion()
            if self.exp2 != exp2Promoted:
                self.exp2 = exp2Promoted.preconvertExpression()

            self.castType = self.exp1.typeId

            # Depending on the sign of the type, the shifts can be arithmetic or logical.
            if self.binaryOperator == BinaryOperator.BITWISE_LEFT_SHIFT:
                # C99 leaves << as undefined on negative numbers. I just use logic in this case.
                self.binaryOperator = BinaryOperator.LOGIC_LEFT_SHIFT
            else:
                # C99 says >> acts on signed integers by sign extension (arithmetic shift).
                # Otherwise, use logical shift.
                if self.castType.isSigned():
                    self.binaryOperator = BinaryOperator.ARITHMETIC_RIGHT_SHIFT
                else:
                    self.binaryOperator = BinaryOperator.LOGIC_RIGHT_SHIFT

            if self.compoundBinary:
                # The return type of a compound operation, is the lvalue's type.
                self.typeId = self.exp1OriginalType
            else:
                # x << y has type of x.
                self.typeId = self.exp1.typeId

        elif self.binaryOperator == BinaryOperator.SUM:
            isExp1Pointer = isinstance(exp1.typeId, (PointerDeclaratorType, ArrayDeclaratorType))
            isExp2Pointer = isinstance(exp2.typeId, (PointerDeclaratorType, ArrayDeclaratorType))

            if not isExp1Pointer and not isExp2Pointer:
                # Arithmetic sum.
                # Apply conversions to the expressions if necessary.
                commonType = self.getCommonType(self.exp1, self.exp2)
                if commonType is None:
                    self.raiseError(f"No common type of {self.exp1.typeId} and {self.exp2.typeId}")

                self.castType = commonType
                if self.exp1.typeId != commonType:
                    self.exp1 = self.createChild(Cast, commonType, self.exp1).preconvertExpression()
                    self.exp1IsCasted = True
                if self.exp2.typeId != commonType:
                    self.exp2 = self.createChild(Cast, commonType, self.exp2).preconvertExpression()

                if self.compoundBinary:
                    # The return type of a compound operation, is the lvalue's type.
                    self.typeId = self.exp1OriginalType
                else:
                    # Arithmetic expressions will return the common type.
                    self.typeId = commonType

            elif isExp1Pointer and not isExp2Pointer:
                # Pointer sum: Convert exp2 to a long.
                if self.exp2.typeId != TypeSpecifier.LONG.toBaseType():
                    self.exp2 = self.createChild(Cast, TypeSpecifier.LONG.toBaseType(), self.exp2).preconvertExpression()
                # The result is a pointer.
                self.typeId = self.exp1.typeId
                self.castType = self.typeId
                
            elif isExp2Pointer and not isExp1Pointer:
                # Pointer sum: Convert exp1 to a long.
                if self.exp1.typeId != TypeSpecifier.LONG.toBaseType():
                    self.exp1 = self.createChild(Cast, TypeSpecifier.LONG.toBaseType(), self.exp1).preconvertExpression()
                # The result is a pointer.
                self.typeId = self.exp2.typeId
                self.castType = self.typeId

        elif self.binaryOperator == BinaryOperator.SUBTRACT:
            isExp1Pointer = isinstance(exp1.typeId, (PointerDeclaratorType, ArrayDeclaratorType))
            isExp2Pointer = isinstance(exp2.typeId, (PointerDeclaratorType, ArrayDeclaratorType))

            if not isExp1Pointer and not isExp2Pointer:
                # Arithmetic subtraction.
                # Apply conversions to the expressions if necessary.
                commonType = self.getCommonType(self.exp1, self.exp2)
                if commonType is None:
                    self.raiseError(f"No common type of {self.exp1.typeId} and {self.exp2.typeId}")

                self.castType = commonType
                if self.exp1.typeId != commonType:
                    self.exp1 = self.createChild(Cast, commonType, self.exp1).preconvertExpression()
                    self.exp1IsCasted = True
                if self.exp2.typeId != commonType:
                    self.exp2 = self.createChild(Cast, commonType, self.exp2).preconvertExpression()

                if self.compoundBinary:
                    # The return type of a compound operation, is the lvalue's type.
                    self.typeId = self.exp1OriginalType
                else:
                    # Arithmetic expressions will return the common type.
                    self.typeId = commonType

            elif isExp1Pointer and not isExp2Pointer:
                # Pointer subtraction: Convert exp2 to a long.
                if self.exp2.typeId != TypeSpecifier.LONG.toBaseType():
                    self.exp2 = self.createChild(Cast, TypeSpecifier.LONG.toBaseType(), self.exp2).preconvertExpression()
                # The result is a pointer.
                self.typeId = self.exp1.typeId
                self.castType = self.typeId

            elif isExp2Pointer and isExp1Pointer:
                # Pointer subtraction: the result is a long.
                self.typeId = TypeSpecifier.LONG.toBaseType()
                self.castType = self.typeId

        else:
            # Apply conversions to the expressions if necessary.
            commonType = self.getCommonType(self.exp1, self.exp2)
            if commonType is None:
                self.raiseError(f"No common type of {self.exp1.typeId} and {self.exp2.typeId}")
            
            # The resulting type used when operating both terms. 
            # This field is used on compound operations in the TAC section.
            self.castType = commonType

            match commonType:
                case BaseDeclaratorType():
                    if self.exp1.typeId != commonType:
                        self.exp1 = self.createChild(Cast, commonType, self.exp1).preconvertExpression()
                        self.exp1IsCasted = True
                    if self.exp2.typeId != commonType:
                        self.exp2 = self.createChild(Cast, commonType, self.exp2).preconvertExpression()

                case PointerDeclaratorType() | ArrayDeclaratorType():
                    if self.binaryOperator in (BinaryOperator.EQUAL, BinaryOperator.NOT_EQUAL):
                        # Only == and != allows implicit castings of pointers.
                        if self.exp1.typeId != commonType:
                            self.exp1 = self.createChild(Cast, commonType, self.exp1).preconvertExpression()
                            self.exp1IsCasted = True
                        if self.exp2.typeId != commonType:
                            self.exp2 = self.createChild(Cast, commonType, self.exp2).preconvertExpression()

                case _:
                    self.raiseError(f"Binary operation not supported for {commonType}")

            if self.exp1.typeId != commonType or self.exp2.typeId != commonType:
                self.raiseError(f"Invalid expression: {self.exp1.typeId} {self.binaryOperator.value} {self.exp2.typeId}")

            # Now, both expressions should have the same type.
            if self.binaryOperator.isComparison():
                # Doing a comparison will always return an integer.
                self.typeId = BaseDeclaratorType(TypeSpecifier.INT)
            elif self.compoundBinary:
                # The return type of a compound operation, is the lvalue's type.
                self.typeId = self.exp1OriginalType
            else:
                # Arithmetic expressions will return the common type.
                self.typeId = commonType

    def staticEval(self) -> StaticEvalValue:
        exp1Eval = self.exp1.staticEval()
        exp2Eval = self.exp2.staticEval()

        # POINTER EVALUATIONS.
        isExp1Pointer = exp1Eval.valType == StaticEvalType.POINTER
        isExp2Pointer = exp2Eval.valType == StaticEvalType.POINTER

        if self.binaryOperator == BinaryOperator.SUM:
            if isExp1Pointer and not isExp2Pointer:
                if not isinstance(self.exp1.typeId, PointerDeclaratorType):
                    raise ValueError()
                
                # Pointer sum: exp2 is long.
                index = exp2Eval.getIntegerValue(self.exp2)
                return StaticEvalValue(StaticEvalType.POINTER, 
                                        exp1Eval.value, 
                                        exp1Eval.pointerOffset + index * self.exp1.typeId.declarator.getByteSize())

            if isExp2Pointer and not isExp1Pointer:
                if not isinstance(self.exp2.typeId, PointerDeclaratorType):
                    raise ValueError()

                # Pointer sum: exp1 is long.
                index = exp1Eval.getIntegerValue(self.exp1)
                return StaticEvalValue(StaticEvalType.POINTER, 
                                        exp2Eval.value,
                                        exp2Eval.pointerOffset + index * self.exp2.typeId.declarator.getByteSize())

        if self.binaryOperator == BinaryOperator.SUBTRACT:
            if isExp1Pointer and not isExp2Pointer:
                if not isinstance(self.exp1.typeId, PointerDeclaratorType):
                    raise ValueError()
                
                # Pointer subtraction: exp2 is a long.
                index = exp2Eval.getIntegerValue(self.exp2)
                return StaticEvalValue(StaticEvalType.POINTER, 
                                        exp1Eval.value, 
                                        exp1Eval.pointerOffset - index * self.exp1.typeId.declarator.getByteSize())

            if isExp2Pointer and isExp1Pointer:
                if not isinstance(self.exp1.typeId, PointerDeclaratorType):
                    raise ValueError()
                
                # Pointer subtraction: the result is a long. In this case, the base address of both
                # must be the same.
                # It's important to use the addresses and not the indices to calculate the resulting
                # index. Subtracting indices does not generate the right index.
                if exp1Eval.value != exp2Eval.value:
                    self.raiseError("Cannot subtract pointers, base addresses must be the same")
                
                index = (exp1Eval.pointerOffset - exp2Eval.pointerOffset) // self.exp1.typeId.declarator.getByteSize()
                return StaticEvalValue(StaticEvalType.INTEGER, str(index))

        if isExp1Pointer and isExp2Pointer and self.binaryOperator.isComparison():
            if exp1Eval.value != exp2Eval.value:
                self.raiseError("Cannot compare pointers with different base addresses")
            # Compare with the indices.
            result, warning = StaticEvaluation.eval(
                self.binaryOperator.value, TypeSpecifier.LONG, TypeSpecifier.LONG,
                exp1Eval.pointerOffset, exp2Eval.pointerOffset
            )
            if warning:
                warning.rise(self.raiseWarning, self.raiseError)

            return StaticEvalValue(StaticEvalType.INTEGER, str(result))

        # ARITHMETIC EVALUATIONS.
        if self.typeId.isDecimal():
            exp1 = exp1Eval.getFloatValue(self.exp1)
            exp2 = exp2Eval.getFloatValue(self.exp2)
        else:
            exp1 = exp1Eval.getIntegerValue(self.exp1)
            exp2 = exp2Eval.getIntegerValue(self.exp2)

        if self.compoundBinary:
            self.raiseError("Cannot evaluate during compilation.")

        result, warning = StaticEvaluation.evalDecl(
            self.binaryOperator.getBaseOperator().value, self.exp1.typeId, self.typeId, exp1, exp2)
        if warning:
            warning.rise(self.raiseWarning, self.raiseError)

        return StaticEvalValue(
            StaticEvalType.INTEGER if isinstance(result, int) else StaticEvalType.FLOAT,
            str(result)
        )

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}{self.typeId} Binary: {"Compound " if self.compoundBinary else ""}{self.binaryOperator.name}(\n'
        ret += self.exp1.print(padding + PADDING_INCREMENT)
        ret += self.exp2.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret

class Assignment(Exp):
    def parse(self, exp1: Exp, exp2: Exp):
        if not exp1.isLvalueAssignable():
            self.raiseError("Left expression must be a modifiable lvalue")

        # If exp1's type is a struct or union, check that the inner members are not constant.
        if isinstance(exp1.typeId, BaseDeclaratorType) and \
           exp1.typeId.baseType.name in ("STRUCT", "UNION"):
            for mem in exp1.typeId.baseType.getMembers():
                if mem.type.getTypeQualifiers().const:
                    self.raiseError("Left expression must be a modifiable lvalue")

        # Exp1 = Exp2
        self.exp1 = exp1.preconvertExpression()
        if self.exp1.typeId == TypeSpecifier.VOID.toBaseType():
            self.raiseError("Cannot assign to a void")

        self.exp2 = exp2.preconvertExpression()

        if self.exp2.typeId != self.exp1.typeId:
            # If the types aren't the same, cast exp2 to exp1's type.
            # This is an implicit cast.
            self.exp2 = self.createChild(Cast, self.exp1.typeId, self.exp2, True).preconvertExpression()
        
        # The return type of the assignment is exp1's type.
        self.typeId = self.exp1.typeId

    def staticEval(self) -> StaticEvalValue:
        raise ValueError()

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f"{pad}{self.typeId} Assignment(\n"
        ret += self.exp1.print(padding + PADDING_INCREMENT)
        ret += self.exp2.print(padding + PADDING_INCREMENT)
        ret += f"{pad})\n"
        return ret
    
class TernaryConditional(Exp):
    def parse(self, condition: Exp, thenExp: Exp, elseExp: Exp):
        self.condition = condition.preconvertExpression()
        if not self.condition.typeId.isScalar():
            self.raiseError(f"Expected a scalar expression, received {self.condition.typeId}")

        self.thenExp = thenExp.preconvertExpression()
        self.elseExp = elseExp.preconvertExpression()

        # Apply conversions to the expressions if necessary.
        commonType = self.getCommonType(self.thenExp, self.elseExp)
        if commonType is None:
            self.raiseError(f"No common type of {self.thenExp.typeId} and {self.elseExp.typeId}")

        if self.thenExp.typeId != commonType:
            self.thenExp = self.createChild(Cast, commonType, self.thenExp).preconvertExpression()
        if self.elseExp.typeId != commonType:
            self.elseExp = self.createChild(Cast, commonType, self.elseExp).preconvertExpression()
        # The return value has the same common type.
        self.typeId = commonType

    def staticEval(self) -> StaticEvalValue:
        conditionEval = self.condition.staticEval()
        thenEval = self.thenExp.staticEval()
        elseEval = self.elseExp.staticEval()

        return thenEval if conditionEval.getIntegerValue(self.condition) else elseEval

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}{self.typeId} TernaryConditional(\n'
        ret += f'{pad}- Condition:\n'
        ret += self.condition.print(padding + PADDING_INCREMENT)
        ret += f'{pad}- Then:\n'
        ret += self.thenExp.print(padding + PADDING_INCREMENT)
        ret += f'{pad}- Else:\n'
        ret += self.elseExp.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret
    
class FunctionCall(Exp):
    def parse(self, *args):
        self.funcIdentifier = self.expect("identifier").value

        if self.funcIdentifier not in self.context.identifierMap:
            self.raiseError(f"Function {self.funcIdentifier} is not declared")
        else:
            # The function could be obscured by a variable. Search in the identifier map.
            if self.funcIdentifier not in self.context.identifierMap:
                self.raiseError(f"Identifier {self.funcIdentifier} is not declared")

            if self.context.identifierMap[self.funcIdentifier].identifierType != IdentifierType.FUNCTION:
                self.raiseError(f"{self.funcIdentifier} is not a function")

        self.expect("(")

        funcCtx = self.context.functionMap[self.funcIdentifier]
        self.typeId = funcCtx.returnType

        self.argumentList: list[Exp] = []
        
        if self.peek().id != ")":
            argumentIndex: int = 0
            while True:
                argExp: Exp = self.createChild(Exp).preconvertExpression()
                
                # See if this argument needs a cast before being passed to the function.
                if argExp.typeId != funcCtx.arguments[argumentIndex].type:
                    argExp = self.createChild(Cast, funcCtx.arguments[argumentIndex].type, argExp, True).preconvertExpression()

                self.argumentList.append(argExp)

                if self.peek().id == ",":
                    self.pop()
                    argumentIndex += 1
                else:
                    break

        self.expect(")")

        passedArgsLen = len(self.argumentList)
        funcDeclLen = len(funcCtx.arguments)
        if passedArgsLen != funcDeclLen:
            self.raiseError(f"Expected {funcDeclLen} arguments, received {passedArgsLen}")

    def staticEval(self) -> StaticEvalValue:
        self.raiseError("Cannot evaluate a function call during compilation")        

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}{self.typeId} FunctionCall: {self.funcIdentifier}(\n'
        if len(self.argumentList) > 0:
            for exp in self.argumentList:
                ret += exp.print(padding + PADDING_INCREMENT)
        else:
            ret += f'{pad}  void\n'
        ret += f'{pad})\n'
        return ret
    
class Dereference(Exp):
    def parse(self):
        self.inner: Exp = self._parseFactor().preconvertExpression()

        if isinstance(self.inner.typeId, PointerDeclaratorType) and self.inner.typeId.declarator == TypeSpecifier.VOID:
            self.raiseError(f"Cannot dereference *void")

        if isinstance(self.inner.typeId, (PointerDeclaratorType, ArrayDeclaratorType)):
            self.typeId = self.inner.typeId.declarator
        else:
            self.raiseError("Cannot dereference non-pointer")

    def staticEval(self) -> StaticEvalValue:
        self.raiseError("Cannot evaluate a dereference during compilation")        

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}{self.typeId} Dereference(\n'
        ret += self.inner.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret

class AddressOf(Exp):
    def parse(self, inner: Exp|None = None):
        if inner is None:
            self.inner: Exp = self._parseFactor()
            if not self.inner.isLvalue():
                self.raiseError(f"Cannot take the address of non-lvalue: {self.inner}")
        else:
            self.inner = inner

        self.typeId = PointerDeclaratorType(self.inner.typeId)
    
    def staticEval(self) -> StaticEvalValue:
        innerEval = self.inner.staticEval()
        if innerEval.valType != StaticEvalType.VARIABLE:
            self.raiseError("Expected a variable for the static evaluation")
        return StaticEvalValue(StaticEvalType.POINTER, innerEval.value, innerEval.pointerOffset)

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}{self.typeId} AddressOf(\n'
        ret += self.inner.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret
    
class Subscript(Exp):
    def parse(self, inner: Exp):
        # In C, the pointer and the index can be interchanged: x[1] and 1[x] are the same. 
        self.pointer = inner.preconvertExpression()

        self.expect("[")
        self.index: Exp = self.createChild(Exp).preconvertExpression()
        self.expect("]")

        # Switch in case the index is the pointer.
        if isinstance(self.index.typeId, (ArrayDeclaratorType, PointerDeclaratorType)) and \
           not isinstance(self.pointer.typeId, (ArrayDeclaratorType, PointerDeclaratorType)):
            temp = self.pointer
            self.pointer = self.index
            self.index = temp

        if isinstance(self.pointer.typeId, (ArrayDeclaratorType, PointerDeclaratorType)):
            if not self.pointer.typeId.declarator.isComplete():
                self.raiseError("Cannot do subscript of an incomplete type")
        else:
            self.raiseError(f"Cannot do subscript {self.pointer.typeId} of non-pointer type")
        
        if not self.index.typeId.isInteger():
            self.raiseError("Index must be an integer")

        if self.index.typeId != TypeSpecifier.LONG.toBaseType():
            self.index = self.createChild(Cast, TypeSpecifier.LONG.toBaseType(), self.index, True)

        # Get inner type.
        self.typeId = self.pointer.typeId.declarator

    def staticEval(self) -> StaticEvalValue:
        if not isinstance(self.pointer.typeId, PointerDeclaratorType):
            raise ValueError()

        innerEval = self.pointer.staticEval()
        if innerEval.valType != StaticEvalType.POINTER:
            self.raiseError("Expression must have a constant value")

        indexEval = self.index.staticEval()

        try:
            indexEvalInteger = indexEval.getIntegerValue(self.index)
        except:
            self.raiseError("Cannot evaluate the index")

        # This is still a variable.
        return StaticEvalValue(StaticEvalType.VARIABLE, 
                               innerEval.value, 
                               innerEval.pointerOffset + indexEvalInteger * self.pointer.typeId.declarator.getByteSize())

    def print(self, padding: int) -> str:
        pad = " " * padding
        ret  = f'{pad}{self.typeId} Subscript(\n'
        ret += f'{pad}- Expression:\n'
        ret += self.pointer.print(padding + PADDING_INCREMENT)
        ret += f'{pad}- Index:\n'
        ret += self.index.print(padding + PADDING_INCREMENT)
        ret += f'{pad})\n'
        return ret

class Dot(Exp):
    def parse(self, leftExp: Exp):
        leftQualifiers = leftExp.typeId.getTypeQualifiers()
        self.leftExp: Exp = leftExp.preconvertExpression()
        
        if not isinstance(self.leftExp.typeId, BaseDeclaratorType) or \
           self.leftExp.typeId.baseType.name not in ("STRUCT", "UNION"):
            self.raiseError(f"Expected an struct or union, received {self.leftExp.typeId}")

        self.expect(".")
        self.member: str = self.expect("identifier").value

        self.memberInfo: ParameterInformation|None = self.leftExp.typeId.baseType.getMember(self.member)
        if self.memberInfo is None:
            self.raiseError(f"Member {self.member} is not part of {self.leftExp.typeId}")

        self.typeId = self.memberInfo.type.copy()
        # It inherits the type qualifiers of the left expression.
        self.typeId.setTypeQualifiers(self.typeId.getTypeQualifiers().union(leftQualifiers))

    # Used to check if the dot expression is an lvalue. For example:
    # str.a.b.c is an lvalue if str is an lvalue.
    # func().a.b is not an lvalue because func() is not an lvalue.
    def getFirstDotOperand(self) -> Exp:
        def getDotLeftExp(inner: Exp) -> Exp:
            if isinstance(inner, Dot):
                return getDotLeftExp(inner.leftExp)
            else:
                return inner
        
        return getDotLeftExp(self.leftExp)

    def staticEval(self) -> StaticEvalValue:
        leftEval = self.leftExp.staticEval()
        # Calculate the byte offset of the member and add it to the offset of leftEval.
        if self.memberInfo is None:
            raise ValueError()
        return StaticEvalValue(StaticEvalType.VARIABLE, 
                               leftEval.value, 
                               leftEval.pointerOffset + self.memberInfo.offset)

    def print(self, padding: int) -> str:
        pad = " " * padding
        leftExpString = self.leftExp.print(0).replace("\n", "")
        return f'{pad}{leftExpString}.{self.member}\n'
    
class Arrow(Exp):
    def parse(self, leftExp: Exp):
        # Kind of redundant, but needed for Pylance.
        if not isinstance(leftExp.typeId, (ArrayDeclaratorType, PointerDeclaratorType)):
            self.raiseError("Expected a pointer to struct or union")

        leftQualifiers = leftExp.typeId.declarator.getTypeQualifiers()
        self.leftExp: Exp = leftExp.preconvertExpression()

        if not isinstance(self.leftExp.typeId, PointerDeclaratorType) or \
           not isinstance(self.leftExp.typeId.declarator, BaseDeclaratorType) or \
           self.leftExp.typeId.declarator.baseType.name not in ("STRUCT", "UNION"):
            self.raiseError(f"Expected a pointer to struct or union, received {self.leftExp.typeId}")

        self.expect("->")
        self.member = self.expect("identifier").value

        # Check the type.
        leftExpMangledName: str = self.leftExp.typeId.declarator.baseType.identifier
        if self.leftExp.typeId.declarator.baseType.name in ("STRUCT", "UNION") and \
           self.context.tagsMap.get(leftExpMangledName) is None:
            self.raiseError(f"{self.leftExp.typeId} is not a valid type")

        self.memberInfo: ParameterInformation|None = self.leftExp.typeId.declarator.baseType.getMember(self.member)
        if self.memberInfo is None:
            self.raiseError(f"Member {self.member} is not part of {self.leftExp.typeId}")

        self.typeId = self.memberInfo.type.copy()
        # It inherits the type qualifiers of the left expression.
        self.typeId.setTypeQualifiers(self.typeId.getTypeQualifiers().union(leftQualifiers))

    def staticEval(self) -> StaticEvalValue:
        leftEval = self.leftExp.staticEval()
        # Calculate the byte offset of the member and add it to the offset of leftEval.
        if self.memberInfo is None:
            raise ValueError()
        return StaticEvalValue(StaticEvalType.VARIABLE, 
                               leftEval.value, 
                               leftEval.pointerOffset + self.memberInfo.offset)
    
    def print(self, padding: int) -> str:
        pad = " " * padding
        leftExpString = self.leftExp.print(0).replace("\n", "")
        return f'{pad}{leftExpString}->{self.member}\n'

class SizeOf(Exp):
    def parse(self, *args):
        self.typeId = TypeSpecifier.ULONG.toBaseType()

        self.expect("sizeof")
        self.inner: Exp = self._parseUnaryExpression()

        if not self.inner.typeId.isComplete():
            self.raiseError("Cannot calculate size of incomplete type")
    
    def staticEval(self) -> StaticEvalValue:
        return StaticEvalValue(StaticEvalType.INTEGER, str(self.inner.typeId.getByteSize()))

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}SizeOf({self.inner})\n'

class SizeOfType(Exp):
    def parse(self, *args):
        self.typeId = TypeSpecifier.ULONG.toBaseType()

        self.expect("sizeof")
        self.expect("(")
        _, declType, _ = self.getStorageClassAndDeclaratorType(expectsStorageClass=False)
        if self.peek().id != ")":
            declarator = self.createChild(TopAbstractDeclarator)
            info: DeclaratorInformation = declarator.process(declType)
            self.inner = info.type
        else:
            self.inner = declType
        self.expect(")")

        if not self.inner.isComplete():
            self.raiseError("Cannot calculate size of incomplete type")

    def staticEval(self) -> StaticEvalValue:
        return StaticEvalValue(StaticEvalType.INTEGER, str(self.inner.getByteSize()))

    def print(self, padding: int) -> str:
        pad = " " * padding
        return f'{pad}SizeOf({self.typeId})\n'


"""
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
INITIALIZERS
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

class Initializer(AST):
    @abstractmethod
    def parse(self, *args):
        pass

    @abstractmethod
    def staticEval(self) -> list[StaticEvalValue]:
        pass

    @abstractmethod
    def print(self, padding: int) -> str:
        pass

class SingleInitializer(Initializer):
    def parse(self, expectedType: DeclaratorType, initExp: Exp|None = None):
        self.typeId = expectedType
        
        if initExp is None:
            self.init: Exp = self.createChild(Exp).preconvertExpression()
        else:
            self.init: Exp = initExp
        
        # Store the original type. Needed for strict initializers.
        self.precastType = self.init.typeId
        if self.init.typeId != expectedType:
            self.init = self.createChild(Cast, expectedType, self.init, True)

    def staticEval(self) -> list[StaticEvalValue]:
        return [self.init.staticEval()]

    def print(self, padding: int) -> str:
        return self.init.print(padding)

    @staticmethod
    def createZeroInitializer(initializerType: DeclaratorType) -> Initializer:
        if isinstance(initializerType, BaseDeclaratorType):
            match initializerType.baseType:
                case TypeSpecifier.VOID:
                    raise ValueError("Cannot create zero initializer of an incomplete type (void)")
                case TypeSpecifier.SIGNED_CHAR | TypeSpecifier.CHAR:
                    constRetType = CharConstant
                case TypeSpecifier.UCHAR:   
                    constRetType = UCharConstant
                case TypeSpecifier.SHORT:   
                    constRetType = ShortConstant
                case TypeSpecifier.SHORT:   
                    constRetType = ShortConstant
                case TypeSpecifier.INT:     
                    constRetType = IntConstant
                case TypeSpecifier.LONG:    
                    constRetType = LongConstant
                case TypeSpecifier.USHORT:  
                    constRetType = UShortConstant
                case TypeSpecifier.UINT:    
                    constRetType = UIntConstant
                case TypeSpecifier.ULONG:   
                    constRetType = ULongConstant
                case TypeSpecifier.DOUBLE:  
                    constRetType = DoubleConstant
                case TypeSpecifier.FLOAT:   
                    constRetType = FloatConstant
                case _:
                    raise ValueError(f"Cannot create zero initializer of type {initializerType}")
                
            return SingleInitializer([], None, None, initializerType.baseType.toBaseType(),  constRetType([], None, None, "0"))

        elif isinstance(initializerType, PointerDeclaratorType):
            return SingleInitializer([], None, None, initializerType,  ULongConstant([], None, None, "0"))
        
        else:
            raise ValueError(f"Cannot create zero initializer of type {initializerType}")
    
class CompoundInitializer(Initializer):
    def parse(self, expectedType: DeclaratorType, initializer: list[Initializer]|None = None):
        self.typeId = expectedType
        self.init: list[Initializer] = []

        if initializer is not None:
            self.init = initializer
            # Nothing more to do...
            return

        if isinstance(expectedType, ArrayDeclaratorType):
            # Initializing an array.
            if self.peek().id == "string":
                if not isinstance(expectedType.declarator, BaseDeclaratorType) or \
                not expectedType.declarator.baseType.isCharacter():
                    self.raiseError("Cannot initialize a non char array with a string")

                stringAST = self.createChild(String)
                for ch in stringAST.value:
                    single = SingleInitializer([], None, None, TypeSpecifier.CHAR.toBaseType(),   
                                            CharConstant([], None, None, str(ord(ch))))
                    self.init.append(single)
            else:
                self.expect("{")
                # Parse initializers with the expected "inner" type. For example, if parsing a float[3][4],
                # the first call to compound initializer expects float[4]; the second, float.
                self.init.append(self.createChild(Initializer, expectedType.declarator))
                
                while self.peek().id == ",":
                    self.pop()

                    if self.peek().id == "}":
                        # There can be a trailing comma.
                        break
                    else:
                        self.init.append(self.createChild(Initializer, expectedType.declarator))

                self.expect("}")

            if len(self.init) > expectedType.size:
                self.raiseError(f"Too many initializers: expected {expectedType.size}, received {len(self.init)}")

            if len(self.init) < expectedType.size:
                # Pad the remainder of the array with zeros.
                remainingItems = expectedType.size - len(self.init)
                # Create individual constants.
                for _ in range(remainingItems):
                    self.init.append(CompoundInitializer.createZeroInitializer(expectedType.declarator))

        elif isinstance(expectedType, BaseDeclaratorType) and expectedType.baseType.name == "STRUCT":
            # Initializing a struct.
            self.expect("{")

            # Parse the inner initializers. Must always be more than one. Can be less than the 
            # member count, but never more.
            members = expectedType.baseType.getMembers()
            self.init.append(self.createChild(Initializer, members[0].type))

            argCount: int = 0
            while self.peek().id == ",":
                argCount += 1
                self.pop()

                if self.peek().id == "}":
                    break

                if argCount >= len(members):
                    self.raiseError(f"Too many initializers for type {expectedType}")
                
                self.init.append(self.createChild(Initializer, members[argCount].type))

            self.expect("}")

            if len(self.init) < len(members):
                # Pad the initialized members with zeros.
                for undefinedMember in members[len(self.init):]:
                    self.init.append(CompoundInitializer.createZeroInitializer(undefinedMember.type))

        elif isinstance(expectedType, BaseDeclaratorType) and expectedType.baseType.name == "UNION":
            # Initializing an union.
            self.expect("{")

            # Parse the inner initializers. Must be a single value surrounded by {}.
            members = expectedType.baseType.getMembers()
            self.init.append(self.createChild(Initializer, members[0].type))
            self.expect("}")

        else:
            self.raiseError(f"Cannot create {expectedType} with a compound initializer")

    def staticEval(self) -> list[StaticEvalValue]:
        evalList: list[StaticEvalValue] = []
        for singleInit in self.init:
            evalList.extend(singleInit.staticEval())
        return evalList

    @staticmethod
    def createZeroInitializer(initializerType: DeclaratorType) -> Initializer:
        # Arrays.
        if isinstance(initializerType, ArrayDeclaratorType):
            initializerList: list[Initializer] = []
            for _ in range(initializerType.size):
                initializerList.append(CompoundInitializer.createZeroInitializer(initializerType.declarator))
            return CompoundInitializer([], None, None, initializerType, initializerList)

        # Structs.
        elif isinstance(initializerType, BaseDeclaratorType) and initializerType.baseType.name == "STRUCT":
            initializerList: list[Initializer] = []
            for structMember in initializerType.baseType.getMembers():
                initializerList.append(CompoundInitializer.createZeroInitializer(structMember.type))
            return CompoundInitializer([], None, None, initializerType, initializerList)

        # Unions.
        elif isinstance(initializerType, BaseDeclaratorType) and initializerType.baseType.name == "UNION":
            firstUnionMember = initializerType.baseType.getMembers()[0]
            return CompoundInitializer([], None, None, initializerType, [CompoundInitializer.createZeroInitializer(firstUnionMember.type)])

        # Simple types.
        else:
            return SingleInitializer.createZeroInitializer(initializerType)

    def print(self, padding: int) -> str:
        pad = ' ' * padding
        ret  = f'{pad}{{\n'
        for init in self.init:
            ret += init.print(padding + PADDING_INCREMENT)
        ret += f'{pad}}}\n'
        return ret
    